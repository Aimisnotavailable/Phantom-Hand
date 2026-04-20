# phantom_hand/tracker.py
"""
PhantomHandTracker: MediaPipe hand tracking with kinematic prediction for occlusion handling.
Uses MediaPipe Tasks API and provides ghost frames when hands are temporarily lost.
"""

"""
TO-DO:

FIX HAND ANCHORING
IMPLEMENT A MUCH ROBUST KALMAN FILTER
CREATE ACOUNT ON PYPI

HOURS WASTED : 5
"""
import math
import logging
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Dict, List, Optional, Tuple, Union
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from .config import *

logger = logging.getLogger(__name__)


class PhantomHandTracker:
    """
    A drop-in enhancement for MediaPipe Hands that maintains hand presence
    during temporary occlusions using kinematic prediction and optical flow.

    Args:
        screen_dim: Tuple of (width, height) for coordinate normalization.
        model_path: Optional path to custom MediaPipe hand landmarker model.
        min_detection_confidence: Confidence threshold for hand detection.
        min_tracking_confidence: Confidence threshold for landmark tracking.
        debug: Enable debug logging.
    """

    SOURCE_REAL = "real"
    SOURCE_GHOST = "ghost"

    def __init__(
        self,
        screen_dim: Tuple[int, int] = (1280, 720),
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        debug: bool = False,
    ):
        self.W, self.H = screen_dim
        self.debug = debug

        # Initialize MediaPipe HandLandmarker (Tasks API)
        base_options = python.BaseOptions(
            model_asset_path=model_path if model_path else None
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # Histogram stores hand states over time.
        self.position_histogram: Dict[str, List[Dict]] = {"LEFT": [], "RIGHT": []}
        self.hands_tracker: Dict[str, int] = {"LEFT": 0, "RIGHT": 0}
        self.presence_counter: Dict[str, int] = {"LEFT": 0, "RIGHT": 0}

        # Ghost kinematics & TTL
        self.ghost_velocity: Dict[str, List[float]] = {
            "LEFT": [0.0] * 6,
            "RIGHT": [0.0] * 6,
        }
        self.ghost_ttl_counter: Dict[str, int] = {"LEFT": 0, "RIGHT": 0}
        self.absent_reset_threshold = max(HISTOGRAM_SIZE, MAX_GHOST_TTL + 1)

        # Optical flow (2D pixel tracking)
        self.prev_gray: Optional[np.ndarray] = None
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.lk_tracked_points: Dict[str, Optional[np.ndarray]] = {
            "LEFT": None,
            "RIGHT": None,
        }

        # Cached hand orientation
        self.last_hand_dir: Dict[str, Optional[Tuple[float, float, float]]] = {
            "LEFT": None,
            "RIGHT": None,
        }
        self.last_hand_normal: Dict[str, Optional[Tuple[float, float, float]]] = {
            "LEFT": None,
            "RIGHT": None,
        }

        self.frame_count = 0

        # Output data structure
        self.ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE": {"LEFT": 1.0, "RIGHT": 1.0},
            "FRAME_TYPE": {"LEFT": "REAL", "RIGHT": "REAL"},
            "HAND_PRESENCE": False,
        }

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, level: str, msg: str) -> None:
        if not self.debug and level != "ERROR":
            return
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(msg)

    # ------------------------------------------------------------------
    # Landmark conversion and storage
    # ------------------------------------------------------------------
    def _landmarks_to_list(
        self, landmarks: List[NormalizedLandmark]
    ) -> List[Tuple[float, float, float]]:
        """Convert MediaPipe landmarks to list of (x,y,z) tuples."""
        return [(lm.x, lm.y, lm.z) for lm in landmarks]

    def _store_landmarks(
        self,
        landmarks: List[Tuple[float, float, float]],
        label: str,
        is_generated: bool = False,
    ) -> List[Tuple[float, float, float]]:
        """Store landmarks in histogram and return them."""
        entry = {
            "pts": landmarks,
            "source": self.SOURCE_GHOST if is_generated else self.SOURCE_REAL,
            "frame": self.frame_count,
        }
        if is_generated:
            entry["gen_frame"] = self.frame_count

        hist = self.position_histogram[label]

        if not is_generated:
            if self.hands_tracker[label] >= self.absent_reset_threshold:
                hist.clear()
                hist.append(entry)
            else:
                # Remove trailing ghosts before appending real data
                while hist and hist[-1].get("source") == self.SOURCE_GHOST:
                    hist.pop()
                if len(hist) < HISTOGRAM_SIZE:
                    hist.append(entry)
                else:
                    hist.pop(0)
                    hist.append(entry)
                self.presence_counter[label] = min(
                    PRESENCE_THRESHOLD_ON, self.presence_counter[label] + 1
                )
            self.hands_tracker[label] = 0
        else:
            if len(hist) < HISTOGRAM_SIZE:
                hist.append(entry)
            else:
                hist.pop(0)
                hist.append(entry)

        return landmarks

    # ------------------------------------------------------------------
    # 3D Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if length < 1e-6:
            return (0.0, 0.0, 0.0)
        return (v[0] / length, v[1] / length, v[2] / length)

    @staticmethod
    def _cross(
        a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _dot(
        a: Tuple[float, float, float], b: Tuple[float, float, float]
    ) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _compute_hand_orientation(
        self, pts: List[Tuple[float, float, float]]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return (hand_dir, normal) from landmark points."""
        wrist = pts[WRIST_IDX]
        mcp_mid = pts[MIDDLE_MCP_IDX]
        hand_dir = (
            mcp_mid[0] - wrist[0],
            mcp_mid[1] - wrist[1],
            mcp_mid[2] - wrist[2],
        )
        hand_dir = self._normalize(hand_dir)

        idx_mcp = pts[INDEX_MCP_IDX]
        pinky_mcp = pts[PINKY_MCP_IDX]
        v1 = (idx_mcp[0] - wrist[0], idx_mcp[1] - wrist[1], idx_mcp[2] - wrist[2])
        v2 = (
            pinky_mcp[0] - wrist[0],
            pinky_mcp[1] - wrist[1],
            pinky_mcp[2] - wrist[2],
        )
        normal = self._cross(v1, v2)
        normal = self._normalize(normal)
        return hand_dir, normal

    def _rotate_point(
        self,
        p: Tuple[float, float, float],
        center: Tuple[float, float, float],
        axis_angle: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Rotate point p around center by axis-angle vector."""
        angle = math.sqrt(axis_angle[0] ** 2 + axis_angle[1] ** 2 + axis_angle[2] ** 2)
        if angle < 1e-6:
            return p
        ax = axis_angle[0] / angle
        ay = axis_angle[1] / angle
        az = axis_angle[2] / angle

        px = p[0] - center[0]
        py = p[1] - center[1]
        pz = p[2] - center[2]

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dot = ax * px + ay * py + az * pz
        cross_x = ay * pz - az * py
        cross_y = az * px - ax * pz
        cross_z = ax * py - ay * px

        rx = px * cos_a + cross_x * sin_a + ax * dot * (1 - cos_a)
        ry = py * cos_a + cross_y * sin_a + ay * dot * (1 - cos_a)
        rz = pz * cos_a + cross_z * sin_a + az * dot * (1 - cos_a)

        return (rx + center[0], ry + center[1], rz + center[2])

    # ------------------------------------------------------------------
    # Velocity estimation (3D kinematics)
    # ------------------------------------------------------------------
    def _calculate_velocity(
        self, label: str, window: int = 4
    ) -> List[float]:
        """Return [vx, vy, vz, ax, ay, az] normalized velocities."""
        hist = self.position_histogram[label]
        valid = []
        frames = []
        for e in reversed(hist):
            if e.get("source") != self.SOURCE_REAL:
                continue
            pts = e.get("pts", [])
            if len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX, INDEX_MCP_IDX, PINKY_MCP_IDX):
                wrist = pts[WRIST_IDX]
                if wrist:
                    hand_dir, normal = self._compute_hand_orientation(pts)
                    valid.append((wrist, hand_dir, normal))
                    frames.append(int(e.get("frame", self.frame_count)))
                    if len(valid) >= window:
                        break

        if len(valid) < 2:
            return [0.0] * 6

        lin_vels = []
        ang_vels = []
        for i in range(len(valid) - 1):
            w_new, d_new, n_new = valid[i]
            w_old, d_old, n_old = valid[i + 1]
            f_new = frames[i]
            f_old = frames[i + 1]
            df = max(1, f_new - f_old)

            vx = (w_new[0] - w_old[0]) / df
            vy = (w_new[1] - w_old[1]) / df
            vz = (w_new[2] - w_old[2]) / df
            lin_vels.append((vx, vy, vz))

            # Angular velocity (approximate)
            axis1 = self._cross(d_old, d_new)
            sin_angle1 = math.sqrt(axis1[0] ** 2 + axis1[1] ** 2 + axis1[2] ** 2)
            cos_angle1 = self._dot(d_old, d_new)
            angle1 = math.atan2(sin_angle1, cos_angle1)
            if sin_angle1 > 0:
                axis1 = (
                    axis1[0] / sin_angle1,
                    axis1[1] / sin_angle1,
                    axis1[2] / sin_angle1,
                )
            else:
                axis1 = (0.0, 1.0, 0.0)

            axis2 = self._cross(n_old, n_new)
            sin_angle2 = math.sqrt(axis2[0] ** 2 + axis2[1] ** 2 + axis2[2] ** 2)
            cos_angle2 = self._dot(n_old, n_new)
            angle2 = math.atan2(sin_angle2, cos_angle2)
            if sin_angle2 > 0:
                axis2 = (
                    axis2[0] / sin_angle2,
                    axis2[1] / sin_angle2,
                    axis2[2] / sin_angle2,
                )
            else:
                axis2 = axis1

            ang_vel = (
                (axis1[0] * angle1 + axis2[0] * angle2) * 0.5,
                (axis1[1] * angle1 + axis2[1] * angle2) * 0.5,
                (axis1[2] * angle1 + axis2[2] * angle2) * 0.5,
            )
            ang_vels.append(ang_vel)

        avg_vx = sum(v[0] for v in lin_vels) / len(lin_vels)
        avg_vy = sum(v[1] for v in lin_vels) / len(lin_vels)
        avg_vz = sum(v[2] for v in lin_vels) / len(lin_vels)
        avg_ax = sum(a[0] for a in ang_vels) / len(ang_vels)
        avg_ay = sum(a[1] for a in ang_vels) / len(ang_vels)
        avg_az = sum(a[2] for a in ang_vels) / len(ang_vels)

        # Clamp angular speed
        ang_speed = math.sqrt(avg_ax**2 + avg_ay**2 + avg_az**2)
        if ang_speed > MAX_ANGULAR_SPEED:
            scale = MAX_ANGULAR_SPEED / ang_speed
            avg_ax *= scale
            avg_ay *= scale
            avg_az *= scale

        # Clamp linear speed
        lin_speed = math.sqrt(avg_vx**2 + avg_vy**2 + avg_vz**2)
        if lin_speed > MAX_LINEAR_SPEED_NORM:
            scale = MAX_LINEAR_SPEED_NORM / lin_speed
            avg_vx *= scale
            avg_vy *= scale
            avg_vz *= scale

        return [avg_vx, avg_vy, avg_vz, avg_ax, avg_ay, avg_az]

    # ------------------------------------------------------------------
    # Optical flow velocity estimation
    # ------------------------------------------------------------------
    def _compute_flow_velocity(
        self, old_points: np.ndarray, new_points: np.ndarray, label: str
    ) -> List[float]:
        """
        old_points, new_points: numpy arrays of shape (n,1,2) in pixel coordinates.
        Returns [vx, vy, vz, ax, ay, az] in normalized units.
        """
        old_wrist = old_points[0, 0]
        new_wrist = new_points[0, 0]
        dx_px = new_wrist[0] - old_wrist[0]
        dy_px = new_wrist[1] - old_wrist[1]
        dx_norm = dx_px / self.W
        dy_norm = dy_px / self.H

        # Estimate depth change from scale change
        old_dists = []
        new_dists = []
        for i in range(1, len(old_points)):
            p_old = old_points[i, 0]
            p_new = new_points[i, 0]
            old_d = math.hypot(p_old[0] - old_wrist[0], p_old[1] - old_wrist[1])
            new_d = math.hypot(p_new[0] - new_wrist[0], p_new[1] - new_wrist[1])
            if old_d > 0:
                old_dists.append(old_d)
                new_dists.append(new_d)

        if old_dists:
            median_old = np.median(old_dists)
            median_new = np.median(new_dists)
            scale_factor = median_new / median_old if median_old > 0 else 1.0
            scale_factor = max(
                1.0 - MAX_SCALE_CHANGE, min(1.0 + MAX_SCALE_CHANGE, scale_factor)
            )
            dz_norm = -(scale_factor - 1.0) * 0.1  # heuristic
        else:
            dz_norm = 0.0

        # Estimate 2D rotation (around Z)
        angles = []
        for i in range(1, len(old_points)):
            v_old = old_points[i, 0] - old_wrist
            v_new = new_points[i, 0] - new_wrist
            if not (v_old[0] == 0 and v_old[1] == 0) and not (
                v_new[0] == 0 and v_new[1] == 0
            ):
                ang = math.atan2(v_new[1], v_new[0]) - math.atan2(
                    v_old[1], v_old[0]
                )
                angles.append(ang)
        angle2d = np.median(angles) if angles else 0.0

        return [dx_norm, dy_norm, dz_norm, 0.0, 0.0, angle2d]

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def _check_drift(
        self, label: str, new_wrist_norm: Tuple[float, float, float]
    ) -> bool:
        hist = self.position_histogram[label]
        if len(hist) < 2:
            return False
        last_entry = hist[-1]
        if last_entry.get("source") != self.SOURCE_GHOST:
            return False
        last_pts = last_entry.get("pts", [])
        if len(last_pts) <= WRIST_IDX:
            return False
        last_wrist = last_pts[WRIST_IDX]

        vx, vy, vz, _, _, _ = self.ghost_velocity[label]
        predicted = (last_wrist[0] + vx, last_wrist[1] + vy, last_wrist[2] + vz)
        dist = math.hypot(
            predicted[0] - new_wrist_norm[0], predicted[1] - new_wrist_norm[1]
        )
        if dist > DRIFT_THRESHOLD:
            self._log("DEBUG", f"Drift detected for {label}: dist={dist:.3f}")
            return True
        return False

    # ------------------------------------------------------------------
    # Ghost frame generation
    # ------------------------------------------------------------------
    def _generate_ghost_frame(self, velocity: List[float], label: str):
        """Create a synthetic hand landmark list based on velocity."""
        hist = self.position_histogram[label]
        if hist:
            base_pts = hist[-1].get("pts", [])
        else:
            base_pts = []

        if not base_pts:
            return None

        vx, vy, vz, ax, ay, az = velocity

        # Limit rotation per frame
        ang_speed = math.sqrt(ax**2 + ay**2 + az**2)
        if ang_speed > 0.35:
            scale = 0.35 / ang_speed
            ax *= scale
            ay *= scale
            az *= scale

        base_wrist = base_pts[WRIST_IDX] if len(base_pts) > WRIST_IDX else None

        ghost_pts = []
        for p in base_pts:
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            if base_wrist is not None:
                rx, ry, rz = self._rotate_point(
                    (px, py, pz), base_wrist, (ax, ay, az)
                )
                new_x = rx + vx
                new_y = ry + vy
                new_z = rz + vz
            else:
                new_x = px + vx
                new_y = py + vy
                new_z = pz + vz
            ghost_pts.append((new_x, new_y, new_z))

        return ghost_pts

    # ------------------------------------------------------------------
    # Handedness reconciliation
    # ------------------------------------------------------------------
    def _reconcile_handedness(
        self, detections: List[Tuple[str, Optional[Tuple[float, float]], List]]
    ) -> None:
        """Correct MediaPipe's left/right labels using spatial continuity."""
        if not detections:
            return

        # Build list of valid blobs (with wrist pixel positions)
        blobs = []
        for i, (orig_label, wrist_px, _) in enumerate(detections):
            if wrist_px is not None:
                blobs.append((i, orig_label, wrist_px))

        if not blobs:
            return

        tracked_px = {}
        for label in ("LEFT", "RIGHT"):
            hist = self.position_histogram[label]
            wrist = None
            for e in reversed(hist):
                if e.get("source") == self.SOURCE_REAL:
                    pts = e.get("pts", [])
                    if len(pts) > WRIST_IDX:
                        wrist = pts[WRIST_IDX]
                        break
            if wrist is not None:
                tracked_px[label] = (wrist[0] * self.W, wrist[1] * self.H)
            else:
                tracked_px[label] = None

        MAX_JUMP = max(self.W, self.H) * 0.10
        assigned_blobs = set()
        assigned_labels = set()
        final_assignments = {}

        # Pair blobs with tracked labels by proximity
        pairs = []
        for b_idx, orig_label, wrist_px in blobs:
            for label, last_pos in tracked_px.items():
                if last_pos is not None:
                    dist = math.hypot(
                        wrist_px[0] - last_pos[0], wrist_px[1] - last_pos[1]
                    )
                    if orig_label != label:
                        frames_absent = self.hands_tracker[label]
                        track_len = len(self.position_histogram[label])
                        if frames_absent > 2 or track_len < 5:
                            dist += MAX_JUMP * 2.0
                        else:
                            dist += MAX_JUMP * 0.6
                    if dist < MAX_JUMP:
                        pairs.append((dist, b_idx, label))

        pairs.sort(key=lambda x: x[0])
        for dist, b_idx, label in pairs:
            if b_idx not in assigned_blobs and label not in assigned_labels:
                final_assignments[b_idx] = label
                assigned_blobs.add(b_idx)
                assigned_labels.add(label)

        unassigned = [b for b in blobs if b[0] not in assigned_blobs]

        # Fallback assignment
        if len(unassigned) == 2 and len(assigned_labels) == 0:
            a, b = unassigned
            if a[1] != b[1] and a[1] in ("LEFT", "RIGHT") and b[1] in ("LEFT", "RIGHT"):
                final_assignments[a[0]] = a[1]
                final_assignments[b[0]] = b[1]
            else:
                # Sort by x coordinate
                unassigned.sort(key=lambda x: x[2][0])
                final_assignments[unassigned[0][0]] = "LEFT"
                final_assignments[unassigned[1][0]] = "RIGHT"
        else:
            for b_idx, orig_label, wrist_px in unassigned:
                avail = [L for L in ("LEFT", "RIGHT") if L not in assigned_labels]
                if not avail:
                    break
                if orig_label in avail:
                    label = orig_label
                elif len(avail) == 1:
                    label = avail[0]
                else:
                    label = "LEFT" if wrist_px[0] < self.W / 2 else "RIGHT"
                final_assignments[b_idx] = label
                assigned_labels.add(label)

        # Apply assignments
        new_detections = []
        for i, (orig_label, wrist_px, lm_list) in enumerate(detections):
            if i in final_assignments:
                new_label = final_assignments[i]
                if new_label != orig_label:
                    self._log(
                        "DEBUG",
                        f"Handedness override: {orig_label} -> {new_label}",
                    )
                new_detections.append((new_label, wrist_px, lm_list))
        detections[:] = new_detections

    # ------------------------------------------------------------------
    # Main update method
    # ------------------------------------------------------------------
    def update(self, frame: np.ndarray) -> Dict:
        """
        Process a new video frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Dictionary containing:
                "POSITION_DATA": {"LEFT": list of (x,y,z), "RIGHT": ...}
                "SCALE": {"LEFT": float, "RIGHT": float}
                "FRAME_TYPE": {"LEFT": "REAL"/"GHOST", "RIGHT": ...}
                "HAND_PRESENCE": bool
        """
        if frame is None:
            self._log("ERROR", "Frame is None")
            return self.ar_data

        self.frame_count += 1
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize output
        ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE": {"LEFT": 1.0, "RIGHT": 1.0},
            "FRAME_TYPE": {"LEFT": "REAL", "RIGHT": "REAL"},
            "HAND_PRESENCE": False,
        }

        # Run MediaPipe HandLandmarker
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect(mp_image)

        # Prune old ghosts
        for label in ("LEFT", "RIGHT"):
            hist = self.position_histogram[label]
            for e in hist[:]:
                if e.get("source") == self.SOURCE_GHOST:
                    gen_frame = e.get("gen_frame", e.get("frame", self.frame_count))
                    age = self.frame_count - int(gen_frame)
                    if age >= GHOST_AGE_DEFAULT:
                        hist.remove(e)
            while len(hist) > HISTOGRAM_SIZE:
                hist.pop(0)

        seen = set()
        detections = []

        # Process MediaPipe detections
        if result.hand_landmarks:
            for i, (landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                label = handedness[0].category_name.upper()
                # Get wrist pixel position
                wrist = landmarks[WRIST_IDX]
                wrist_px = (wrist.x * self.W, wrist.y * self.H)
                detections.append((label, wrist_px, landmarks))

            # Correct handedness
            if len(detections) > 0:
                self._reconcile_handedness(detections)

            # Process each detected hand
            for label, wrist_px, landmarks in detections:
                seen.add(label)

                # Convert landmarks to list of tuples
                pts = self._landmarks_to_list(landmarks)
                # Store in histogram
                stored_pts = self._store_landmarks(pts, label, is_generated=False)

                # Cache orientation
                if len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX, INDEX_MCP_IDX, PINKY_MCP_IDX):
                    hand_dir, normal = self._compute_hand_orientation(pts)
                    self.last_hand_dir[label] = hand_dir
                    self.last_hand_normal[label] = normal

                # Update optical flow tracking points
                flow_pts_px = []
                for idx in FLOW_POINT_INDICES:
                    if idx < len(pts):
                        x_norm, y_norm, _ = pts[idx]
                        flow_pts_px.append([x_norm * self.W, y_norm * self.H])
                if len(flow_pts_px) == len(FLOW_POINT_INDICES):
                    self.lk_tracked_points[label] = np.array(
                        flow_pts_px, dtype=np.float32
                    ).reshape(-1, 1, 2)

                # Remove ghosts when real hand appears
                hist = self.position_histogram[label]
                pure_real = [e for e in hist if e.get("source") == self.SOURCE_REAL]
                if len(pure_real) != len(hist):
                    hist.clear()
                    hist.extend(pure_real)

                # Compute hand scale
                hand_scale = 1.0
                if len(pts) > MIDDLE_MCP_IDX:
                    wrist = pts[WRIST_IDX]
                    mcp = pts[MIDDLE_MCP_IDX]
                    dx = mcp[0] - wrist[0]
                    dy = mcp[1] - wrist[1]
                    dz = mcp[2] - wrist[2]
                    hand_scale = math.sqrt(dx**2 + dy**2 + dz**2) or 1.0

                ar_data["POSITION_DATA"][label] = stored_pts
                ar_data["FRAME_TYPE"][label] = "REAL"
                ar_data["SCALE"][label] = hand_scale
                self.hands_tracker[label] = 0

        # Handle missing hands (generate ghosts)
        for label in ("LEFT", "RIGHT"):
            if label not in seen:
                self.hands_tracker[label] += 1
                self.presence_counter[label] = max(
                    PRESENCE_THRESHOLD_OFF, self.presence_counter[label] - 1
                )

                ghost_pts = []

                if self.hands_tracker[label] == 1:
                    # Just lost: compute initial velocity
                    real_hist = [
                        e
                        for e in self.position_histogram[label]
                        if e.get("source") == self.SOURCE_REAL
                    ]
                    if len(real_hist) >= 3:
                        self.ghost_velocity[label] = self._calculate_velocity(label)
                        self.ghost_ttl_counter[label] = MAX_GHOST_TTL
                    else:
                        self.ghost_velocity[label] = [0.0] * 6
                        self.ghost_ttl_counter[label] = 0
                        self.lk_tracked_points[label] = None

                if (
                    self.ghost_ttl_counter[label] > 0
                    and self.hands_tracker[label] < self.absent_reset_threshold
                ):
                    flow_success = False
                    measured_vel = None

                    # Attempt optical flow
                    if (
                        self.lk_tracked_points[label] is not None
                        and self.prev_gray is not None
                        and len(self.lk_tracked_points[label]) >= 2
                    ):
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray,
                            current_gray,
                            self.lk_tracked_points[label],
                            None,
                            **self.lk_params,
                        )
                        if st is not None and np.all(st == 1):
                            measured_vel = self._compute_flow_velocity(
                                self.lk_tracked_points[label], p1, label
                            )
                            new_wrist_px = p1[0, 0]
                            new_wrist_norm = (
                                new_wrist_px[0] / self.W,
                                new_wrist_px[1] / self.H,
                                0.0,
                            )
                            if not self._check_drift(label, new_wrist_norm):
                                flow_success = True
                                self.lk_tracked_points[label] = p1
                                self._log("DEBUG", f"Optical flow tracked {label}")
                            else:
                                self._log("DEBUG", f"Flow drifted for {label}")

                    if flow_success and measured_vel is not None:
                        # Smooth velocity with EMA
                        old_vel = self.ghost_velocity[label]
                        smoothed = [
                            EMA_ALPHA * measured_vel[i] + (1 - EMA_ALPHA) * old_vel[i]
                            for i in range(6)
                        ]
                        self.ghost_velocity[label] = smoothed
                    else:
                        # Kinematic decay
                        vx, vy, vz, ax, ay, az = self.ghost_velocity[label]
                        speed = math.sqrt(vx**2 + vy**2 + vz**2)
                        frames_lost = self.hands_tracker[label]

                        if frames_lost <= CONSTANT_VELOCITY_FRAMES:
                            pass  # constant velocity
                        else:
                            if speed < 0.001:
                                self.ghost_velocity[label] = [0.0] * 6
                                self.ghost_ttl_counter[label] = min(
                                    3, self.ghost_ttl_counter[label]
                                )
                            else:
                                self.ghost_velocity[label][0] *= KINEMATIC_FRICTION
                                self.ghost_velocity[label][1] *= KINEMATIC_FRICTION
                                self.ghost_velocity[label][2] *= KINEMATIC_FRICTION
                                self.ghost_velocity[label][3] *= (
                                    KINEMATIC_FRICTION * 0.7
                                )
                                self.ghost_velocity[label][4] *= (
                                    KINEMATIC_FRICTION * 0.7
                                )
                                self.ghost_velocity[label][5] *= (
                                    KINEMATIC_FRICTION * 0.7
                                )
                                self._log("DEBUG", f"Kinematic decay for {label}")

                        self.lk_tracked_points[label] = None

                    # Generate ghost
                    ghost_pts = self._generate_ghost_frame(
                        self.ghost_velocity[label], label
                    )
                    if ghost_pts:
                        ghost_pts = self._store_landmarks(
                            ghost_pts, label, is_generated=True
                        )

                    self.ghost_ttl_counter[label] -= 1
                else:
                    self.ghost_velocity[label] = [0.0] * 6
                    if self.hands_tracker[label] > MAX_GHOST_TTL:
                        self.position_histogram[label].clear()
                        self.lk_tracked_points[label] = None

                if ghost_pts:
                    ar_data["FRAME_TYPE"][label] = "GHOST"
                    ar_data["POSITION_DATA"][label] = ghost_pts
                else:
                    ar_data["FRAME_TYPE"][label] = "REAL"
                    ar_data["POSITION_DATA"][label] = []
                ar_data["SCALE"][label] = 1.0

        ar_data["HAND_PRESENCE"] = any(
            self.presence_counter[label] >= PRESENCE_THRESHOLD_ON
            for label in ("LEFT", "RIGHT")
        )

        self.ar_data = ar_data
        self.prev_gray = current_gray
        return ar_data

    def close(self):
        """Release MediaPipe resources."""
        self.hand_landmarker.close()
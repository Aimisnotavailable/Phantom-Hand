
---
# 👻 PhantomHand

[![PyPI version](https://badge.fury.io/py/phantom-hand-tracker.svg)](https://badge.fury.io/py/phantom-hand-tracker)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/phantom-hand-tracker)](https://pepy.tech/project/phantom-hand-tracker)

**MediaPipe hand tracking that refuses to let go.**

PhantomHand enhances MediaPipe's powerful hand landmarker with a short‑term kinematic memory. When your hand is temporarily occluded—by another hand, an object, poor lighting, or motion blur—PhantomHand generates **ghost frames**: physically plausible predictions of where your hand *should* be. Your application sees a continuous stream of hand data, eliminating jarring dropouts and jitter.

## 📦 Installation

```bash
pip install phantom-hand-tracker
```

PhantomHand requires Python 3.8 or later and automatically installs `opencv-python`, `numpy`, and `mediapipe` (Tasks API).

## 🚀 Quick Start

```python
import cv2
from phantom_hand import PhantomHandTracker

tracker = PhantomHandTracker(screen_dim=(1280, 720))
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    data = tracker.update(frame)

    for hand in ("LEFT", "RIGHT"):
        pts = data["POSITION_DATA"][hand]
        is_ghost = data["FRAME_TYPE"][hand] == "GHOST"
        color = (255, 0, 0) if is_ghost else (0, 255, 0)  # Blue = ghost, Green = real

        for x, y, _ in pts:
            px, py = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (px, py), 5, color, -1)

    cv2.imshow("PhantomHand", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

tracker.close()
cap.release()
cv2.destroyAllWindows()
```

## 🎥 Demo Videos

| Normal Tracking | Occlusion Handling |
|-----------------|---------------------|
| ![Normal tracking](docs/normal.gif) | ![Occlusion handling](docs/occlusion.gif) |

*Left: MediaPipe loses the hand behind paper. Right: PhantomHand's blue ghost maintains continuity.*

*(Replace these placeholder GIF paths with actual demo recordings.)*

## 🧠 How It Works

PhantomHand wraps MediaPipe's `HandLandmarker` (Tasks API) and adds a temporal state layer:

1. **Velocity Estimation** – Computes 3D linear and angular velocity from recent real frames.
2. **Optical Flow Validation** – Tracks key points (wrist, MCPs, tips) with Lucas‑Kanade to correct drift.
3. **Kinematic Prediction** – Generates ghost frames using constant‑velocity physics with exponential decay.
4. **Graceful Handoff** – When MediaPipe re‑acquires the hand, ghosts are smoothly replaced by real landmarks.

## 📄 API Reference

### `PhantomHandTracker`

```python
tracker = PhantomHandTracker(
    screen_dim=(1280, 720),
    model_path=None,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    debug=False
)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `screen_dim` | `tuple` | `(1280, 720)` | (width, height) for coordinate normalization |
| `model_path` | `str` | `None` | Path to custom MediaPipe `.task` model file |
| `min_detection_confidence` | `float` | `0.7` | Minimum confidence for hand detection |
| `min_tracking_confidence` | `float` | `0.5` | Minimum confidence for landmark tracking |
| `debug` | `bool` | `False` | Enable debug logging |

### `update(frame: np.ndarray) -> dict`

Process a BGR frame and return hand data.

**Returns:**
```python
{
    "POSITION_DATA": {
        "LEFT":  [(x, y, z), ...],   # 21 landmarks, normalized [0,1]
        "RIGHT": [(x, y, z), ...]
    },
    "FRAME_TYPE": {
        "LEFT":  "REAL" | "GHOST",
        "RIGHT": "REAL" | "GHOST"
    },
    "SCALE": {
        "LEFT":  float,   # Estimated hand size (wrist → middle MCP)
        "RIGHT": float
    },
    "HAND_PRESENCE": bool   # True if at least one hand is active
}
```

### `close()`

Release MediaPipe resources. Always call this when finished.

## ⚙️ Configuration

All tunable constants live in `phantom_hand/config.py`. You can modify them directly or subclass `PhantomHandTracker` to override.

| Constant | Default | Description |
|----------|---------|-------------|
| `MAX_GHOST_TTL` | `15` | Maximum frames a ghost persists without real detection |
| `KINEMATIC_FRICTION` | `0.85` | Velocity multiplier per frame after `CONSTANT_VELOCITY_FRAMES` |
| `CONSTANT_VELOCITY_FRAMES` | `3` | Frames to maintain full velocity before decay begins |
| `DRIFT_THRESHOLD` | `0.05` | Normalized distance before ghost is considered unreliable |
| `EMA_ALPHA` | `0.3` | Smoothing factor for velocity updates |
| `HISTOGRAM_SIZE` | `10` | Number of historical frames kept for velocity calculation |

## 📁 Project Structure

```
phantom_hand/
├── __init__.py          # Package entry, exposes PhantomHandTracker
├── tracker.py           # Main tracking logic
├── config.py            # Constants and landmark indices
examples/
├── basic_demo.py        # Minimal usage example
```

## 🧪 Advanced Usage

### Custom Gesture Detection with Ghost Frames

```python
from phantom_hand import PhantomHandTracker

tracker = PhantomHandTracker(debug=True)

while True:
    data = tracker.update(frame)

    for hand in ("LEFT", "RIGHT"):
        pts = data["POSITION_DATA"][hand]
        if not pts:
            continue

        # Your gesture logic here
        thumb_tip = pts[4]
        index_tip = pts[8]
        distance = ((thumb_tip[0] - index_tip[0])**2 + 
                    (thumb_tip[1] - index_tip[1])**2) ** 0.5

        is_pinch = distance < 0.05
        is_ghost = data["FRAME_TYPE"][hand] == "GHOST"

        if is_pinch and not is_ghost:
            # Trigger action only on real pinch
            print(f"{hand} hand pinched!")

tracker.close()
```

### Changing MediaPipe Model

You can download custom `.task` models from [MediaPipe's model card](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and pass the path:

```python
tracker = PhantomHandTracker(model_path="hand_landmarker_full.task")
```

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe.tasks'"
Ensure you have MediaPipe ≥ 0.10.0:
```bash
pip install --upgrade mediapipe
```

### Ghost hand drifts too far
- Reduce `MAX_GHOST_TTL` in `config.py`.
- Increase `DRIFT_THRESHOLD` to make drift detection stricter.
- Ensure your camera is not moving rapidly (optical flow assumes a static scene).

### Performance is slow
- Reduce frame resolution before passing to `update()`.
- Set `debug=False` to disable logging overhead.
- Consider running PhantomHand in a separate thread.

## 🤝 Contributing

Contributions are welcome! If you encounter a case where the ghost drifts significantly or fails to recover, please open an issue with:
- A short video clip showing the failure.
- Your camera resolution and frame rate.
- Any custom configuration you're using.

Pull requests for improved kinematics, additional validation, or better documentation are appreciated.

## 📜 License

MIT © 2025 Your Name

## 🙏 Acknowledgements

PhantomHand is built on top of Google's incredible [MediaPipe](https://developers.google.com/mediapipe) framework. The ghost prediction approach is inspired by classical Kalman filtering techniques used in VR/AR hand tracking systems.

---

**PhantomHand: Because hands shouldn't just vanish.**

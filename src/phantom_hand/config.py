# ghost_hand_tracker/config.py
"""Configuration constants for GhostHandTracker."""

# MediaPipe landmark indices (standard 21-point hand skeleton)
WRIST_IDX = 0
THUMB_CMC_IDX = 1
THUMB_MCP_IDX = 2
THUMB_IP_IDX = 3
THUMB_TIP_IDX = 4
INDEX_MCP_IDX = 5
INDEX_PIP_IDX = 6
INDEX_DIP_IDX = 7
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9
MIDDLE_PIP_IDX = 10
MIDDLE_DIP_IDX = 11
MIDDLE_TIP_IDX = 12
RING_MCP_IDX = 13
RING_PIP_IDX = 14
RING_DIP_IDX = 15
RING_TIP_IDX = 16
PINKY_MCP_IDX = 17
PINKY_PIP_IDX = 18
PINKY_DIP_IDX = 19
PINKY_TIP_IDX = 20

# Histogram size for temporal smoothing
HISTOGRAM_SIZE = 10

# Drift detection threshold (normalized distance)
DRIFT_THRESHOLD = 0.05

# Landmark indices used for optical flow tracking
FLOW_POINT_INDICES = [WRIST_IDX, INDEX_MCP_IDX, PINKY_MCP_IDX, INDEX_TIP_IDX, PINKY_TIP_IDX]

# Velocity smoothing factor (EMA)
EMA_ALPHA = 0.3

# Frames to maintain constant velocity after loss
CONSTANT_VELOCITY_FRAMES = 3

# Maximum allowed scale factor change per frame (for depth estimation)
MAX_SCALE_CHANGE = 0.5

# Ghost lifespan (frames)
MAX_GHOST_TTL = 15
GHOST_AGE_DEFAULT = 2

# Kinematic friction coefficient
KINEMATIC_FRICTION = 0.85

# Velocity clamping
MAX_LINEAR_SPEED_NORM = 0.1
MAX_ANGULAR_SPEED = 0.5

# Hand presence hysteresis
PRESENCE_THRESHOLD_ON = 2
PRESENCE_THRESHOLD_OFF = -2
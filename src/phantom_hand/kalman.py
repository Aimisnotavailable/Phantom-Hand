class WristKalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        # State: x, y, z, vx, vy, vz
        self.x = np.zeros((6, 1))
        # Initial covariance
        self.P = np.eye(6) * 1000.0

        # State transition (constant velocity)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ])

        # Measurement matrix (we observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise (tune these values)
        q_pos = 0.0001
        q_vel = 0.001
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        # Measurement noise (position uncertainty from MediaPipe)
        r_pos = 0.0001
        self.R = np.diag([r_pos, r_pos, r_pos])

        # For prediction-only steps
        self.predicted_state = self.x.copy()
        self.predicted_P = self.P.copy()

    def predict(self):
        self.predicted_state = self.F @ self.x
        self.predicted_P = self.F @ self.P @ self.F.T + self.Q
        # For ghost frames we'll use predicted_state

    def update(self, measurement_xyz):
        """Measurement_xyz is (3,) array of wrist position."""
        z = np.array(measurement_xyz).reshape(3, 1)
        y = z - self.H @ self.predicted_state
        S = self.H @ self.predicted_P @ self.H.T + self.R
        K = self.predicted_P @ self.H.T @ np.linalg.inv(S)
        self.x = self.predicted_state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.predicted_P

    def get_velocity(self):
        return self.x[3:6].flatten()

    def get_position(self):
        return self.x[0:3].flatten()
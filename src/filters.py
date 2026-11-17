"""Kalman filters used in the notebooks and in scripts/run_flight.py.

Plain numpy, no filterpy. The update uses the Joseph form, which matches
filterpy's KalmanFilter numerically, so results are the same as in the
original notebooks.

All three filters take the noisy GPS array (N, 3) with NaN rows during
dropouts, plus the boolean `valid` mask, and return (est, sig):
    est  (N, n_state)  state estimate at every step
    sig  (N, n_state)  filter's own 1-sigma uncertainty per state

State layouts:
    run_cv_kf        [x, y, vx, vy]
    run_imu_kf       [x, y, vx, vy]
    run_imu_bias_kf  [x, y, vx, vy, bx, by]   (b = accelerometer bias)
"""
import numpy as np

from sensors import rpy_to_rotmat, GRAVITY


class LinearKF:
    """Minimal linear Kalman filter with an optional control input."""

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = None if B is None else np.asarray(B, dtype=float)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()

    def predict(self, u=None):
        self.x = self.F @ self.x
        if u is not None and self.B is not None:
            self.x = self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = np.linalg.solve(S, self.H @ self.P).T        # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form


def a_world_from_imu(accel, rpy):
    """World-frame acceleration from body-frame accelerometer readings.

    Undoes what sensors.make_imu does: a_world = R @ accel + g.
    Uses the TRUE attitude (rpy), so this is still a cheat until the EKF
    estimates attitude itself.
    """
    R = rpy_to_rotmat(rpy)
    return np.einsum("nij,nj->ni", R, accel) + GRAVITY


def _first_fix(gps, valid):
    i = int(np.flatnonzero(valid)[0])
    return gps[i, 0], gps[i, 1]


def _run(kf, gps, valid, a_world=None):
    n = len(gps)
    k = len(kf.x)
    est = np.zeros((n, k))
    sig = np.zeros((n, k))
    for i in range(n):
        if a_world is None:
            kf.predict()
        else:
            kf.predict(u=a_world[i, :2])
        if valid[i]:
            kf.update(gps[i, :2])
        est[i] = kf.x
        sig[i] = np.sqrt(np.diag(kf.P))
    return est, sig


def _F_cv(dt):
    return np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)


def _B_accel(dt, n_state):
    B = np.zeros((n_state, 2))
    B[0, 0] = B[1, 1] = 0.5 * dt**2
    B[2, 0] = B[3, 1] = dt
    return B


def run_cv_kf(gps, valid, dt, q=1e-4, gps_sigma=0.03):
    """Constant-velocity filter, GPS only. q = velocity process noise."""
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
    x0 = np.array([*_first_fix(gps, valid), 0.0, 0.0])
    kf = LinearKF(F=_F_cv(dt), H=H,
                  Q=np.diag([0, 0, q, q]),
                  R=np.eye(2) * gps_sigma**2,
                  x0=x0,
                  P0=np.diag([gps_sigma**2, gps_sigma**2, 0.2**2, 0.2**2]))
    return _run(kf, gps, valid)


def run_imu_kf(a_world, gps, valid, dt, q=None, gps_sigma=0.03):
    """IMU acceleration drives the predict step; no bias in the state."""
    q = (0.05 * dt) ** 2 if q is None else q
    H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
    x0 = np.array([*_first_fix(gps, valid), 0.0, 0.0])
    kf = LinearKF(F=_F_cv(dt), H=H, B=_B_accel(dt, 4),
                  Q=np.diag([0, 0, q, q]),
                  R=np.eye(2) * gps_sigma**2,
                  x0=x0,
                  P0=np.diag([gps_sigma**2, gps_sigma**2, 0.2**2, 0.2**2]))
    return _run(kf, gps, valid, a_world)


def run_imu_bias_kf(a_world, gps, valid, dt, q_vel=None, q_bias=None,
                    gps_sigma=0.03):
    """IMU-driven filter with the accelerometer bias in the state.

    True acceleration = measured acceleration - bias, so the bias columns of F
    subtract 0.5*dt^2*b from position and dt*b from velocity. The bias itself
    is a random walk (identity rows). It is only observable while GPS is
    available.
    """
    q_vel = (0.05 * dt) ** 2 if q_vel is None else q_vel
    q_bias = 0.01**2 * dt if q_bias is None else q_bias

    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[0, 4] = F[1, 5] = -0.5 * dt**2
    F[2, 4] = F[3, 5] = -dt

    H = np.zeros((2, 6)); H[0, 0] = H[1, 1] = 1
    x0 = np.array([*_first_fix(gps, valid), 0, 0, 0, 0], dtype=float)
    kf = LinearKF(F=F, H=H, B=_B_accel(dt, 6),
                  Q=np.diag([0, 0, q_vel, q_vel, q_bias, q_bias]),
                  R=np.eye(2) * gps_sigma**2,
                  x0=x0,
                  P0=np.diag([gps_sigma**2, gps_sigma**2, 0.2**2, 0.2**2,
                              0.05**2, 0.05**2]))
    return _run(kf, gps, valid, a_world)


def rmse(err, mask=None):
    """Root mean squared error, optionally over a boolean mask."""
    err = np.asarray(err)
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(err**2)))

"""Sensor layer: turns ground truth from fly_pid.run() into noisy measurements.

GPS: true position + white noise, and NaN inside dropout zones.
IMU: accelerometer (specific force, body frame) and gyroscope (body frame),
     each = truth + slowly drifting bias (random walk) + white noise.

Everything is seeded so a run is reproducible.

Usage:
    from fly_pid import run
    from sensors import make_sensors
    data = run(duration_sec=30)
    meas = make_sensors(data, seed=0)
"""
import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])

# A zone is (center_x, center_y, radius) in metres, in the world X-Y plane.
# Default sits on the right side of the circle (circle: centre (0, -0.3), r 0.3).
DEFAULT_DROPOUT_ZONES = [(0.3, -0.3, 0.12)]


def rpy_to_rotmat(rpy):
    """Body-to-world rotation matrices for an (N, 3) array of roll/pitch/yaw.

    Convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (same as pybullet).
    Returns an (N, 3, 3) array.
    """
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.empty((len(rpy), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def in_dropout_zone(pos, zones):
    """Boolean array (N,): True where the true x-y position is inside any zone."""
    inside = np.zeros(len(pos), dtype=bool)
    for cx, cy, rad in zones:
        d = np.hypot(pos[:, 0] - cx, pos[:, 1] - cy)
        inside |= d < rad
    return inside


def make_gps(pos, zones=DEFAULT_DROPOUT_ZONES, sigma=0.03, rng=None):
    """Noisy GPS positions (N, 3) with NaN rows inside dropout zones.

    sigma: std of the noise per axis, metres. 0.03 m is 10% of the 0.3 m
    circle radius, i.e. clearly visible but not swamping the path.
    Returns (gps, valid) where valid is a boolean mask of usable fixes.
    """
    rng = np.random.default_rng() if rng is None else rng
    gps = pos + rng.normal(0.0, sigma, size=pos.shape)
    dropped = in_dropout_zone(pos, zones)
    gps[dropped] = np.nan
    return gps, ~dropped


def make_imu(time, vel, rpy, ang_vel, rng=None,
             accel_noise=0.05, accel_bias_walk=0.01, accel_bias_init=0.02,
             gyro_noise=0.005, gyro_bias_walk=0.001, gyro_bias_init=0.002):
    """Noisy IMU readings in the body frame.

    accel_noise / gyro_noise:   white noise std per sample (m/s^2, rad/s)
    *_bias_walk:                random-walk strength, units per sqrt(second)
    *_bias_init:                std of the starting bias
    Returns (accel (N,3), gyro (N,3), accel_bias (N,3), gyro_bias (N,3)).
    The true biases are returned so you can check the filter against them.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(time)
    dt = np.gradient(time)[:, None]

    # True specific force: what an accelerometer measures = R^T (a_world - g)
    a_world = np.gradient(vel, time, axis=0)
    R = rpy_to_rotmat(rpy)
    Rt = np.transpose(R, (0, 2, 1))
    f_body = np.einsum("nij,nj->ni", Rt, a_world - GRAVITY)

    # True body rates: pybullet gives world-frame angular velocity
    w_body = np.einsum("nij,nj->ni", Rt, ang_vel)

    def bias_path(init_std, walk):
        steps = rng.normal(0.0, 1.0, size=(n, 3)) * walk * np.sqrt(dt)
        steps[0] = rng.normal(0.0, init_std, size=3)
        return np.cumsum(steps, axis=0)

    accel_bias = bias_path(accel_bias_init, accel_bias_walk)
    gyro_bias = bias_path(gyro_bias_init, gyro_bias_walk)

    accel = f_body + accel_bias + rng.normal(0.0, accel_noise, size=(n, 3))
    gyro = w_body + gyro_bias + rng.normal(0.0, gyro_noise, size=(n, 3))
    return accel, gyro, accel_bias, gyro_bias


def make_sensors(data, seed=0, zones=DEFAULT_DROPOUT_ZONES, gps_sigma=0.03,
                 **imu_kwargs):
    """Run both sensors on a fly_pid.run() result. Returns a dict."""
    rng = np.random.default_rng(seed)
    gps, gps_valid = make_gps(data["pos"], zones, gps_sigma, rng)
    accel, gyro, accel_bias, gyro_bias = make_imu(
        data["time"], data["vel"], data["rpy"], data["ang_vel"], rng,
        **imu_kwargs)
    return {"time": data["time"], "gps": gps, "gps_valid": gps_valid,
            "accel": accel, "gyro": gyro,
            "accel_bias": accel_bias, "gyro_bias": gyro_bias,
            "zones": np.array(zones)}
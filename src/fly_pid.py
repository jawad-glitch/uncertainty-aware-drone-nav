"""Single-drone PID circle flight that records ground-truth data.

Adapted from gym-pybullet-drones' examples/pid.py. One Crazyflie flies a
circle in the X-Y plane (radius R, centred at (0, -R)) at a fixed height,
tracked by the built-in DSLPIDControl. Everything the sensor layer will need
later (true position, true velocity, target) is logged at every control step.

From a notebook:
    from fly_pid import run
    data = run(duration_sec=12, output_folder="/content/drive/MyDrive/.../results")
"""
import os
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Layout of obs[0] (20 values), from BaseAviary._getDroneStateVector
POS = slice(0, 3)      # x, y, z
QUAT = slice(3, 7)     # quaternion
RPY = slice(7, 10)     # roll, pitch, yaw
VEL = slice(10, 13)    # vx, vy, vz
ANG_VEL = slice(13, 16)


def run(
        duration_sec=12,
        radius=0.3,
        height=0.1,
        period_sec=10,
        control_freq_hz=48,
        simulation_freq_hz=240,
        drone=DroneModel.CF2X,
        physics=Physics.PYB,
        gui=False,
        obstacles=False,
        output_folder=None,
        save_name="pid_flight",
):
    """Fly one drone around a circle and return the ground-truth log.

    Returns a dict of numpy arrays:
        time    (N,)    seconds, time at which each observation was taken
        pos     (N, 3)  true position
        vel     (N, 3)  true velocity
        rpy     (N, 3)  true roll, pitch, yaw
        target  (N, 3)  target position the controller used for that step
    If output_folder is given, the dict is also saved there as <save_name>.npz.
    """
    # ---- start pose: on the circle at its first waypoint ----
    init_xyz = np.array([[0.0, 0.0, height]])
    init_rpy = np.zeros((1, 3))

    # ---- circular trajectory (x, y only; height is held constant) ----
    num_wp = int(control_freq_hz * period_sec)
    theta = 2 * np.pi * np.arange(num_wp) / num_wp + np.pi / 2
    circle_xy = np.stack([radius * np.cos(theta),
                          radius * np.sin(theta) - radius], axis=1)

    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=init_xyz,
                     initial_rpys=init_rpy,
                     physics=physics,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     obstacles=obstacles)

    try:
        ctrl = DSLPIDControl(drone_model=drone)

        n_steps = int(duration_sec * env.CTRL_FREQ)
        time_log = np.zeros(n_steps)
        pos_log = np.zeros((n_steps, 3))
        vel_log = np.zeros((n_steps, 3))
        rpy_log = np.zeros((n_steps, 3))
        target_log = np.zeros((n_steps, 3))

        action = np.zeros((1, 4))   # first step applies zero thrust
        wp = 0

        for i in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs[0]          # ground truth; Phase 2 swaps in the filter estimate here

            target = np.array([circle_xy[wp, 0], circle_xy[wp, 1], height])

            action[0, :], _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=state,
                target_pos=target,
                target_rpy=init_rpy[0],
            )

            # log the state the controller saw and the target it used
            time_log[i] = (i + 1) * env.CTRL_TIMESTEP
            pos_log[i] = state[POS]
            vel_log[i] = state[VEL]
            rpy_log[i] = state[RPY]
            target_log[i] = target

            wp = (wp + 1) % num_wp  # advance after logging
    finally:
        env.close()

    data = {"time": time_log, "pos": pos_log, "vel": vel_log,
            "rpy": rpy_log, "target": target_log}

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        np.savez(os.path.join(output_folder, f"{save_name}.npz"), **data)

    return data
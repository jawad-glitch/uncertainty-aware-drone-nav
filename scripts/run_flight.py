"""One-command demo.

    python scripts/run_flight.py

Flies one drone around a circle (headless PyBullet), saves the ground truth to
results/pid_flight.npz, corrupts it with seeded GPS/IMU noise, runs the three
Kalman filters and prints the comparison table.

Options:
    --duration 30        flight length in seconds
    --seed 0             sensor noise seed
    --zone-radius 0.12   GPS dropout zone radius in metres
    --skip-flight        reuse results/pid_flight.npz instead of re-simulating
    --gui                show the PyBullet window (needs a display)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sensors
import filters

CTRL_FREQ_HZ = 48
RESULTS = ROOT / "results"
FLIGHT_FILE = RESULTS / "pid_flight.npz"


def get_flight(args):
    if args.skip_flight:
        if not FLIGHT_FILE.exists():
            sys.exit(f"{FLIGHT_FILE} not found; run without --skip-flight first.")
        print(f"Loading {FLIGHT_FILE}")
        return dict(np.load(FLIGHT_FILE))

    # imported here so --skip-flight works without the simulator installed
    from fly_pid import run
    print(f"Flying {args.duration} s ...")
    return run(duration_sec=args.duration, gui=args.gui,
               output_folder=str(RESULTS), save_name="pid_flight")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zone-radius", type=float, default=None)
    ap.add_argument("--skip-flight", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    data = get_flight(args)
    dt = 1.0 / CTRL_FREQ_HZ

    zones = (sensors.DEFAULT_DROPOUT_ZONES if args.zone_radius is None
             else [(0.3, -0.3, args.zone_radius)])
    meas = sensors.make_sensors(data, seed=args.seed, zones=zones)
    gps, valid = meas["gps"], meas["gps_valid"]
    a_world = filters.a_world_from_imu(meas["accel"], data["rpy"])

    est_cv, _ = filters.run_cv_kf(gps, valid, dt)
    est_imu, _ = filters.run_imu_kf(a_world, gps, valid, dt)
    est_bias, _ = filters.run_imu_bias_kf(a_world, gps, valid, dt)

    true_xy = data["pos"][:, :2]

    def err(est):
        return np.linalg.norm(est[:, :2] - true_xy, axis=1)

    gps_err = np.linalg.norm(gps[:, :2] - true_xy, axis=1)   # NaN in dropouts
    rows = {
        "raw GPS":     gps_err,
        "const-vel":   err(est_cv),
        "IMU":         err(est_imu),
        "IMU + bias":  err(est_bias),
    }

    print(f"\nPosition RMSE (m)   seed={args.seed}  "
          f"GPS available {valid.mean():.1%} of the time")
    print(f"{'':<12}{'all steps':>12}{'GPS steps':>12}{'in dropout':>12}")
    for name, e in rows.items():
        if name == "raw GPS":
            print(f"{name:<12}{filters.rmse(e[valid]):>12.4f}"
                  f"{filters.rmse(e[valid]):>12.4f}{'n/a':>12}")
        else:
            print(f"{name:<12}{filters.rmse(e):>12.4f}"
                  f"{filters.rmse(e, valid):>12.4f}{filters.rmse(e, ~valid):>12.4f}")
    print("\n(raw GPS has no reading inside a dropout, so its 'all steps' is over valid fixes only)")


if __name__ == "__main__":
    main()

# Uncertainty-aware drone navigation

State estimation for a simulated quadrotor that loses GPS in known zones.
A Crazyflie flies a circle in PyBullet (gym-pybullet-drones). Noisy GPS and
IMU readings are generated from the ground truth, then fused with Kalman
filters. The long-term goal is a planner that avoids zones where the filter's
uncertainty grows (see the roadmap).

## Where things are

| Path | What it is |
|---|---|
| `src/fly_pid.py` | Flies one drone around a circle, logs ground truth |
| `src/sensors.py` | GPS (noise + dropout zones) and IMU (noise + bias drift), seeded |
| `src/filters.py` | Const-velocity, IMU, and IMU+bias Kalman filters (plain numpy) |
| `scripts/run_flight.py` | One-command demo, prints the comparison table |
| `notebooks/` | Walkthroughs with plots |
| `results/pid_flight.npz` | Saved ground-truth flight, so notebooks run without the simulator |

## Run locally (Linux / WSL2)

Developed on Python 3.14 (WSL2 Ubuntu) and 3.13 (Colab). PyBullet has no
prebuilt wheel for these versions, so it compiles from source, which needs a
C++ compiler and takes a few minutes.

```bash
sudo apt install -y build-essential python3-dev
git clone https://github.com/jawad-glitch/uncertainty-aware-drone-nav.git
cd uncertainty-aware-drone-nav
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # avoids ~2 GB of CUDA
pip install -r requirements.txt
python scripts/run_flight.py
```

The torch line comes first because the simulator lists PyTorch as a
dependency (only its RL examples use it). Without it, pip downloads the CUDA
build.

The demo flies 30 s headless, saves `results/pid_flight.npz`, adds sensor noise
(seed 0) and prints position RMSE for raw GPS and the three filters, over all
steps, GPS-available steps, and inside the dropout zone.

Useful options:

```bash
python scripts/run_flight.py --skip-flight        # reuse the saved flight, no simulator needed
python scripts/run_flight.py --zone-radius 0.2    # bigger dropout zone, bigger gap between filters
python scripts/run_flight.py --gui                # watch it fly (needs a display, e.g. WSLg)
```

## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jawad-glitch/uncertainty-aware-drone-nav/blob/main/notebooks/kf_imu.ipynb)

First cell of any notebook:

```python
!git clone https://github.com/jawad-glitch/uncertainty-aware-drone-nav.git
%cd uncertainty-aware-drone-nav
!pip install -r requirements.txt
```

Colab already ships PyTorch, so no separate torch line. The filter notebooks
(`02`, `03`) only need `results/pid_flight.npz`, which is in the repo, so they
run without the simulator. Colab sessions are temporary: download or commit
anything you want to keep.

## Reproducibility

- The simulator is pinned to an exact commit in `requirements.txt`.
- Sensor noise is seeded (`make_sensors(data, seed=0)`).
- Reference numbers from the author's run (30 s flight, seed 0, default
  dropout zone), position RMSE in metres:

| | all steps | GPS steps | in dropout |
|---|---|---|---|
| raw GPS | 0.0425 | 0.0425 | n/a |
| const-vel | 0.0254 | 0.0152 | 0.0594 |
| IMU (no bias) | 0.0589 | 0.0437 | 0.1200 |
| IMU + bias | 0.0168 | 0.0114 | 0.0367 |

Small differences in the last digit across machines are normal. A large
difference means something in the setup changed.

## Status

Done: simulation and ground-truth logging, sensor models, three linear Kalman
filters. Next: an EKF that estimates attitude from the gyro instead of using
the true orientation.

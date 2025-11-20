# Tello Mapping, Demo of Range

This repository contains a minimal demo that integrates camera-based obstacle detection, a simple occupancy grid mapper, an A* planner, and a flight runner that can operate in simulation (using your webcam) or with a real Tello drone.

Files added:
- `mapper.py` — simple occupancy grid mapping utilities.
- `planner.py` — A* planner on the occupancy grid.
- `flight_runner.py` — integrates camera, mapper, planner and shows visualization.
- `requirements.txt` — Python packages needed.

Important limitations:
- With only a monocular camera you do not have true distance (range) measurement. The code uses conservative heuristics to estimate obstacle distance. For accurate wall positions you must add a range sensor (e.g., ultrasonic / LiDAR) or a depth camera.
- The demo is conservative and meant to be a starting point.

Quick start (simulation using a webcam):
1. Create a Python venv and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the flight runner in simulation:

```bash
python flight_runner.py
```

Switch to a real Tello by editing `flight_runner.py` and setting `USE_SIM = False`. Make sure the Tello SDK is available and you are on the Tello's Wi‑Fi network.

I dont have accurate range, so gotta grab a LIDAR or Ultrasonic range sensor.

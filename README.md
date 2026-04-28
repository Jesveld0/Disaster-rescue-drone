# Dual-Sensor Fire Rescue Drone Payload System

Human-in-Fire Detection + Assisted Obstacle Stop

A modular attachable payload system for quadcopter drones that detects humans, fire zones, and obstacles using fused RGB + thermal imaging with AI inference. Features RF-DETR object detection, DeepSORT human tracking, A* pathfinding, and IR proximity sensor obstacle avoidance.

## Architecture

```
┌─────────────────────────────────┐       UDP 5000              ┌─────────────────────────────────────┐
│       DRONE PAYLOAD (Pi 4)      │ ──── Frame Data ──────────► │       GROUND STATION (Laptop)       │
│                                 │       UDP 5002              │                                     │
│  USB Camera (1280×720 @ 30fps)  │ ──── Obstacle Msg ────────► │  UDP Receiver + Fragment Reassembly  │
│  MLX90640 Thermal (32×24)       │                             │  Obstacle Receiver (IR data)         │
│  4× IR Proximity Sensors        │ ◄── Commands (STOP/SAFE) ── │  RF-DETR Detection (CUDA)           │
│  JPEG Compression               │       UDP 5001              │  DeepSORT Human Tracking            │
│  UDP Sender + Obstacle Sender   │                             │  MiDaS Depth Estimation             │
└─────────────────────────────────┘                             │  A* Pathfinding + Occupancy Grid    │
                                                                │  Thermal-RGB Fusion                 │
                                                                │  Decision Engine (IR-aware)          │
                                                                │  Visualizer (tracks + grid + IR)     │
                                                                └─────────────────────────────────────┘
```

## Quick Start

### Ground Station (Laptop with GPU)
```bash
pip install -r requirements_ground.txt
python3 -m ground_station.pipeline
```

### Drone Payload (Raspberry Pi 4)
```bash
pip install -r requirements_pi.txt
python3 -m edge.sender --ground-ip <LAPTOP_IP>
```

### Run Tests
```bash
python3 -m pytest tests/ -v
```

## Project Structure

```
project/
├── config.py                     # All constants & settings
├── protocol.py                   # Binary packet encoding/decoding (frame + command + obstacle)
├── requirements_pi.txt           # Pi dependencies
├── requirements_ground.txt       # Ground station dependencies
├── edge/                         # Raspberry Pi code
│   ├── rgb_capture.py            # USB camera capture
│   ├── thermal_capture.py        # MLX90640 + grayscale conversion
│   ├── ir_sensor.py              # IR proximity sensor reader (4 directional)
│   └── sender.py                 # UDP sender (main Pi entry)
├── ground_station/               # Ground station code
│   ├── receiver.py               # UDP receiver + reassembly
│   ├── decoder.py                # JPEG/thermal decoding
│   ├── thermal_processing.py     # Upscale, align, fire mask
│   ├── detector.py               # RF-DETR inference (replaced YOLOv8)
│   ├── tracker.py                # DeepSORT human tracking
│   ├── depth_estimator.py        # MiDaS monocular depth
│   ├── pathfinder.py             # A* pathfinding + occupancy grid
│   ├── obstacle_receiver.py      # UDP receiver for IR obstacle data
│   ├── fusion.py                 # Thermal-RGB fusion logic
│   ├── decision.py               # Command generation (IR-aware)
│   ├── command_sender.py         # UDP commands to drone
│   ├── visualizer.py             # Annotated display + JSON (tracks + IR + grid)
│   └── pipeline.py               # Threaded pipeline (main GS entry)
├── calibration/
│   └── calibrate_homography.py   # Thermal→RGB alignment tool
└── tests/
    ├── test_protocol.py          # Protocol encode/decode tests
    ├── test_obstacle_protocol.py # Obstacle packet tests
    ├── test_pathfinder.py        # A* pathfinding tests
    ├── test_tracker.py           # DeepSORT tracker tests
    ├── test_thermal.py           # Thermal processing tests
    └── test_loopback.py          # UDP loopback integration tests
```

## Detection & Tracking Capabilities

| Feature | Model/Method | Details |
|---------|-------------|---------|
| Person Detection | RF-DETR Base (COCO) | DINOv2 backbone, confidence > 0.4 |
| Human Tracking | DeepSORT | Persistent track IDs across frames |
| Fire Zone | RF-DETR + Thermal cross-validation | thermal > 50°C |
| Human in Fire | Person bbox + thermal fusion | max_temp > 50°C AND hot_ratio > 20% |
| Obstacle (Visual) | RF-DETR + MiDaS depth | depth < 3m OR bbox > 15% screen |
| Obstacle (Physical) | IR Proximity Sensors | 4 directional sensors on RPi GPIO |
| Pathfinding | A* Algorithm | 20×20 occupancy grid with decay |

## Command Protocol

| Code | Command | Priority | Trigger |
|------|---------|----------|---------|
| 4 | HUMAN_IN_FIRE | Highest | Person detected in fire zone |
| 3 | FIRE_ALERT | High | Confirmed fire zone |
| 2 | STOP | Medium | IR front obstacle / visual obstacle too close / timeout |
| 1 | SLOW | Low | IR side/back obstacle / obstacle approaching |
| 0 | SAFE | Lowest | No threats detected |

**Fail-safe**: Auto-STOP if no inference result within 500ms.

## UDP Ports

| Port | Direction | Data |
|------|-----------|------|
| 5000 | Drone → GS | Frame data (RGB + thermal) |
| 5001 | GS → Drone | Commands (STOP/SAFE/etc.) |
| 5002 | Drone → GS | IR obstacle data (20-byte packets) |

## IR Proximity Sensors

The drone payload includes 4 IR proximity sensors connected to the Raspberry Pi GPIO:

| Direction | GPIO Pin (BCM) | Stop Trigger |
|-----------|---------------|-------------|
| Front | 17 | CMD_STOP (highest danger) |
| Back | 27 | CMD_SLOW |
| Left | 22 | CMD_SLOW |
| Right | 23 | CMD_SLOW |

Obstacle data is sent as 20-byte binary packets (magic `0x4F425354` + frame_id + timestamp + 4 booleans).

## Keyboard Controls (Ground Station Display)

- `t` — Toggle thermal overlay
- `d` — Toggle depth map overlay
- `f` — Toggle fire mask overlay
- `p` — Toggle pathfinding grid overlay
- `q` — Quit

## CLI Options

### Ground Station
```bash
python3 -m ground_station.pipeline --help
  --port PORT           UDP data port (default: 5000)
  --command-port PORT   UDP command port (default: 5001)
  --obstacle-port PORT  UDP obstacle port (default: 5002)
  --no-depth            Disable MiDaS depth estimation
  --no-tracking         Disable DeepSORT human tracking
  --no-pathfinding      Disable A* pathfinding
  --headless            Run without display window
```

### Drone Sender
```bash
python3 -m edge.sender --help
  --ground-ip IP        Ground station IP address
  --data-port PORT      Data port (default: 5000)
  --obstacle-port PORT  Obstacle data port (default: 5002)
  --no-ir               Disable IR proximity sensors
  --device INDEX        Camera device index
```

## Configuration

Edit `config.py` to adjust:
- Network IPs and ports
- Fire threshold temperature (default: 50°C)
- RF-DETR confidence thresholds
- DeepSORT tracker parameters (max_age, n_init)
- IR sensor GPIO pin mapping
- Pathfinding grid size and cell size
- Obstacle detection sensitivity
- Homography calibration matrix

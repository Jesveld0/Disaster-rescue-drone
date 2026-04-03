# Dual-Sensor Fire Rescue Drone Payload System

Human-in-Fire Detection + Assisted Obstacle Stop

A modular attachable payload system for quadcopter drones that detects humans, fire zones, and obstacles using fused RGB + thermal imaging with AI inference.

## Architecture

```
┌─────────────────────────────────┐       UDP (5GHz WiFi)       ┌──────────────────────────────────┐
│       DRONE PAYLOAD (Pi 4)      │ ──────────────────────────► │       GROUND STATION (Laptop)    │
│                                 │      Frame Data (20 FPS)    │                                  │
│  USB Camera (1280×720 @ 30fps)  │                             │  UDP Receiver + Fragment Reassembly│
│  MLX90640 Thermal (32×24)       │ ◄────────────────────────── │  YOLOv8 Detection (CUDA)         │
│  JPEG Compression               │      Commands (STOP/SAFE)   │  MiDaS Depth Estimation          │
│  UDP Sender with Fragmentation  │                             │  Thermal-RGB Fusion              │
└─────────────────────────────────┘                             │  Decision Engine + Visualizer    │
                                                                └──────────────────────────────────┘
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
├── protocol.py                   # Binary packet encoding/decoding
├── requirements_pi.txt           # Pi dependencies
├── requirements_ground.txt       # Ground station dependencies
├── edge/                         # Raspberry Pi code
│   ├── rgb_capture.py            # USB camera capture
│   ├── thermal_capture.py        # MLX90640 + BGR heatmap generation
│   └── sender.py                 # UDP sender (main Pi entry)
├── ground_station/               # Ground station code
│   ├── receiver.py               # UDP receiver + reassembly
│   ├── decoder.py                # JPEG/thermal decoding
│   ├── thermal_processing.py     # Upscale, align, fire mask
│   ├── detector.py               # YOLOv8 inference
│   ├── depth_estimator.py        # MiDaS monocular depth
│   ├── fusion.py                 # Thermal-RGB fusion logic
│   ├── decision.py               # Command generation
│   ├── command_sender.py         # UDP commands to drone
│   ├── visualizer.py             # Annotated display + JSON
│   └── pipeline.py               # Threaded pipeline (main GS entry)
├── calibration/
│   └── calibrate_homography.py   # Thermal→RGB alignment tool
└── tests/
    ├── test_protocol.py          # Protocol encode/decode tests
    ├── test_thermal.py           # Thermal processing tests
    └── test_loopback.py          # UDP loopback integration tests
```

## Detection Capabilities

| Class | Source | Threshold |
|-------|--------|-----------|
| Person | YOLOv8 (COCO) | confidence > 0.4 |
| Fire Zone | YOLOv8 + Thermal cross-validation | thermal > 50°C |
| Human in Fire | Person bbox + thermal fusion | max_temp > 50°C AND hot_ratio > 20% |
| Obstacle | YOLOv8 + MiDaS depth | depth < 3m OR bbox > 15% screen |

## Command Protocol

| Code | Command | Priority | Trigger |
|------|---------|----------|---------|
| 4 | HUMAN_IN_FIRE | Highest | Person detected in fire zone |
| 3 | FIRE_ALERT | High | Confirmed fire zone |
| 2 | STOP | Medium | Obstacle too close / timeout |
| 1 | SLOW | Low | Obstacle approaching |
| 0 | SAFE | Lowest | No threats detected |

**Fail-safe**: Auto-STOP if no inference result within 500ms.

## Keyboard Controls (Ground Station Display)

- `t` — Toggle thermal overlay
- `d` — Toggle depth map overlay
- `f` — Toggle fire mask overlay
- `q` — Quit

## Configuration

Edit `config.py` to adjust:
- Network IPs and ports
- Fire threshold temperature (default: 50°C)
- YOLO confidence thresholds
- Obstacle detection sensitivity
- Homography calibration matrix

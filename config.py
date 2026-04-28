"""
config.py — Global configuration for Dual-Sensor Fire Rescue Drone Payload System.

All shared constants, thresholds, and tunable parameters live here.
Both edge (Pi) and ground station import from this module.
"""

import numpy as np

# =============================================================================
# Network Configuration
# =============================================================================
GROUND_STATION_IP = "192.168.1.100"   # Laptop IP on the 5 GHz WiFi network
DRONE_IP = "192.168.1.101"            # Raspberry Pi IP
DATA_PORT = 5000                      # Drone → Ground Station (frame data)
COMMAND_PORT = 5001                   # Ground Station → Drone (commands)
BUFFER_SIZE = 65535                   # Max UDP datagram (practical limit)
MAX_PACKET_SIZE = 8000                # Max payload per UDP fragment (safe for all platforms)

# =============================================================================
# Camera Configuration
# =============================================================================
RGB_WIDTH = 1280
RGB_HEIGHT = 720
RGB_FPS = 30
THERMAL_WIDTH = 32
THERMAL_HEIGHT = 24
THERMAL_PIXELS = THERMAL_WIDTH * THERMAL_HEIGHT  # 768

# =============================================================================
# Thermal Processing
# =============================================================================
THERMAL_MIN_TEMP = 20.0    # °C — assumed ambient minimum
THERMAL_MAX_TEMP = 150.0   # °C — assumed maximum detectable
THERMAL_RANGE = THERMAL_MAX_TEMP - THERMAL_MIN_TEMP  # 130.0
FIRE_THRESHOLD_TEMP = 50.0  # °C — fire detection threshold

# Grayscale mapping: gray = clip((temp - 20) / 130 * 255, 0, 255)
# Reverse: temp = gray / 255 * 130 + 20
FIRE_THRESHOLD_GRAY = int(
    np.clip((FIRE_THRESHOLD_TEMP - THERMAL_MIN_TEMP) / THERMAL_RANGE * 255, 0, 255)
)

# Human-in-fire decision thresholds
HOT_PIXEL_RATIO_THRESHOLD = 0.2  # 20% of bbox pixels must exceed fire temp

# =============================================================================
# Protocol Constants
# =============================================================================
MAGIC_NUMBER = 0x46495245  # "FIRE" in ASCII hex
HEADER_SIZE = 16           # magic(4) + frame_id(4) + timestamp(8)
METADATA_SIZE = 8          # rgb_w(2) + rgb_h(2) + therm_w(2) + therm_h(2)
COMMAND_PACKET_SIZE = HEADER_SIZE + 1  # header + command_code

# Command codes (Ground Station → Drone)
CMD_SAFE = 0
CMD_SLOW = 1
CMD_STOP = 2
CMD_FIRE_ALERT = 3
CMD_HUMAN_IN_FIRE = 4

COMMAND_NAMES = {
    CMD_SAFE: "SAFE",
    CMD_SLOW: "SLOW",
    CMD_STOP: "STOP",
    CMD_FIRE_ALERT: "FIRE_ALERT",
    CMD_HUMAN_IN_FIRE: "HUMAN_IN_FIRE",
}

# Command priority (higher index = higher priority)
COMMAND_PRIORITY = [CMD_SAFE, CMD_SLOW, CMD_STOP, CMD_FIRE_ALERT, CMD_HUMAN_IN_FIRE]

# =============================================================================
# Performance Targets
# =============================================================================
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 50 ms
TIMEOUT_MS = 500                     # No-packet timeout → fail-safe STOP
TIMEOUT_SEC = TIMEOUT_MS / 1000.0    # 0.5 s

# JPEG compression quality (trade-off: size vs quality)
JPEG_QUALITY = 80  # Produces ~80-120 KB at 720p

# =============================================================================
# AI Model Configuration — RF-DETR
# =============================================================================
RFDETR_MODEL_SIZE = "base"            # 'base', 'small', or 'large'
RFDETR_CONFIDENCE = 0.4               # Minimum detection confidence

# COCO class mapping (RF-DETR uses standard COCO 80 classes)
PERSON_CLASS_ID = 0
# Fire and obstacle classes will come from custom model or be mapped separately
FIRE_CLASS_LABEL = "fire"
OBSTACLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle",
                    "bench", "chair", "potted plant"]

# MiDaS depth estimation
MIDAS_MODEL_TYPE = "MiDaS_small"      # Lightweight for real-time
OBSTACLE_DEPTH_THRESHOLD = 3.0        # meters — trigger STOP if closer
OBSTACLE_BBOX_AREA_RATIO = 0.15       # bbox area / frame area threshold

# =============================================================================
# Human Tracking (DeepSORT)
# =============================================================================
TRACKER_MAX_AGE = 30                  # Frames to keep unmatched track alive
TRACKER_N_INIT = 3                    # Hits before track is confirmed
TRACKER_MAX_IOU_DISTANCE = 0.7        # Max IoU distance for matching

# =============================================================================
# IR Proximity Sensors (Raspberry Pi GPIO — BCM numbering)
# =============================================================================
IR_FRONT_PIN = 17
IR_BACK_PIN = 27
IR_LEFT_PIN = 22
IR_RIGHT_PIN = 23
IR_POLL_INTERVAL = 0.05               # 50ms polling interval
IR_DEBOUNCE_COUNT = 3                 # Consecutive reads to confirm obstacle

# Obstacle Protocol
OBSTACLE_MAGIC = 0x4F425354           # "OBST" in ASCII hex
OBSTACLE_PORT = 5002                  # Separate UDP port for obstacle data

# =============================================================================
# Pathfinding (A*)
# =============================================================================
GRID_WIDTH = 20                       # Grid columns
GRID_HEIGHT = 20                      # Grid rows
GRID_CELL_SIZE = 0.5                  # Meters per cell
OBSTACLE_DECAY_FRAMES = 10            # Frames before obstacle cell fades

# =============================================================================
# Homography Calibration (Thermal → RGB alignment)
# =============================================================================
# Default: identity matrix (no warp). Replace after running calibration tool.
HOMOGRAPHY_MATRIX = np.eye(3, dtype=np.float64)

# =============================================================================
# Visualization
# =============================================================================
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
THERMAL_OVERLAY_ALPHA = 0.3           # Transparency for thermal overlay
FIRE_ZONE_COLOR = (0, 0, 255)         # BGR red — intentionally same as HUMAN_IN_FIRE_COLOR
HUMAN_IN_FIRE_COLOR = (0, 0, 255)     # BGR red — kept separate so each can be tuned independently
PERSON_COLOR = (0, 255, 0)            # BGR green
OBSTACLE_COLOR = (0, 165, 255)        # BGR orange
FIRE_ZONE_OVERLAY_COLOR = (0, 0, 200) # BGR dark red

# =============================================================================
# Logging
# =============================================================================
LOG_JSON_OUTPUT = True                 # Write per-frame JSON to stdout
LOG_FILE_PATH = "output_log.jsonl"     # JSONL log file path

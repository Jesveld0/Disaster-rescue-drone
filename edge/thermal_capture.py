"""
thermal_capture.py — MLX90640 thermal camera capture module for Raspberry Pi.

Uses the exact working I2C code from thermal_video.py, wrapped in a
ThermalCamera class with open() / read() / close() — the same interface
as RGBCamera — so the sender can capture and transmit thermal heatmaps
to the ground station.

read() returns (temperatures, heatmap_bgr):
    temperatures — raw float list of 768 values from the sensor
    heatmap_bgr  — numpy uint8 array (24, 32, 3) BGR colormap image
"""

import logging
import threading
import numpy as np

from config import THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_PIXELS

logger = logging.getLogger(__name__)

# MLX90640 library is only available on Raspberry Pi with I2C hardware.
try:
    import board
    import busio
    import adafruit_mlx90640
    HAS_MLX90640 = True
except ImportError:
    HAS_MLX90640 = False
    logger.warning(
        "MLX90640 libraries not available. Using simulated thermal data. "
        "Install adafruit-circuitpython-mlx90640 on Raspberry Pi."
    )


class ThermalCamera:
    """
    MLX90640 thermal camera handler.

    Uses the proven thermal_video.py I2C init and frame reading.
    Produces a BGR heatmap image (24, 32, 3) for transmission,
    matching how RGBCamera produces BGR images for the ground station.

    Falls back to simulated data when hardware is not available.

    Usage:
        cam = ThermalCamera()
        cam.open()
        temps, heatmap_bgr = cam.read()
        cam.close()
    """

    # JET-like BGR color palette (matching thermal_video.py's 6 terminal colors)
    # These are actual BGR values for building an OpenCV-compatible image.
    _PALETTE = np.array([
        [255, 0, 0],       # Blue (Coldest)
        [255, 255, 0],     # Cyan
        [0, 128, 0],       # Green
        [0, 255, 255],     # Yellow
        [0, 0, 255],       # Red (Hottest)
        [255, 0, 255],     # Magenta (Burning)
    ], dtype=np.uint8)

    def __init__(self, refresh_rate: int = 8):
        self.refresh_rate = refresh_rate
        self.mlx = None
        self._i2c = None
        self._frame_buffer = [0.0] * THERMAL_PIXELS
        self._simulated = not HAS_MLX90640
        self._lock = threading.Lock()
        self._latest_temps = [0.0] * THERMAL_PIXELS
        self._latest_heatmap = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
        self._running = False
        self._thread = None

    def open(self) -> bool:
        """
        Initialize the MLX90640 sensor.
        Exact same I2C setup as thermal_video.py.
        """
        if self._simulated:
            logger.info("ThermalCamera running in SIMULATED mode")
            return True

        try:
            logger.info("Initializing MLX90640 Thermal Camera...")
            self._i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(self._i2c)

            rate_map = {
                2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
                16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
                32: adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
                64: adafruit_mlx90640.RefreshRate.REFRESH_64_HZ,
            }
            self.mlx.refresh_rate = rate_map.get(
                self.refresh_rate,
                adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
            )
            logger.info("MLX90640 initialized at %d Hz", self.refresh_rate)

            # Background thread to continuously read frames
            self._running = True
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()
            return True

        except Exception as e:
            logger.error("Failed to initialize MLX90640: %s", e)
            self._simulated = True
            logger.info("Falling back to simulated thermal data")
            return True

    def _update_loop(self):
        """
        Background thread — exact same getFrame() loop as thermal_video.py.
        Converts each frame into a BGR heatmap image.
        """
        while self._running:
            try:
                self.mlx.getFrame(self._frame_buffer)

                # Build heatmap exactly like thermal_video.py
                min_temp = min(self._frame_buffer)
                max_temp = max(self._frame_buffer)
                range_temp = max_temp - min_temp if max_temp != min_temp else 1

                heatmap = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
                for h in range(THERMAL_HEIGHT):
                    for w in range(THERMAL_WIDTH):
                        temp = self._frame_buffer[h * THERMAL_WIDTH + w]
                        norm = (temp - min_temp) / range_temp
                        color_idx = int(norm * (len(self._PALETTE) - 1))
                        heatmap[h, w] = self._PALETTE[color_idx]

                with self._lock:
                    self._latest_temps = list(self._frame_buffer)
                    self._latest_heatmap = heatmap

            except ValueError:
                pass  # Ignore missed internal subpages
            except Exception as e:
                logger.debug("Frame read error (non-fatal): %s", e)

    def read(self) -> tuple[list, np.ndarray]:
        """
        Read the latest thermal frame.

        Returns:
            (temperatures, heatmap_bgr):
                temperatures — list of 768 floats (raw °C)
                heatmap_bgr  — uint8 numpy array (24, 32, 3) BGR heatmap
        """
        if self._simulated:
            return self._read_simulated()

        with self._lock:
            return list(self._latest_temps), self._latest_heatmap.copy()

    def _read_simulated(self) -> tuple[list, np.ndarray]:
        """Generate simulated thermal data for dev/testing."""
        temperatures = np.random.normal(25.0, 2.0, THERMAL_PIXELS).astype(np.float32)

        # Add a random hot spot
        cx = np.random.randint(8, 24)
        cy = np.random.randint(6, 18)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y, x = cy + dy, cx + dx
                if 0 <= y < THERMAL_HEIGHT and 0 <= x < THERMAL_WIDTH:
                    dist = np.sqrt(dx**2 + dy**2)
                    idx = y * THERMAL_WIDTH + x
                    temperatures[idx] = max(
                        temperatures[idx],
                        80.0 - dist * 10.0 + np.random.normal(0, 3),
                    )

        temps_list = temperatures.tolist()
        min_temp = min(temps_list)
        max_temp = max(temps_list)
        range_temp = max_temp - min_temp if max_temp != min_temp else 1

        heatmap = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
        for h in range(THERMAL_HEIGHT):
            for w in range(THERMAL_WIDTH):
                temp = temps_list[h * THERMAL_WIDTH + w]
                norm = (temp - min_temp) / range_temp
                color_idx = int(norm * (len(self._PALETTE) - 1))
                heatmap[h, w] = self._PALETTE[color_idx]

        return temps_list, heatmap

    def close(self):
        """Clean up sensor resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._i2c is not None:
            try:
                self._i2c.deinit()
            except Exception:
                pass
            self._i2c = None
        self.mlx = None
        logger.info("ThermalCamera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ─── Standalone terminal heatmap (identical to thermal_video.py) ────────────
if __name__ == "__main__":
    import time
    import board
    import busio
    import adafruit_mlx90640

    COLORS = ['\033[44m  \033[0m',   # Deep Blue (Coldest)
              '\033[46m  \033[0m',   # Cyan
              '\033[42m  \033[0m',   # Green
              '\033[43m  \033[0m',   # Yellow
              '\033[41m  \033[0m',   # Red (Hottest)
              '\033[45m  \033[0m']   # Magenta (Burning)

    print("Initializing MLX90640 Thermal Camera...")
    i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ

    frame = [0.0] * 768

    while True:
        try:
            mlx.getFrame(frame)

            min_temp = min(frame)
            max_temp = max(frame)
            range_temp = max_temp - min_temp if max_temp != min_temp else 1

            print('\033[2J\033[H', end="")
            print(f"--- LIVE THERMAL STREAM: Max Temp: {max_temp:.1f}°C ---")

            for h in range(24):
                line = ""
                for w in range(32):
                    temp = frame[h * 32 + w]
                    norm = (temp - min_temp) / range_temp
                    color_idx = int(norm * (len(COLORS) - 1))
                    line += COLORS[color_idx]
                print(line)

        except ValueError:
            pass
        except Exception as e:
            print(f"Error: {e}")
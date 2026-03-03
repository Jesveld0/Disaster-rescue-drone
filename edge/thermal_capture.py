"""
thermal_capture.py — MLX90640 thermal camera capture module for Raspberry Pi.

Reads temperature data from the MLX90640 infrared sensor over I2C and converts
the raw temperature array to a grayscale image for transmission.

Grayscale mapping:
    gray = clip((temp - 20) / 130 * 255, 0, 255)
"""

import logging
import numpy as np

from config import (
    THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_PIXELS,
    THERMAL_MIN_TEMP, THERMAL_RANGE,
)

logger = logging.getLogger(__name__)

# MLX90640 library is only available on Raspberry Pi with I2C hardware.
# We handle ImportError gracefully to allow testing on dev machines.
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

    Reads 32x24 temperature values via I2C and provides:
    - Raw temperature arrays (float, °C)
    - Grayscale-converted arrays (uint8) for network transmission

    Falls back to simulated data when hardware is not available.

    Usage:
        thermal = ThermalCamera()
        thermal.open()
        temps, grayscale = thermal.read()
        thermal.close()
    """

    def __init__(self, refresh_rate: int = 8):
        """
        Args:
            refresh_rate: MLX90640 refresh rate in Hz (2, 4, 8, 16, 32, 64).
        """
        self.refresh_rate = refresh_rate
        self.mlx = None
        self._frame_buffer = [0.0] * THERMAL_PIXELS
        self._simulated = not HAS_MLX90640

    def open(self) -> bool:
        """
        Initialize the MLX90640 sensor.

        Returns:
            True if sensor initialized (or simulated mode active).
        """
        if self._simulated:
            logger.info("ThermalCamera running in SIMULATED mode")
            return True

        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
            self.mlx = adafruit_mlx90640.MLX90640(i2c)

            # Set refresh rate
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
            logger.info(
                "MLX90640 initialized at %d Hz, serial: %s",
                self.refresh_rate, [hex(i) for i in self.mlx.serial_number],
            )
            return True

        except Exception as e:
            logger.error("Failed to initialize MLX90640: %s", e)
            self._simulated = True
            logger.info("Falling back to simulated thermal data")
            return True

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Read a single thermal frame.

        Returns:
            (temperatures, grayscale):
                temperatures — float32 array (24, 32) in °C
                grayscale — uint8 array (24, 32) normalized to 0-255
        """
        if self._simulated:
            return self._read_simulated()

        try:
            self.mlx.getFrame(self._frame_buffer)
            temperatures = np.array(self._frame_buffer, dtype=np.float32).reshape(
                (THERMAL_HEIGHT, THERMAL_WIDTH)
            )
        except Exception as e:
            logger.warning("MLX90640 read error: %s — returning zeros", e)
            temperatures = np.zeros(
                (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.float32
            )

        grayscale = self.temps_to_grayscale(temperatures)
        return temperatures, grayscale

    def _read_simulated(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated thermal data for testing.

        Creates a base ambient temperature field (~25°C) with random
        variation and a simulated hot spot.
        """
        # Base ambient temperature with noise
        temperatures = np.random.normal(25.0, 2.0, (THERMAL_HEIGHT, THERMAL_WIDTH))
        temperatures = temperatures.astype(np.float32)

        # Add a random hot spot (simulating fire)
        cx = np.random.randint(8, 24)
        cy = np.random.randint(6, 18)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                y, x = cy + dy, cx + dx
                if 0 <= y < THERMAL_HEIGHT and 0 <= x < THERMAL_WIDTH:
                    dist = np.sqrt(dx**2 + dy**2)
                    temperatures[y, x] = max(
                        temperatures[y, x],
                        80.0 - dist * 10.0 + np.random.normal(0, 3),
                    )

        grayscale = self.temps_to_grayscale(temperatures)
        return temperatures, grayscale

    @staticmethod
    def temps_to_grayscale(temperatures: np.ndarray) -> np.ndarray:
        """
        Convert temperature array to grayscale using the mapping formula:
            gray = clip((temp - 20) / 130 * 255, 0, 255)

        Args:
            temperatures: float array of temperatures in °C.

        Returns:
            uint8 array of grayscale values.
        """
        normalized = (temperatures - THERMAL_MIN_TEMP) / THERMAL_RANGE * 255.0
        grayscale = np.clip(normalized, 0, 255).astype(np.uint8)
        return grayscale

    @staticmethod
    def grayscale_to_temps(grayscale: np.ndarray) -> np.ndarray:
        """
        Reverse grayscale mapping to approximate temperatures:
            temp = gray / 255 * 130 + 20

        Args:
            grayscale: uint8 array of grayscale values.

        Returns:
            float32 array of approximate temperatures in °C.
        """
        return (grayscale.astype(np.float32) / 255.0) * THERMAL_RANGE + THERMAL_MIN_TEMP

    def close(self):
        """Clean up sensor resources."""
        self.mlx = None
        logger.info("ThermalCamera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

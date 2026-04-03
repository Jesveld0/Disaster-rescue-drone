"""
thermal_processing.py — Thermal image normalization, upscaling, and alignment.

Processes 32x24 thermal heatmaps:
1. Recover the original JET intensity from the transmitted heatmap
2. Upscale to RGB resolution (1280x720) via bicubic interpolation
3. Apply homography warp to align with RGB frame
4. Generate binary fire mask (pixels > 50°C)
"""

import logging
import cv2
import numpy as np

from config import (
    RGB_WIDTH, RGB_HEIGHT,
    THERMAL_WIDTH, THERMAL_HEIGHT,
    THERMAL_MIN_TEMP, THERMAL_RANGE,
    FIRE_THRESHOLD_TEMP, FIRE_THRESHOLD_GRAY,
    HOMOGRAPHY_MATRIX,
)

logger = logging.getLogger(__name__)


class ThermalProcessor:
    """
    Handles all thermal image processing on the ground station.

    Pipeline:
        heatmap (32x24x3) → recover intensity → upscale → warp (homography) → fire mask
    """

    def __init__(self, homography_matrix: np.ndarray = None):
        """
        Args:
            homography_matrix: 3x3 homography for thermal→RGB alignment.
                               Defaults to identity (no warp).
        """
        self.homography = homography_matrix if homography_matrix is not None else HOMOGRAPHY_MATRIX.copy()
        self._target_size = (RGB_WIDTH, RGB_HEIGHT)

        # Precompute JET colormap lookup table so we can recover intensity
        # (0..255) from the transmitted 3-channel heatmap.
        self._jet_lut_bgr = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            intensity = np.array([[i]], dtype=np.uint8)
            self._jet_lut_bgr[i] = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)[0, 0, :]

    def process(self, thermal_heatmap: np.ndarray) -> dict:
        """
        Full thermal processing pipeline.

        Args:
            thermal_heatmap: Raw heatmap BGR thermal image (24, 32, 3) uint8.

        Returns:
            Dictionary with processed outputs:
                - 'upscaled': Upscaled intensity (720, 1280) uint8
                - 'aligned': Homography-warped intensity (720, 1280) uint8
                - 'temperatures': Approximate temperature map (720, 1280) float32
                - 'fire_mask': Binary mask of fire pixels (720, 1280) uint8
                - 'colormap': Colored thermal visualization (720, 1280, 3) uint8
        """
        intensity_24x32 = self.heatmap_to_intensity(thermal_heatmap)

        upscaled_intensity = self.upscale_intensity(intensity_24x32)
        aligned_intensity = self.align(upscaled_intensity)
        temperatures = self.intensity_to_temps(aligned_intensity)
        fire_mask = self.compute_fire_mask(aligned_intensity)

        upscaled_heatmap = self.upscale_heatmap(thermal_heatmap)
        aligned_heatmap = self.align(upscaled_heatmap)
        colormap = aligned_heatmap

        return {
            "upscaled": upscaled_intensity,
            "aligned": aligned_intensity,
            "temperatures": temperatures,
            "fire_mask": fire_mask,
            "colormap": colormap,
        }

    def heatmap_to_intensity(self, thermal_heatmap: np.ndarray) -> np.ndarray:
        """
        Recover the 0..255 intensity from a JET heatmap by nearest-color matching.

        This allows us to compute temperatures/fire-mask without transmitting grayscale.
        """
        pixels = thermal_heatmap.reshape(-1, 3).astype(np.int32)  # (N,3)
        lut = self._jet_lut_bgr.astype(np.int32)                  # (256,3)
        # Squared Euclidean distance in BGR space. N is small (24*32=768).
        dist2 = np.sum((pixels[:, None, :] - lut[None, :, :]) ** 2, axis=2)  # (N,256)
        intensity_idx = np.argmin(dist2, axis=1).astype(np.uint8)
        return intensity_idx.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))

    def upscale_intensity(self, intensity: np.ndarray) -> np.ndarray:
        """
        Upscale 32x24 intensity to 1280x720 using bicubic interpolation.

        Args:
            intensity: Input intensity (24, 32) uint8.

        Returns:
            Upscaled image (720, 1280) uint8.
        """
        return cv2.resize(
            intensity,
            self._target_size,
            interpolation=cv2.INTER_CUBIC,
        )

    def upscale_heatmap(self, thermal_heatmap: np.ndarray) -> np.ndarray:
        """
        Upscale 32x24 heatmap to 1280x720 using bicubic interpolation.
        """
        return cv2.resize(
            thermal_heatmap,
            self._target_size,
            interpolation=cv2.INTER_CUBIC,
        )

    def align(self, thermal_upscaled: np.ndarray) -> np.ndarray:
        """
        Apply homography warp to align a thermal image with RGB frame.

        Args:
            thermal_upscaled: Upscaled thermal image (720, 1280) uint8 or (720, 1280, 3).

        Returns:
            Warped image aligned to RGB coordinate system.
        """
        # If homography is identity, skip warp for efficiency
        if np.allclose(self.homography, np.eye(3)):
            return thermal_upscaled

        return cv2.warpPerspective(
            thermal_upscaled,
            self.homography,
            self._target_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def compute_fire_mask(self, aligned_intensity: np.ndarray) -> np.ndarray:
        """
        Generate binary fire mask: pixels exceeding 50°C threshold.

        Args:
            aligned_intensity: Aligned intensity thermal image uint8.

        Returns:
            Binary mask (720, 1280) uint8, 255 = fire, 0 = no fire.
        """
        _, mask = cv2.threshold(
            aligned_intensity, FIRE_THRESHOLD_GRAY, 255, cv2.THRESH_BINARY
        )
        return mask

    def extract_region_temps(
        self, temperatures: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> dict:
        """
        Extract thermal statistics for a bounding box region.

        Args:
            temperatures: Full-resolution temperature map (720, 1280) float32.
            bbox: (x1, y1, x2, y2) bounding box coordinates.

        Returns:
            Dictionary with max_temp, mean_temp, hot_pixel_ratio.
        """
        x1, y1, x2, y2 = bbox
        # Clamp to valid range
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(temperatures.shape[1], int(x2))
        y2 = min(temperatures.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            return {"max_temp": 0.0, "mean_temp": 0.0, "hot_pixel_ratio": 0.0}

        region = temperatures[y1:y2, x1:x2]
        total_pixels = region.size

        if total_pixels == 0:
            return {"max_temp": 0.0, "mean_temp": 0.0, "hot_pixel_ratio": 0.0}

        max_temp = float(np.max(region))
        mean_temp = float(np.mean(region))
        hot_pixels = int(np.sum(region > FIRE_THRESHOLD_TEMP))
        hot_pixel_ratio = hot_pixels / total_pixels

        return {
            "max_temp": round(max_temp, 1),
            "mean_temp": round(mean_temp, 1),
            "hot_pixel_ratio": round(hot_pixel_ratio, 3),
        }

    def set_homography(self, matrix: np.ndarray):
        """Update the homography calibration matrix."""
        if matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")
        self.homography = matrix.astype(np.float64)
        logger.info("Homography matrix updated")

    @staticmethod
    def intensity_to_temps(intensity: np.ndarray) -> np.ndarray:
        """
        Convert JET intensity values back to approximate temperatures.

        Formula: temp = gray / 255 * 130 + 20

        Args:
            intensity: uint8 intensity image.

        Returns:
            float32 temperature map in °C.
        """
        return (intensity.astype(np.float32) / 255.0) * THERMAL_RANGE + THERMAL_MIN_TEMP
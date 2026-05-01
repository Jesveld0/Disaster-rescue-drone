"""
test_thermal.py — Unit tests for thermal processing.

Tests:
- Heatmap-to-intensity recovery (JET colormap inversion)
- Intensity-to-temperature conversion
- Upscaling dimensions
- Fire mask thresholding
- Region temperature extraction
- Full processing pipeline with heatmap input
"""

import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, ".")

from config import (
    THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_MIN_TEMP, THERMAL_RANGE,
    FIRE_THRESHOLD_TEMP, FIRE_THRESHOLD_GRAY, RGB_WIDTH, RGB_HEIGHT,
)
from ground_station.thermal_processing import ThermalProcessor


class TestIntensityConversion:
    """Tests for heatmap ↔ intensity ↔ temperature conversion."""

    def setup_method(self):
        self.processor = ThermalProcessor()

    def test_intensity_to_temp_min(self):
        """Intensity 0 should map to minimum temperature (20°C)."""
        intensity = np.array([[0]], dtype=np.uint8)
        temps = ThermalProcessor.intensity_to_temps(intensity)
        assert abs(temps[0, 0] - THERMAL_MIN_TEMP) < 0.1

    def test_intensity_to_temp_max(self):
        """Intensity 255 should map to maximum temperature (150°C)."""
        intensity = np.array([[255]], dtype=np.uint8)
        temps = ThermalProcessor.intensity_to_temps(intensity)
        expected = THERMAL_MIN_TEMP + THERMAL_RANGE  # 150.0
        assert abs(temps[0, 0] - expected) < 0.6

    def test_intensity_to_temp_fire_threshold(self):
        """FIRE_THRESHOLD_GRAY should map to approximately 50°C."""
        intensity = np.array([[FIRE_THRESHOLD_GRAY]], dtype=np.uint8)
        temps = ThermalProcessor.intensity_to_temps(intensity)
        # uint8 quantization introduces ≤0.51°C error (130/255)
        assert abs(temps[0, 0] - FIRE_THRESHOLD_TEMP) < 1.0

    def test_intensity_monotonic(self):
        """Temperature should increase monotonically with intensity."""
        intensity = np.arange(0, 256, dtype=np.uint8).reshape(1, -1)
        temps = ThermalProcessor.intensity_to_temps(intensity)
        assert np.all(np.diff(temps.flatten()) >= 0)

    def test_heatmap_to_intensity_roundtrip(self):
        """Creating a JET heatmap and recovering intensity should be close."""
        # Create a known intensity image at thermal sensor dimensions
        original = np.random.randint(0, 256, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        # Apply JET colormap (simulate what the edge device does)
        heatmap = cv2.applyColorMap(original, cv2.COLORMAP_JET)
        # Recover intensity
        recovered = self.processor.heatmap_to_intensity(heatmap)
        # Allow small error from nearest-color matching
        diff = np.abs(original.astype(int) - recovered.astype(int))
        assert np.mean(diff) < 5.0, f"Mean intensity error too large: {np.mean(diff)}"

    def test_heatmap_to_intensity_shape(self):
        """Recovered intensity should be 2D (H, W) from 3-channel input."""
        heatmap = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
        intensity = self.processor.heatmap_to_intensity(heatmap)
        assert intensity.shape == (THERMAL_HEIGHT, THERMAL_WIDTH)
        assert intensity.dtype == np.uint8


class TestThermalProcessor:
    """Tests for the ground station thermal processing pipeline."""

    def setup_method(self):
        self.processor = ThermalProcessor()

    def test_upscale_intensity_dimensions(self):
        """Upscaled intensity image should be RGB resolution."""
        thermal = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        upscaled = self.processor.upscale_intensity(thermal)
        assert upscaled.shape == (RGB_HEIGHT, RGB_WIDTH)

    def test_upscale_heatmap_dimensions(self):
        """Upscaled heatmap should be RGB resolution with 3 channels."""
        heatmap = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
        upscaled = self.processor.upscale_heatmap(heatmap)
        assert upscaled.shape == (RGB_HEIGHT, RGB_WIDTH, 3)

    def test_fire_mask_threshold(self):
        """Fire mask should correctly threshold at 50°C gray value."""
        # Create intensity with half above threshold
        intensity = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        intensity[:, THERMAL_WIDTH // 2:] = FIRE_THRESHOLD_GRAY + 10  # Above threshold
        intensity[:, :THERMAL_WIDTH // 2] = FIRE_THRESHOLD_GRAY - 10  # Below threshold

        upscaled = self.processor.upscale_intensity(intensity)
        mask = self.processor.compute_fire_mask(upscaled)

        # Right half should be fire, left half should not
        assert mask.shape == (RGB_HEIGHT, RGB_WIDTH)
        right_fire = np.mean(mask[:, RGB_WIDTH * 3 // 4:] > 0)
        left_fire = np.mean(mask[:, :RGB_WIDTH // 4] > 0)
        assert right_fire > 0.8  # Most right pixels should be fire
        assert left_fire < 0.2   # Most left pixels should not be fire

    def test_extract_region_temps_valid(self):
        """Should extract correct stats from a region."""
        temps = np.ones((RGB_HEIGHT, RGB_WIDTH), dtype=np.float32) * 30.0
        # Put a hot spot in a known region
        temps[100:200, 100:200] = 80.0

        stats = self.processor.extract_region_temps(temps, (100, 100, 200, 200))
        assert stats["max_temp"] == 80.0
        assert stats["mean_temp"] == 80.0
        assert stats["hot_pixel_ratio"] == 1.0  # All pixels > 50°C

    def test_extract_region_temps_cold(self):
        """Cold region should have zero hot pixel ratio."""
        temps = np.ones((RGB_HEIGHT, RGB_WIDTH), dtype=np.float32) * 25.0
        stats = self.processor.extract_region_temps(temps, (0, 0, 100, 100))
        assert stats["hot_pixel_ratio"] == 0.0

    def test_extract_region_temps_invalid_bbox(self):
        """Invalid bounding box should return zeros."""
        temps = np.ones((RGB_HEIGHT, RGB_WIDTH), dtype=np.float32) * 50.0
        stats = self.processor.extract_region_temps(temps, (100, 100, 50, 50))
        assert stats["max_temp"] == 0.0

    def test_process_full_pipeline(self):
        """Full process() should return all expected outputs with heatmap input."""
        # Generate a synthetic JET heatmap (simulating what the edge sends)
        intensity = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)

        result = self.processor.process(heatmap)

        assert "upscaled" in result
        assert "aligned" in result
        assert "temperatures" in result
        assert "fire_mask" in result
        assert "colormap" in result

        assert result["upscaled"].shape == (RGB_HEIGHT, RGB_WIDTH)
        assert result["aligned"].shape == (RGB_HEIGHT, RGB_WIDTH)
        assert result["temperatures"].shape == (RGB_HEIGHT, RGB_WIDTH)
        assert result["fire_mask"].shape == (RGB_HEIGHT, RGB_WIDTH)
        assert result["colormap"].shape == (RGB_HEIGHT, RGB_WIDTH, 3)

    def test_identity_homography_preserves_image(self):
        """With identity homography, align should not change the image."""
        thermal = np.random.randint(0, 255, (RGB_HEIGHT, RGB_WIDTH), dtype=np.uint8)
        aligned = self.processor.align(thermal)
        np.testing.assert_array_equal(aligned, thermal)

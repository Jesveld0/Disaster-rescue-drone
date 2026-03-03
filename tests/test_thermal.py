"""
test_thermal.py — Unit tests for thermal processing.

Tests:
- Grayscale conversion formula
- Temperature recovery from grayscale
- Upscaling dimensions
- Fire mask thresholding
- Region temperature extraction
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from config import (
    THERMAL_WIDTH, THERMAL_HEIGHT, THERMAL_MIN_TEMP, THERMAL_RANGE,
    FIRE_THRESHOLD_TEMP, FIRE_THRESHOLD_GRAY, RGB_WIDTH, RGB_HEIGHT,
)
from edge.thermal_capture import ThermalCamera
from ground_station.thermal_processing import ThermalProcessor


class TestThermalConversion:
    """Tests for temperature ↔ grayscale conversion."""

    def test_min_temp_maps_to_zero(self):
        """20°C should map to grayscale 0."""
        temps = np.array([[THERMAL_MIN_TEMP]], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        assert gray[0, 0] == 0

    def test_max_temp_maps_to_255(self):
        """150°C should map to grayscale 255."""
        temps = np.array([[150.0]], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        assert gray[0, 0] == 255

    def test_fire_threshold_gray_value(self):
        """50°C should map to the expected grayscale value."""
        temps = np.array([[FIRE_THRESHOLD_TEMP]], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        expected = int(np.clip((50.0 - 20.0) / 130.0 * 255, 0, 255))
        assert gray[0, 0] == expected
        assert gray[0, 0] == FIRE_THRESHOLD_GRAY

    def test_below_min_clamps_to_zero(self):
        """Temperatures below 20°C should clamp to 0."""
        temps = np.array([[-10.0, 0.0, 10.0]], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        assert np.all(gray == 0)

    def test_above_max_clamps_to_255(self):
        """Temperatures above 150°C should clamp to 255."""
        temps = np.array([[200.0, 300.0]], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        assert np.all(gray == 255)

    def test_roundtrip_conversion(self):
        """temp → gray → temp should approximately preserve values."""
        original_temps = np.array([25.0, 50.0, 80.0, 120.0], dtype=np.float32)
        gray = ThermalCamera.temps_to_grayscale(original_temps.reshape(1, -1))
        recovered = ThermalCamera.grayscale_to_temps(gray)

        # uint8 quantization introduces ≤0.51°C error (130/255)
        np.testing.assert_allclose(
            recovered.flatten(), original_temps, atol=0.6
        )

    def test_full_range_monotonic(self):
        """Grayscale values should be monotonically increasing with temperature."""
        temps = np.linspace(20.0, 150.0, 100).reshape(1, -1).astype(np.float32)
        gray = ThermalCamera.temps_to_grayscale(temps)
        assert np.all(np.diff(gray.flatten()) >= 0)


class TestThermalProcessor:
    """Tests for the ground station thermal processing pipeline."""

    def setup_method(self):
        self.processor = ThermalProcessor()

    def test_upscale_dimensions(self):
        """Upscaled image should be RGB resolution."""
        thermal = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        upscaled = self.processor.upscale(thermal)
        assert upscaled.shape == (RGB_HEIGHT, RGB_WIDTH)

    def test_fire_mask_threshold(self):
        """Fire mask should correctly threshold at 50°C gray value."""
        # Create thermal with half above threshold
        thermal = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        thermal[:, THERMAL_WIDTH // 2:] = FIRE_THRESHOLD_GRAY + 10  # Above threshold
        thermal[:, :THERMAL_WIDTH // 2] = FIRE_THRESHOLD_GRAY - 10  # Below threshold

        upscaled = self.processor.upscale(thermal)
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
        """Full process() should return all expected outputs."""
        thermal = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)
        result = self.processor.process(thermal)

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

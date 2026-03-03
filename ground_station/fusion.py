"""
fusion.py — Thermal + RGB fusion logic for human-in-fire classification.

Combines YOLO detection results with thermal data to:
1. Classify persons as being in fire (HUMAN_IN_FIRE)
2. Cross-validate fire detections (thermal + visual)
3. Generate per-person thermal statistics
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import FIRE_THRESHOLD_TEMP, HOT_PIXEL_RATIO_THRESHOLD
from ground_station.detector import Detection, DetectionResult
from ground_station.thermal_processing import ThermalProcessor

logger = logging.getLogger(__name__)


@dataclass
class PersonAnalysis:
    """Thermal analysis result for a detected person."""
    detection: Detection
    max_temp: float = 0.0
    mean_temp: float = 0.0
    hot_pixel_ratio: float = 0.0
    in_fire: bool = False


@dataclass
class FireZone:
    """Identified fire zone with combined visual + thermal evidence."""
    bbox: tuple[int, int, int, int]
    confidence: float
    thermal_confirmed: bool = False
    max_temp: float = 0.0


@dataclass
class FusionResult:
    """Combined results from thermal-RGB fusion analysis."""
    persons: list[PersonAnalysis] = field(default_factory=list)
    fire_zones: list[FireZone] = field(default_factory=list)
    humans_in_fire: int = 0
    any_fire: bool = False


class ThermalFusion:
    """
    Fuses YOLO detection results with thermal data for advanced classification.

    Decision Rule for HUMAN_IN_FIRE:
        IF person_detected
        AND max_temp > 50°C
        AND hot_pixel_ratio > 0.2
        THEN classify as HUMAN_IN_FIRE

    Fire Cross-Validation:
        Fire is confirmed only if BOTH:
        - YOLO detects fire visually
        - Thermal data exceeds threshold in the same region
        This reduces false positives from heated asphalt or sunlight.
    """

    def __init__(self, thermal_processor: ThermalProcessor):
        """
        Args:
            thermal_processor: Instance for extracting thermal region stats.
        """
        self.thermal_processor = thermal_processor

    def analyze(
        self,
        detections: DetectionResult,
        temperatures: np.ndarray,
        fire_mask: np.ndarray,
    ) -> FusionResult:
        """
        Perform full fusion analysis on a frame.

        Args:
            detections: YOLO detection results.
            temperatures: Full-resolution temperature map (720, 1280) float32.
            fire_mask: Binary fire mask (720, 1280) uint8.

        Returns:
            FusionResult with classified persons and fire zones.
        """
        result = FusionResult()

        # Analyze each detected person
        for person_det in detections.persons:
            analysis = self._analyze_person(person_det, temperatures)
            result.persons.append(analysis)
            if analysis.in_fire:
                result.humans_in_fire += 1

        # Analyze fire detections
        for fire_det in detections.fires:
            fire_zone = self._analyze_fire(fire_det, temperatures, fire_mask)
            result.fire_zones.append(fire_zone)
            if fire_zone.thermal_confirmed:
                result.any_fire = True

        # Also check for thermal-only fire zones (no YOLO fire detection)
        thermal_fire_zones = self._detect_thermal_fire_zones(
            fire_mask, temperatures, detections
        )
        result.fire_zones.extend(thermal_fire_zones)

        if result.fire_zones:
            result.any_fire = True

        return result

    def _analyze_person(
        self, detection: Detection, temperatures: np.ndarray
    ) -> PersonAnalysis:
        """
        Analyze thermal conditions for a detected person.

        Applies the HUMAN_IN_FIRE decision rule.
        """
        stats = self.thermal_processor.extract_region_temps(
            temperatures, detection.bbox
        )

        in_fire = (
            stats["max_temp"] > FIRE_THRESHOLD_TEMP
            and stats["hot_pixel_ratio"] > HOT_PIXEL_RATIO_THRESHOLD
        )

        if in_fire:
            logger.warning(
                "🔥 HUMAN IN FIRE detected at bbox %s — "
                "max_temp=%.1f°C, hot_ratio=%.3f",
                detection.bbox, stats["max_temp"], stats["hot_pixel_ratio"],
            )

        return PersonAnalysis(
            detection=detection,
            max_temp=stats["max_temp"],
            mean_temp=stats["mean_temp"],
            hot_pixel_ratio=stats["hot_pixel_ratio"],
            in_fire=in_fire,
        )

    def _analyze_fire(
        self,
        detection: Detection,
        temperatures: np.ndarray,
        fire_mask: np.ndarray,
    ) -> FireZone:
        """
        Cross-validate a YOLO fire detection with thermal data.

        Fire is confirmed only if thermal data also shows elevated temperatures
        in the same region, reducing false positives.
        """
        stats = self.thermal_processor.extract_region_temps(
            temperatures, detection.bbox
        )

        # Check thermal confirmation
        x1, y1, x2, y2 = detection.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(fire_mask.shape[1], x2)
        y2 = min(fire_mask.shape[0], y2)

        fire_region = fire_mask[y1:y2, x1:x2]
        fire_pixel_ratio = np.mean(fire_region > 0) if fire_region.size > 0 else 0.0

        # Cross-validation: thermal must confirm YOLO fire
        thermal_confirmed = (
            stats["max_temp"] > FIRE_THRESHOLD_TEMP
            and fire_pixel_ratio > 0.1  # At least 10% fire pixels in thermal
        )

        return FireZone(
            bbox=detection.bbox,
            confidence=detection.confidence,
            thermal_confirmed=thermal_confirmed,
            max_temp=stats["max_temp"],
        )

    def _detect_thermal_fire_zones(
        self,
        fire_mask: np.ndarray,
        temperatures: np.ndarray,
        detections: DetectionResult,
    ) -> list[FireZone]:
        """
        Detect fire zones from thermal data alone (no YOLO fire detection).

        Uses contour detection on the fire mask to find connected hot regions.
        These are secondary fire indicators when YOLO doesn't detect fire visually.
        """
        import cv2

        zones = []

        # Find contours in the fire mask
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Ignore tiny noise regions
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)

            # Check if this overlaps with any existing YOLO fire detection
            already_detected = any(
                self._bbox_overlap(bbox, f.bbox) > 0.3
                for f in detections.fires
            )
            if already_detected:
                continue

            stats = self.thermal_processor.extract_region_temps(temperatures, bbox)

            zones.append(FireZone(
                bbox=bbox,
                confidence=0.5,  # Lower confidence for thermal-only detection
                thermal_confirmed=True,
                max_temp=stats["max_temp"],
            ))

        return zones

    @staticmethod
    def _bbox_overlap(
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU (Intersection over Union) between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

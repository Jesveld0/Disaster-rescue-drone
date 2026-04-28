"""
test_tracker.py — Tests for DeepSORT human tracker.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from ground_station.detector import Detection, DetectionResult
from ground_station.tracker import HumanTracker, TrackedPerson, TrackingResult


class TestHumanTracker:
    """Test DeepSORT tracker initialization and basic operations."""

    def test_tracker_initialization(self):
        """Tracker should initialize (even if DeepSORT not installed)."""
        tracker = HumanTracker()
        # Should not crash — might be loaded or not depending on deps

    def test_empty_detections(self):
        """Tracker should handle empty detections gracefully."""
        tracker = HumanTracker()
        detections = DetectionResult()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = tracker.update(detections, frame)
        assert isinstance(result, TrackingResult)
        assert result.active_tracks == 0

    def test_tracking_result_dataclass(self):
        """TrackingResult should have correct default values."""
        result = TrackingResult()
        assert result.tracked_persons == []
        assert result.active_tracks == 0
        assert result.confirmed_tracks == 0
        assert result.tracking_time_ms == 0.0

    def test_tracked_person_dataclass(self):
        """TrackedPerson should hold all required fields."""
        person = TrackedPerson(
            track_id=1,
            bbox=(100, 200, 300, 400),
            confidence=0.95,
            is_confirmed=True,
            age=5,
        )
        assert person.track_id == 1
        assert person.bbox == (100, 200, 300, 400)
        assert person.confidence == 0.95
        assert person.is_confirmed is True
        assert person.age == 5
        assert person.detection is None

    def test_iou_computation(self):
        """IoU computation should work correctly."""
        # Identical boxes → IoU = 1.0
        iou = HumanTracker._compute_iou(
            (0, 0, 100, 100), (0, 0, 100, 100)
        )
        assert abs(iou - 1.0) < 0.001

        # No overlap → IoU = 0.0
        iou = HumanTracker._compute_iou(
            (0, 0, 50, 50), (100, 100, 200, 200)
        )
        assert iou == 0.0

        # Partial overlap
        iou = HumanTracker._compute_iou(
            (0, 0, 100, 100), (50, 50, 150, 150)
        )
        assert 0.0 < iou < 1.0

    def test_single_detection_creates_track(self):
        """A single person detection should create a track."""
        tracker = HumanTracker()
        if not tracker.is_loaded:
            return  # Skip if DeepSORT not installed

        person_det = Detection(
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            class_name="person",
            category="person",
        )
        detections = DetectionResult(persons=[person_det])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = tracker.update(detections, frame)
        assert result.active_tracks > 0

    def test_multiple_frames_track_persistence(self):
        """Track IDs should persist across multiple frames."""
        tracker = HumanTracker(n_init=1)  # Confirm after 1 hit
        if not tracker.is_loaded:
            return  # Skip if DeepSORT not installed

        person_det = Detection(
            bbox=(100, 100, 200, 300),
            confidence=0.9,
            class_id=0,
            class_name="person",
            category="person",
        )
        detections = DetectionResult(persons=[person_det])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First frame
        result1 = tracker.update(detections, frame)

        # Second frame (same detection)
        result2 = tracker.update(detections, frame)

        if result1.tracked_persons and result2.tracked_persons:
            # Track IDs should match
            assert result1.tracked_persons[0].track_id == result2.tracked_persons[0].track_id

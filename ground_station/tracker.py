"""
tracker.py — DeepSORT-based human tracking for the ground station.

Provides persistent track IDs for detected persons across frames,
enabling continuous human tracking even with temporary occlusions
or detection gaps.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import TRACKER_MAX_AGE, TRACKER_N_INIT, TRACKER_MAX_IOU_DISTANCE
from ground_station.detector import Detection, DetectionResult

logger = logging.getLogger(__name__)

# Import DeepSORT (lazy to handle missing dependency gracefully)
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except ImportError:
    HAS_DEEPSORT = False
    logger.warning("deep-sort-realtime not installed — human tracking disabled")


@dataclass
class TrackedPerson:
    """A tracked person with a persistent ID across frames."""
    track_id: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    is_confirmed: bool                 # True after TRACKER_N_INIT hits
    age: int                           # Frames since track creation
    detection: Optional[Detection] = None  # Original detection if available


@dataclass
class TrackingResult:
    """Tracking output for a single frame."""
    tracked_persons: list[TrackedPerson] = field(default_factory=list)
    active_tracks: int = 0
    confirmed_tracks: int = 0
    tracking_time_ms: float = 0.0


class HumanTracker:
    """
    DeepSORT-based multi-object tracker for persistent human tracking.

    Takes person detections from RF-DETR and assigns persistent track IDs
    that survive across frames even during brief occlusions.

    Usage:
        tracker = HumanTracker()
        tracking_result = tracker.update(detections, bgr_frame)
    """

    def __init__(
        self,
        max_age: int = TRACKER_MAX_AGE,
        n_init: int = TRACKER_N_INIT,
        max_iou_distance: float = TRACKER_MAX_IOU_DISTANCE,
    ):
        """
        Args:
            max_age: Maximum frames to keep an unmatched track alive.
            n_init: Number of consecutive hits before a track is confirmed.
            max_iou_distance: Maximum IoU distance for matching detections to tracks.
        """
        self.tracker = None
        self._max_age = max_age
        self._n_init = n_init
        self._max_iou_distance = max_iou_distance

        self._init_tracker()

    def _init_tracker(self):
        """Initialize the DeepSORT tracker."""
        if not HAS_DEEPSORT:
            logger.error("Cannot initialize tracker — deep-sort-realtime not installed")
            return

        try:
            self.tracker = DeepSort(
                max_age=self._max_age,
                n_init=self._n_init,
                max_iou_distance=self._max_iou_distance,
            )
            logger.info(
                "DeepSORT tracker initialized (max_age=%d, n_init=%d, max_iou=%.2f)",
                self._max_age, self._n_init, self._max_iou_distance,
            )
        except Exception as e:
            logger.error("Failed to initialize DeepSORT tracker: %s", e)
            self.tracker = None

    def update(
        self, detections: DetectionResult, frame: np.ndarray
    ) -> TrackingResult:
        """
        Update tracker with new detections and return tracked persons.

        Args:
            detections: RF-DETR detection results for the current frame.
            frame: BGR image (H, W, 3) — used for feature extraction.

        Returns:
            TrackingResult with tracked persons and statistics.
        """
        result = TrackingResult()

        if self.tracker is None:
            return result

        start = time.monotonic()

        try:
            # Convert person detections to DeepSORT format
            # Format: list of ([left, top, width, height], confidence, detection_class)
            raw_detections = []
            detection_map = {}  # Map index to original Detection

            for i, person in enumerate(detections.persons):
                x1, y1, x2, y2 = person.bbox
                w = x2 - x1
                h = y2 - y1
                raw_detections.append(
                    ([x1, y1, w, h], person.confidence, "person")
                )
                detection_map[i] = person

            # Update tracker
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)

            # Process tracked objects
            for track in tracks:
                track_id = track.track_id
                is_confirmed = track.is_confirmed()

                # Get bounding box in (left, top, right, bottom) format
                ltrb = track.to_ltrb()
                bbox = (
                    int(max(0, ltrb[0])),
                    int(max(0, ltrb[1])),
                    int(ltrb[2]),
                    int(ltrb[3]),
                )

                # Find the matching original detection if available
                original_det = None
                if track.det_conf is not None:
                    confidence = float(track.det_conf)
                else:
                    confidence = 0.0

                # Try to match back to original detection by IoU
                best_iou = 0.0
                for det in detections.persons:
                    iou = self._compute_iou(bbox, det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        original_det = det
                        if det.confidence > confidence:
                            confidence = det.confidence

                tracked = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                    is_confirmed=is_confirmed,
                    age=track.age,
                    detection=original_det,
                )
                result.tracked_persons.append(tracked)

                if is_confirmed:
                    result.confirmed_tracks += 1

            result.active_tracks = len(result.tracked_persons)
            result.tracking_time_ms = round(
                (time.monotonic() - start) * 1000, 1
            )

        except Exception as e:
            logger.error("Tracking error: %s", e)
            result.tracking_time_ms = (time.monotonic() - start) * 1000

        return result

    @staticmethod
    def _compute_iou(
        bbox1: tuple[int, int, int, int],
        bbox2: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
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

    @property
    def is_loaded(self) -> bool:
        """Check if the tracker is initialized and ready."""
        return self.tracker is not None

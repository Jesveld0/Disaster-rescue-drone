"""
detector.py — RT-DETR object detection for the ground station.

Uses Ultralytics RT-DETR (rtdetr-l.pt) — the same model confirmed working
in main.py. Detects persons (COCO class 0) with high recall, and optionally
obstacle classes.

Reference: main.py (uses RTDETR with classes=[0], conf=0.25, ultralytics API)
"""

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from config import RTDETR_WEIGHTS, RTDETR_CONFIDENCE, RTDETR_PERSON_CONFIDENCE, OBSTACLE_COCO_IDS

logger = logging.getLogger(__name__)

try:
    from ultralytics import RTDETR
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("ultralytics not installed — RT-DETR detection disabled")

# COCO 0-indexed class names (Ultralytics standard)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

PERSON_CLASS_ID = 0  # Ultralytics uses 0-indexed COCO; person = 0


@dataclass
class Detection:
    """Single detection result."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    category: str  # 'person', 'fire', 'obstacle'


@dataclass
class DetectionResult:
    """Aggregated detection results for a single frame."""
    persons: list[Detection] = field(default_factory=list)
    fires: list[Detection] = field(default_factory=list)
    obstacles: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0


class RTDETRDetector:
    """
    Ultralytics RT-DETR detector — mirrors the approach in main.py.

    Uses rtdetr-l.pt (large model) for best person detection accuracy.
    Filters to person class at a low confidence threshold (0.25) to
    maximise recall in rescue scenarios.

    Usage:
        detector = RTDETRDetector()
        result = detector.detect(bgr_frame)
    """

    # Physical outdoor obstacle class IDs (0-indexed COCO)
    OBSTACLE_CLASS_IDS = {
        1,   # bicycle
        2,   # car
        3,   # motorcycle
        5,   # bus
        7,   # truck
        9,   # traffic light
        11,  # stop sign
    }

    def __init__(self, weights: str = None, device: str = None):
        self.model = None

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self._weights = weights or RTDETR_WEIGHTS
        self._load_model()

    def _load_model(self):
        if not HAS_ULTRALYTICS:
            logger.error("Cannot load RT-DETR — ultralytics not installed")
            return
        try:
            logger.info("Loading RT-DETR: %s on %s", self._weights, self.device)
            self.model = RTDETR(self._weights)
            # Warm up
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy, device=self.device,
                classes=[PERSON_CLASS_ID], conf=RTDETR_PERSON_CONFIDENCE,
                verbose=False,
            )
            logger.info(
                "RT-DETR loaded. person_conf=%.2f  obstacle_conf=%.2f  device=%s",
                RTDETR_PERSON_CONFIDENCE, RTDETR_CONFIDENCE, self.device,
            )
        except Exception as e:
            logger.error("Failed to load RT-DETR: %s", e)
            self.model = None

    def detect(self, bgr_frame: np.ndarray) -> DetectionResult:
        """
        Run detection on a BGR frame.

        Runs two passes (both are fast — Ultralytics batches internally):
          Pass 1: persons only at low conf (0.25) — maximise recall
          Pass 2: obstacle classes at normal conf (0.3)

        Returns DetectionResult with persons, fires, obstacles lists.
        """
        result = DetectionResult()
        if self.model is None:
            return result

        start = time.monotonic()

        try:
            # --- Pass 1: persons only, low threshold ---
            res_persons = self.model.predict(
                source=bgr_frame,
                device=self.device,
                classes=[PERSON_CLASS_ID],
                conf=RTDETR_PERSON_CONFIDENCE,
                verbose=False,
            )[0]

            if res_persons.boxes is not None and len(res_persons.boxes) > 0:
                xyxy  = res_persons.boxes.xyxy.cpu().numpy()
                confs = res_persons.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    result.persons.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(c),
                        class_id=PERSON_CLASS_ID,
                        class_name="person",
                        category="person",
                    ))

            # --- Pass 2: obstacle classes, normal threshold ---
            obstacle_ids = list(self.OBSTACLE_CLASS_IDS)
            res_obs = self.model.predict(
                source=bgr_frame,
                device=self.device,
                classes=obstacle_ids,
                conf=RTDETR_CONFIDENCE,
                verbose=False,
            )[0]

            if res_obs.boxes is not None and len(res_obs.boxes) > 0:
                xyxy    = res_obs.boxes.xyxy.cpu().numpy()
                confs   = res_obs.boxes.conf.cpu().numpy()
                cls_ids = res_obs.boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, cls_ids):
                    class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                    result.obstacles.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(c),
                        class_id=int(cls_id),
                        class_name=class_name,
                        category="obstacle",
                    ))

        except Exception as e:
            logger.error("RT-DETR detection error: %s", e)

        result.inference_time_ms = round((time.monotonic() - start) * 1000, 1)

        if result.persons:
            logger.debug(
                "Detected %d person(s), %d obstacle(s) in %.0fms",
                len(result.persons), len(result.obstacles), result.inference_time_ms,
            )

        return result

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# Keep old name as alias so pipeline.py import still works
RFDETRDetector = RTDETRDetector

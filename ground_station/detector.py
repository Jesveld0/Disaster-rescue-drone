"""
detector.py — RF-DETR object detection for the ground station.

Detects persons, fire, and obstacle classes using RF-DETR model
with COCO pre-trained weights and CUDA acceleration.

RF-DETR is built on a DINOv2 vision transformer backbone and provides
state-of-the-art real-time detection performance.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    RFDETR_MODEL_SIZE, RFDETR_CONFIDENCE,
    PERSON_CLASS_ID, FIRE_CLASS_LABEL, OBSTACLE_CLASSES,
)

logger = logging.getLogger(__name__)

# Import RF-DETR (lazy to handle missing dependency gracefully)
try:
    from rfdetr import RFDETRBase
    try:
        from rfdetr import RFDETRSmall
    except ImportError:
        RFDETRSmall = None
    try:
        from rfdetr import RFDETRLarge
    except ImportError:
        RFDETRLarge = None
    from rfdetr.util.coco_classes import COCO_CLASSES
    HAS_RFDETR = True
except ImportError:
    HAS_RFDETR = False
    COCO_CLASSES = []
    logger.warning("rfdetr not installed — RF-DETR detection disabled")


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


class RFDETRDetector:
    """
    RF-DETR-based object detector with CUDA support.

    Detects:
    - Persons (COCO class 0)
    - Fire (custom class or placeholder)
    - Obstacles (vehicles, furniture, etc.)

    Usage:
        detector = RFDETRDetector()
        result = detector.detect(bgr_frame)
    """

    # COCO class names that map to "obstacle" category
    # Only physical outdoor obstacles the drone might collide with
    OBSTACLE_COCO_NAMES = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "traffic light", "stop sign", "fire hydrant",
        "suitcase", "umbrella",
    }

    MODEL_CLASSES = {
        "base": RFDETRBase if HAS_RFDETR else None,
        "small": RFDETRSmall if HAS_RFDETR else None,
        "large": RFDETRLarge if HAS_RFDETR else None,
    }

    def __init__(self, model_size: str = RFDETR_MODEL_SIZE, device: str = None):
        """
        Args:
            model_size: RF-DETR model variant ('base', 'small', 'large').
            device: Compute device ('cuda', 'cpu', or None for auto).
        """
        self.model_size = model_size
        self.model = None

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        """Load the RF-DETR model."""
        if not HAS_RFDETR:
            logger.error("Cannot load RF-DETR — rfdetr not installed")
            return

        try:
            model_cls = self.MODEL_CLASSES.get(self.model_size)
            if model_cls is None:
                logger.warning(
                    "RF-DETR '%s' not available, falling back to 'base'",
                    self.model_size,
                )
                model_cls = RFDETRBase

            logger.info(
                "Loading RF-DETR model: %s on %s",
                model_cls.__name__, self.device,
            )
            self.model = model_cls()

            # Warm up the model with a dummy inference
            dummy = Image.fromarray(
                np.zeros((640, 640, 3), dtype=np.uint8)
            )
            self.model.predict(dummy, threshold=RFDETR_CONFIDENCE)

            logger.info(
                "RF-DETR model loaded successfully. COCO classes: %d, Device: %s",
                len(COCO_CLASSES), self.device,
            )

        except Exception as e:
            logger.error("Failed to load RF-DETR model: %s", e)
            self.model = None

    def detect(self, bgr_frame: np.ndarray) -> DetectionResult:
        """
        Run object detection on a BGR frame.

        Args:
            bgr_frame: OpenCV BGR image (H, W, 3).

        Returns:
            DetectionResult with categorized detections.
        """
        result = DetectionResult()

        if self.model is None:
            return result

        start = time.monotonic()

        try:
            # Convert BGR to RGB PIL Image for RF-DETR
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Run inference — returns supervision.Detections object
            detections = self.model.predict(
                pil_image,
                threshold=RFDETR_CONFIDENCE,
            )

            inference_time = (time.monotonic() - start) * 1000
            result.inference_time_ms = round(inference_time, 1)

            if detections is None or len(detections) == 0:
                return result

            # Extract detection arrays from supervision.Detections
            boxes = detections.xyxy            # (N, 4) numpy array
            confidences = detections.confidence  # (N,) numpy array
            class_ids = detections.class_id      # (N,) numpy array

            for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, bbox)
                cls_id = int(cls_id)

                # Get class name from COCO classes
                if 0 <= cls_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[cls_id]
                else:
                    class_name = f"class_{cls_id}"

                logger.debug(
                    "Raw detection: id=%d name='%s' conf=%.2f",
                    cls_id, class_name, conf,
                )

                detection = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=class_name,
                    category=self._categorize(class_name, cls_id),
                )

                # Route to appropriate list
                if detection.category == "person":
                    result.persons.append(detection)
                elif detection.category == "fire":
                    result.fires.append(detection)
                elif detection.category == "obstacle":
                    result.obstacles.append(detection)

        except Exception as e:
            logger.error("RF-DETR detection error: %s", e)
            result.inference_time_ms = (time.monotonic() - start) * 1000

        return result

    def _categorize(self, class_name: str, class_id: int) -> str:
        """
        Categorize a detected class into person/fire/obstacle.

        Checks class name first (robust across RF-DETR version differences
        in whether COCO classes are 0-indexed or 1-indexed).
        """
        name = class_name.lower().strip()

        # Person — check by name first, then by ID as fallback
        if name == "person" or class_id == PERSON_CLASS_ID:
            return "person"

        if name in ("fire", "flame", "smoke"):
            return "fire"

        if name in self.OBSTACLE_COCO_NAMES:
            return "obstacle"

        return "unknown"

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None

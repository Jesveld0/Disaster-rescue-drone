"""
detector.py — YOLOv8 object detection for the ground station.

Detects persons, fire, and obstacle classes using YOLOv8 nano model
with CUDA acceleration.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from config import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    PERSON_CLASS_ID, FIRE_CLASS_LABEL, OBSTACLE_CLASSES,
)

logger = logging.getLogger(__name__)

# Import ultralytics (lazy to handle missing dependency gracefully)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("ultralytics not installed — YOLO detection disabled")


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


class YOLODetector:
    """
    YOLOv8-based object detector with CUDA support.

    Detects:
    - Persons (COCO class 0)
    - Fire (custom class or placeholder)
    - Obstacles (vehicles, furniture, etc.)

    Usage:
        detector = YOLODetector()
        result = detector.detect(bgr_frame)
    """

    # COCO class names that map to "obstacle" category
    OBSTACLE_COCO_NAMES = {
        "car", "truck", "bus", "motorcycle", "bicycle",
        "bench", "chair", "potted plant", "couch", "bed",
        "dining table", "tv", "refrigerator", "oven",
        "suitcase", "backpack",
    }

    def __init__(self, model_path: str = YOLO_MODEL_PATH, device: str = None, imgsz: int = None):
        """
        Args:
            model_path: Path to YOLOv8 model weights (.pt file).
            device: Compute device ('cuda', 'cpu', or None for auto).
            imgsz: Inference input size. None = auto (320 on CPU, 640 on CUDA).
        """
        self.model_path = model_path
        self.model = None
        self._class_names = {}

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Auto-select inference resolution: 320 on CPU (fast), 640 on CUDA (accurate)
        if imgsz is not None:
            self.imgsz = imgsz
        else:
            self.imgsz = 640 if self.device == "cuda" else 320

        self._load_model()

    def _load_model(self):
        """Load the YOLOv8 model."""
        if not HAS_ULTRALYTICS:
            logger.error("Cannot load YOLO — ultralytics not installed")
            return

        try:
            logger.info(
                "Loading YOLOv8 model: %s on %s (imgsz=%d)",
                self.model_path, self.device, self.imgsz,
            )
            self.model = YOLO(self.model_path)

            # Warm up the model with a dummy inference
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, imgsz=self.imgsz, verbose=False)

            # Cache class names
            self._class_names = self.model.names
            logger.info(
                "YOLO model loaded successfully. Classes: %d, Device: %s",
                len(self._class_names), self.device,
            )

        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
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
            # Run inference
            predictions = self.model.predict(
                bgr_frame,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU_THRESHOLD,
                device=self.device,
                imgsz=self.imgsz,
                verbose=False,
                half=self.device == "cuda",  # FP16 on GPU
            )

            inference_time = (time.monotonic() - start) * 1000
            result.inference_time_ms = round(inference_time, 1)

            if not predictions or len(predictions) == 0:
                return result

            # Process detections
            pred = predictions[0]
            if pred.boxes is None or len(pred.boxes) == 0:
                return result

            boxes = pred.boxes.xyxy.cpu().numpy()
            confidences = pred.boxes.conf.cpu().numpy()
            class_ids = pred.boxes.cls.cpu().numpy().astype(int)

            for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, bbox)
                class_name = self._class_names.get(cls_id, f"class_{cls_id}")

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
            logger.error("YOLO detection error: %s", e)
            result.inference_time_ms = (time.monotonic() - start) * 1000

        return result

    def _categorize(self, class_name: str, class_id: int) -> str:
        """
        Categorize a detected class into person/fire/obstacle.

        Args:
            class_name: YOLO class name string.
            class_id: YOLO class ID integer.

        Returns:
            Category string: 'person', 'fire', 'obstacle', or 'unknown'.
        """
        if class_id == PERSON_CLASS_ID:
            return "person"

        if class_name.lower() in ("fire", "flame", "smoke"):
            return "fire"

        if class_name.lower() in self.OBSTACLE_COCO_NAMES:
            return "obstacle"

        return "unknown"

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None

"""
depth_estimator.py — Monocular depth estimation using MiDaS for obstacle proximity.

Since no physical depth sensor is available, MiDaS provides relative depth
estimates from a single RGB image. Used to detect obstacles that are
dangerously close to the drone (< 3m equivalent).
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch

from config import (
    MIDAS_MODEL_TYPE, OBSTACLE_DEPTH_THRESHOLD,
    OBSTACLE_BBOX_AREA_RATIO, RGB_WIDTH, RGB_HEIGHT,
)

logger = logging.getLogger(__name__)


class DepthEstimator:
    """
    MiDaS-based monocular depth estimator for obstacle proximity detection.

    MiDaS produces relative (inverse) depth maps — higher values indicate
    closer objects. We calibrate a threshold to approximate the 3-meter
    danger zone.

    Usage:
        estimator = DepthEstimator()
        depth_map = estimator.estimate(bgr_frame)
        is_close = estimator.is_obstacle_close(depth_map, bbox)
    """

    def __init__(self, model_type: str = MIDAS_MODEL_TYPE, device: str = None):
        """
        Args:
            model_type: MiDaS model variant ('MiDaS_small', 'DPT_Hybrid', 'DPT_Large').
            device: Compute device ('cuda', 'mps', 'cpu', or None for auto).
        """
        self.model_type = model_type
        self.model = None
        self.transform = None

        # Auto-detect device: CUDA → MPS (Apple Silicon) → CPU
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load MiDaS model from torch hub."""
        try:
            logger.info("Loading MiDaS model: %s on %s", self.model_type, self.device)

            self.model = torch.hub.load(
                "intel-isl/MiDaS", self.model_type, trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()

            # Load appropriate transform
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )

            if self.model_type in ("DPT_Large", "DPT_Hybrid"):
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            logger.info("MiDaS model loaded successfully")

        except Exception as e:
            logger.error("Failed to load MiDaS model: %s", e)
            self.model = None

    def estimate(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate relative depth from a BGR image.

        Args:
            bgr_frame: OpenCV BGR image (H, W, 3).

        Returns:
            Depth map (H, W) float32 with relative depth values.
            Higher values = closer objects. None on failure.
        """
        if self.model is None:
            return None

        try:
            # MiDaS expects RGB input
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # Apply transform
            input_batch = self.transform(rgb_frame).to(self.device)

            # Inference
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(bgr_frame.shape[0], bgr_frame.shape[1]),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Normalize to 0-1 range (higher = closer)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 0:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map.astype(np.float32)

        except Exception as e:
            logger.error("Depth estimation error: %s", e)
            return None

    def is_obstacle_close(
        self,
        depth_map: np.ndarray,
        bbox: tuple[int, int, int, int],
        depth_threshold: float = 0.7,
    ) -> tuple[bool, float]:
        """
        Check if an obstacle within a bounding box is dangerously close.

        Uses two criteria:
        1. Mean relative depth in bbox exceeds threshold
        2. Bounding box area exceeds screen area ratio

        Args:
            depth_map: Normalized depth map (H, W) float32, 0-1.
            bbox: (x1, y1, x2, y2) bounding box.
            depth_threshold: Relative depth threshold (0-1). Default 0.7
                             corresponds roughly to 3m at typical scenes.

        Returns:
            (is_close, mean_depth): Bool indicating danger + mean depth value.
        """
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(depth_map.shape[1], int(x2))
        y2 = min(depth_map.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        region = depth_map[y1:y2, x1:x2]
        mean_depth = float(np.mean(region))

        # Check bbox area ratio (large objects on screen = close)
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = RGB_WIDTH * RGB_HEIGHT
        area_ratio = bbox_area / frame_area

        is_close_by_depth = mean_depth > depth_threshold
        is_close_by_area = area_ratio > OBSTACLE_BBOX_AREA_RATIO

        is_close = is_close_by_depth or is_close_by_area

        if is_close:
            logger.debug(
                "Obstacle CLOSE: depth=%.3f (thresh=%.3f), area_ratio=%.3f (thresh=%.3f)",
                mean_depth, depth_threshold, area_ratio, OBSTACLE_BBOX_AREA_RATIO,
            )

        return is_close, mean_depth

    def get_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to a colorized visualization.

        Args:
            depth_map: Normalized depth map (H, W) float32, 0-1.

        Returns:
            BGR colorized depth image (H, W, 3) uint8.
        """
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

    @property
    def is_loaded(self) -> bool:
        """Check if MiDaS model is loaded and ready."""
        return self.model is not None

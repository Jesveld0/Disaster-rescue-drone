"""
decision.py — Command decision engine for the ground station.

Evaluates fusion results, IR obstacle data, and generates the
appropriate command for the drone.
Priority order: HUMAN_IN_FIRE > FIRE_ALERT > STOP > SLOW > SAFE

Fail-safe: defaults to STOP if no valid inference within 500ms.
"""

import logging
import time
from typing import Optional

from config import (
    CMD_SAFE, CMD_SLOW, CMD_STOP, CMD_FIRE_ALERT, CMD_HUMAN_IN_FIRE,
    COMMAND_NAMES, TIMEOUT_SEC,
    RGB_WIDTH, RGB_HEIGHT,
)
from ground_station.fusion import FusionResult
from ground_station.detector import DetectionResult
from ground_station.depth_estimator import DepthEstimator
from protocol import ObstaclePacket

import numpy as np

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Generates flight commands based on perception results.

    Evaluates multiple threat levels and selects the highest-priority command:
        HUMAN_IN_FIRE (4) > FIRE_ALERT (3) > STOP (2) > SLOW (1) > SAFE (0)

    Now integrates IR proximity sensor data for real hardware obstacle detection.
    IR front obstacle triggers immediate STOP (highest physical danger).

    Includes fail-safe: if no inference result arrives within 500ms,
    automatically issues STOP.
    """

    def __init__(self, depth_estimator: Optional[DepthEstimator] = None):
        """
        Args:
            depth_estimator: Optional MiDaS depth estimator for obstacle proximity.
        """
        self.depth_estimator = depth_estimator
        self._last_inference_time = time.monotonic()
        self._last_command = CMD_SAFE

    def evaluate(
        self,
        fusion_result: FusionResult,
        detections: DetectionResult,
        depth_map: Optional[np.ndarray] = None,
        ir_obstacles: Optional[ObstaclePacket] = None,
    ) -> int:
        """
        Evaluate all perception data and decide on a command.

        Args:
            fusion_result: Thermal-RGB fusion analysis.
            detections: Raw detection results.
            depth_map: Optional MiDaS depth map for obstacle proximity.
            ir_obstacles: Optional IR proximity sensor data from the drone.

        Returns:
            Command code (0-4).
        """
        self._last_inference_time = time.monotonic()
        command = CMD_SAFE

        # Priority 1: Human in fire (highest priority)
        if fusion_result.humans_in_fire > 0:
            command = max(command, CMD_HUMAN_IN_FIRE)
            logger.critical(
                "🔥 HUMAN_IN_FIRE: %d person(s) detected in fire!",
                fusion_result.humans_in_fire,
            )

        # Priority 2: Fire alert
        if fusion_result.any_fire:
            command = max(command, CMD_FIRE_ALERT)

        # Priority 3a: IR proximity STOP (hardware sensors — highest obstacle priority)
        if ir_obstacles is not None:
            ir_stop = self._check_ir_stop(ir_obstacles)
            if ir_stop:
                command = max(command, CMD_STOP)

        # Priority 3b: Visual/depth obstacle STOP
        obstacle_stop = self._check_obstacles(detections, depth_map)
        if obstacle_stop:
            command = max(command, CMD_STOP)

        # Priority 4a: IR proximity SLOW (side/back obstacles)
        if ir_obstacles is not None and command < CMD_STOP:
            ir_slow = self._check_ir_slow(ir_obstacles)
            if ir_slow:
                command = max(command, CMD_SLOW)

        # Priority 4b: Visual obstacle SLOW (close but not critical)
        obstacle_slow = self._check_obstacles_slow(detections, depth_map)
        if obstacle_slow and command < CMD_STOP:
            command = max(command, CMD_SLOW)

        self._last_command = command

        if command >= CMD_STOP:
            logger.warning(
                "Decision: %s (code=%d)",
                COMMAND_NAMES.get(command, "UNKNOWN"), command,
            )

        return command

    def _check_ir_stop(self, ir_obstacles: ObstaclePacket) -> bool:
        """
        Check if IR sensors indicate a critical front obstacle.

        Front obstacle is the most dangerous (drone flying into something).
        """
        if ir_obstacles.front:
            logger.warning(
                "⚠️  IR STOP: Front obstacle detected by proximity sensor!"
            )
            return True
        return False

    def _check_ir_slow(self, ir_obstacles: ObstaclePacket) -> bool:
        """
        Check if IR sensors indicate nearby side/back obstacles.

        Side and back obstacles warrant SLOW but not STOP.
        """
        if ir_obstacles.back or ir_obstacles.left or ir_obstacles.right:
            directions = []
            if ir_obstacles.back:
                directions.append("back")
            if ir_obstacles.left:
                directions.append("left")
            if ir_obstacles.right:
                directions.append("right")
            logger.info(
                "IR SLOW: Obstacle detected at: %s",
                ", ".join(directions),
            )
            return True
        return False

    def check_failsafe(self) -> Optional[int]:
        """
        Check if fail-safe STOP should be triggered due to inference timeout.

        Returns:
            CMD_STOP if timeout exceeded, None otherwise.
        """
        elapsed = time.monotonic() - self._last_inference_time
        if elapsed > TIMEOUT_SEC:
            logger.critical(
                "⚠️  FAIL-SAFE: No inference for %.1f s — issuing STOP!",
                elapsed,
            )
            return CMD_STOP
        return None

    def _check_obstacles(
        self,
        detections: DetectionResult,
        depth_map: Optional[np.ndarray],
    ) -> bool:
        """Check if any obstacle is critically close (STOP)."""
        if not detections.obstacles:
            return False

        for obstacle in detections.obstacles:
            # Method A: Bounding box area threshold
            x1, y1, x2, y2 = obstacle.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = RGB_WIDTH * RGB_HEIGHT
            area_ratio = bbox_area / frame_area

            if area_ratio > 0.20:  # Very large on screen = very close
                logger.warning(
                    "STOP: Obstacle '%s' too close (area_ratio=%.3f)",
                    obstacle.class_name, area_ratio,
                )
                return True

            # Method B: MiDaS depth check
            if depth_map is not None and self.depth_estimator is not None:
                is_close, mean_depth = self.depth_estimator.is_obstacle_close(
                    depth_map, obstacle.bbox
                )
                if is_close:
                    logger.warning(
                        "STOP: Obstacle '%s' too close (depth=%.3f)",
                        obstacle.class_name, mean_depth,
                    )
                    return True

        return False

    def _check_obstacles_slow(
        self,
        detections: DetectionResult,
        depth_map: Optional[np.ndarray],
    ) -> bool:
        """Check if any obstacle warrants SLOW (approaching but not critical)."""
        if not detections.obstacles:
            return False

        for obstacle in detections.obstacles:
            x1, y1, x2, y2 = obstacle.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = RGB_WIDTH * RGB_HEIGHT
            area_ratio = bbox_area / frame_area

            if area_ratio > 0.10:  # Medium-large on screen
                return True

            if depth_map is not None and self.depth_estimator is not None:
                is_close, mean_depth = self.depth_estimator.is_obstacle_close(
                    depth_map, obstacle.bbox, depth_threshold=0.5
                )
                if is_close:
                    return True

        return False

    @property
    def last_command(self) -> int:
        """Return the last generated command."""
        return self._last_command

    @property
    def seconds_since_inference(self) -> float:
        """Seconds elapsed since last successful inference."""
        return time.monotonic() - self._last_inference_time

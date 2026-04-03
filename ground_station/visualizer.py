"""
visualizer.py — Annotated display window and JSON output for the ground station.

Renders:
- Bounding boxes for persons (green), fire zones (red), obstacles (orange)
- Humans in fire highlighted in bright red with warning text
- Thermal overlay toggle
- Fire zone semi-transparent highlighting
- Status bar with FPS, command, frame info
- Per-frame structured JSON output
"""

import json
import logging
import time
from typing import Optional

import cv2
import numpy as np

from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    THERMAL_OVERLAY_ALPHA,
    FIRE_ZONE_COLOR, HUMAN_IN_FIRE_COLOR,
    PERSON_COLOR, OBSTACLE_COLOR,
    COMMAND_NAMES, LOG_JSON_OUTPUT, LOG_FILE_PATH,
)
from ground_station.fusion import FusionResult, PersonAnalysis, FireZone
from ground_station.detector import DetectionResult

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Real-time annotated display with status information and JSON logging.

    Keyboard controls:
        'q' — Quit
        't' — Toggle thermal overlay
        'd' — Toggle depth map overlay
        'f' — Toggle fire mask overlay
    """

    def __init__(self, window_name: str = "Fire Rescue Drone — Ground Station"):
        self.window_name = window_name
        self.show_thermal_overlay = True
        self.show_depth_overlay = False
        self.show_fire_overlay = True
        self._last_thermal = None

        # FPS tracking
        self._frame_times: list[float] = []
        self._fps = 0.0

        # JSON logging
        self._log_file = None
        if LOG_JSON_OUTPUT:
            try:
                self._log_file = open(LOG_FILE_PATH, "a")
            except IOError:
                logger.warning("Could not open log file: %s", LOG_FILE_PATH)

    def render(
        self,
        rgb_frame: np.ndarray,
        fusion_result: FusionResult,
        detections: DetectionResult,
        command_code: int,
        frame_id: int,
        thermal_colormap: Optional[np.ndarray] = None,
        fire_mask: Optional[np.ndarray] = None,
        depth_colormap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a fully annotated display frame.

        Args:
            rgb_frame: Base BGR image (720, 1280, 3).
            fusion_result: Fusion analysis results.
            detections: Raw YOLO detections.
            command_code: Current command being sent to drone.
            frame_id: Current frame ID.
            thermal_colormap: Optional colorized thermal overlay.
            fire_mask: Optional binary fire mask.
            depth_colormap: Optional colorized depth map.

        Returns:
            Annotated BGR frame.
        """
        display = rgb_frame.copy()

        # Save thermal colormap for separate display
        if thermal_colormap is not None:
            self._last_thermal = thermal_colormap.copy()

        # Apply overlays (thermal overlay removed since it's displayed in a separate window)

        if self.show_depth_overlay and depth_colormap is not None:
            display = self._apply_overlay(display, depth_colormap, 0.4)

        if self.show_fire_overlay and fire_mask is not None:
            display = self._apply_fire_overlay(display, fire_mask)

        # Draw fire zones
        for fire_zone in fusion_result.fire_zones:
            self._draw_fire_zone(display, fire_zone)

        # Draw obstacles
        for obstacle in detections.obstacles:
            self._draw_bbox(
                display, obstacle.bbox,
                f"{obstacle.class_name} ({obstacle.confidence:.2f})",
                OBSTACLE_COLOR, thickness=2,
            )

        # Draw persons (green for safe, red for in-fire)
        for person in fusion_result.persons:
            self._draw_person(display, person)

        # Draw status bar
        self._draw_status_bar(display, command_code, frame_id, detections, fusion_result)

        # Update FPS
        self._update_fps()

        # Generate JSON output
        self._output_json(frame_id, fusion_result, detections, command_code)

        return display

    def show(self, frame: np.ndarray) -> int:
        """
        Display frame and handle keyboard input.

        Returns:
            Key code pressed, or -1 if none.
        """
        cv2.imshow(self.window_name, frame)
        
        # Display thermal camera in a separate window
        if self.show_thermal_overlay and self._last_thermal is not None:
            thermal_disp = self._last_thermal
            # Detect all-zeros (sensor failure / no signal)
            if np.max(thermal_disp) == 0:
                h, w = thermal_disp.shape[:2]
                thermal_disp = np.zeros((h, w, 3), dtype=np.uint8)
                thermal_disp[:, :] = (40, 20, 20)  # Dark blue-gray
                cv2.putText(
                    thermal_disp, "No Thermal Signal",
                    (w // 2 - 160, h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 150, 255), 2,
                )
                cv2.putText(
                    thermal_disp, "Check MLX90640 sensor connection",
                    (w // 2 - 220, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 120), 1,
                )
            cv2.imshow("Thermal Camera", thermal_disp)
        elif not self.show_thermal_overlay:
            try:
                cv2.destroyWindow("Thermal Camera")
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            self.show_thermal_overlay = not self.show_thermal_overlay
            logger.info("Thermal window: %s", "ON" if self.show_thermal_overlay else "OFF")
        elif key == ord('d'):
            self.show_depth_overlay = not self.show_depth_overlay
            logger.info("Depth overlay: %s", "ON" if self.show_depth_overlay else "OFF")
        elif key == ord('f'):
            self.show_fire_overlay = not self.show_fire_overlay
            logger.info("Fire overlay: %s", "ON" if self.show_fire_overlay else "OFF")

        return key

    def _draw_person(self, frame: np.ndarray, person: PersonAnalysis):
        """Draw a person bounding box with thermal info."""
        if person.in_fire:
            color = HUMAN_IN_FIRE_COLOR
            label = f"HUMAN IN FIRE! max={person.max_temp:.0f}°C"
            thickness = 3

            # Flashing effect for critical detection
            if int(time.time() * 4) % 2 == 0:
                # Draw filled red rectangle behind text for visibility
                x1, y1, x2, y2 = person.detection.bbox
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 200), -1)
        else:
            color = PERSON_COLOR
            label = f"Person ({person.detection.confidence:.2f}) {person.max_temp:.0f}°C"
            thickness = 2

        self._draw_bbox(frame, person.detection.bbox, label, color, thickness)

    def _draw_fire_zone(self, frame: np.ndarray, fire_zone: FireZone):
        """Draw a fire zone bounding box."""
        x1, y1, x2, y2 = fire_zone.bbox
        status = "CONFIRMED" if fire_zone.thermal_confirmed else "visual"
        color = (0, 0, 255) if fire_zone.thermal_confirmed else (0, 100, 255)

        label = f"FIRE ({status}) max={fire_zone.max_temp:.0f}°C"
        self._draw_bbox(frame, fire_zone.bbox, label, color, 2)

        # Semi-transparent fire overlay
        if fire_zone.thermal_confirmed:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw a labeled bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_status_bar(
        self,
        frame: np.ndarray,
        command_code: int,
        frame_id: int,
        detections: DetectionResult,
        fusion_result: FusionResult,
    ):
        """Draw status bar at the bottom of the frame."""
        bar_height = 40
        h, w = frame.shape[:2]
        bar_y = h - bar_height

        # Background
        cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)

        # Command status with color coding
        cmd_name = COMMAND_NAMES.get(command_code, "UNKNOWN")
        cmd_colors = {
            0: (0, 200, 0),     # SAFE = green
            1: (0, 200, 255),   # SLOW = yellow
            2: (0, 0, 255),     # STOP = red
            3: (0, 100, 255),   # FIRE_ALERT = orange
            4: (0, 0, 255),     # HUMAN_IN_FIRE = red
        }
        cmd_color = cmd_colors.get(command_code, (200, 200, 200))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Command
        cv2.putText(
            frame, f"CMD: {cmd_name}", (10, bar_y + 28),
            font, 0.6, cmd_color, 2, cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            frame, f"FPS: {self._fps:.0f}", (250, bar_y + 28),
            font, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Frame ID
        cv2.putText(
            frame, f"Frame: {frame_id}", (400, bar_y + 28),
            font, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Detection counts
        stats_text = (
            f"Persons: {len(fusion_result.persons)} | "
            f"Fire: {len(fusion_result.fire_zones)} | "
            f"Obstacles: {len(detections.obstacles)} | "
            f"In-Fire: {fusion_result.humans_in_fire}"
        )
        cv2.putText(
            frame, stats_text, (550, bar_y + 28),
            font, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # YOLO inference time
        cv2.putText(
            frame, f"YOLO: {detections.inference_time_ms:.0f}ms",
            (1100, bar_y + 28),
            font, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
        )

    def _apply_overlay(
        self, base: np.ndarray, overlay: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Blend an overlay onto the base image."""
        if overlay.shape[:2] != base.shape[:2]:
            overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))
        if len(overlay.shape) == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    def _apply_fire_overlay(self, frame: np.ndarray, fire_mask: np.ndarray):
        """Apply red tint to fire regions."""
        if fire_mask.shape[:2] != frame.shape[:2]:
            fire_mask = cv2.resize(fire_mask, (frame.shape[1], frame.shape[0]))

        fire_regions = fire_mask > 0
        if not np.any(fire_regions):
            return frame

        overlay = frame.copy()
        overlay[fire_regions] = [0, 0, 200]  # Red tint
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def _update_fps(self):
        """Track and compute rolling FPS."""
        now = time.monotonic()
        self._frame_times.append(now)
        # Keep last 30 frame times
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed

    def _output_json(
        self,
        frame_id: int,
        fusion_result: FusionResult,
        detections: DetectionResult,
        command_code: int,
    ):
        """Generate structured JSON output for the frame."""
        output = {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "humans": [
                {
                    "bbox": list(p.detection.bbox),
                    "in_fire": p.in_fire,
                    "max_temp": p.max_temp,
                    "mean_temp": p.mean_temp,
                    "confidence": p.detection.confidence,
                }
                for p in fusion_result.persons
            ],
            "fire_zones": [
                {
                    "bbox": list(f.bbox),
                    "thermal_confirmed": f.thermal_confirmed,
                    "max_temp": f.max_temp,
                    "confidence": f.confidence,
                }
                for f in fusion_result.fire_zones
            ],
            "obstacles": [
                {
                    "bbox": list(o.bbox),
                    "class": o.class_name,
                    "confidence": o.confidence,
                }
                for o in detections.obstacles
            ],
            "command_sent": command_code,
            "command_name": COMMAND_NAMES.get(command_code, "UNKNOWN"),
            "inference_time_ms": detections.inference_time_ms,
            "fps": round(self._fps, 1),
        }

        # Write to log file
        if self._log_file:
            try:
                self._log_file.write(json.dumps(output) + "\n")
                self._log_file.flush()
            except IOError:
                pass

    def close(self):
        """Clean up display window and log file."""
        cv2.destroyAllWindows()
        if self._log_file:
            self._log_file.close()
        logger.info("Visualizer closed")
"""
visualizer.py — Annotated display window and JSON output for the ground station.

Renders:
- Bounding boxes for persons (green), fire zones (red), obstacles (orange)
- Persistent track IDs above person bounding boxes
- Humans in fire highlighted in bright red with warning text
- IR proximity sensor directional indicators
- Pathfinding grid overlay (toggle with 'p')
- Thermal overlay toggle
- Fire zone semi-transparent highlighting
- Status bar with FPS, command, frame info, tracker stats
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
from ground_station.tracker import TrackedPerson, TrackingResult
from protocol import ObstaclePacket

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Real-time annotated display with status information and JSON logging.

    Keyboard controls:
        'q' — Quit
        't' — Toggle thermal overlay
        'd' — Toggle depth map overlay
        'f' — Toggle fire mask overlay
        'p' — Toggle pathfinding grid overlay
    """

    # Colors for track ID display
    TRACK_COLORS = [
        (255, 50, 50),    # Blue-ish
        (50, 255, 50),    # Green
        (50, 50, 255),    # Red
        (255, 255, 50),   # Cyan
        (255, 50, 255),   # Magenta
        (50, 255, 255),   # Yellow
        (200, 150, 50),   # Steel blue
        (50, 200, 150),   # Teal
    ]

    def __init__(self, window_name: str = "Fire Rescue Drone — Ground Station"):
        self.window_name = window_name
        self.show_thermal_overlay = True
        self.show_depth_overlay = False
        self.show_fire_overlay = True
        self.show_grid_overlay = False
        self._last_thermal = None

        # Drop mechanism state
        self._drop_flash_until: float = 0.0   # monotonic time until flash expires
        self._drop_pending: bool = False       # set True by Space key, consumed by pipeline

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
        tracking_result: Optional[TrackingResult] = None,
        ir_obstacles: Optional[ObstaclePacket] = None,
        grid_overlay: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render a fully annotated display frame.

        Args:
            rgb_frame: Base BGR image (720, 1280, 3).
            fusion_result: Fusion analysis results.
            detections: Raw detection results.
            command_code: Current command being sent to drone.
            frame_id: Current frame ID.
            thermal_colormap: Optional colorized thermal overlay.
            fire_mask: Optional binary fire mask.
            depth_colormap: Optional colorized depth map.
            tracking_result: Optional DeepSORT tracking results.
            ir_obstacles: Optional IR proximity sensor data.
            grid_overlay: Optional pathfinding grid image.

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

        # Draw tracked persons with IDs (if tracking available)
        if tracking_result and tracking_result.tracked_persons:
            self._draw_tracked_persons(display, tracking_result, fusion_result)
        else:
            # Fallback: draw persons without track IDs
            for person in fusion_result.persons:
                self._draw_person(display, person)

        # Draw IR proximity sensor indicators
        if ir_obstacles is not None:
            self._draw_ir_indicators(display, ir_obstacles)

        # Draw pathfinding grid overlay (bottom-right corner)
        if self.show_grid_overlay and grid_overlay is not None:
            self._draw_grid_overlay(display, grid_overlay)

        # Draw status bar
        self._draw_status_bar(
            display, command_code, frame_id, detections,
            fusion_result, tracking_result, ir_obstacles,
        )

        # Drop flash overlay (shown after Space press)
        self._draw_drop_flash(display)

        # Update FPS
        self._update_fps()

        # Generate JSON output
        self._output_json(
            frame_id, fusion_result, detections, command_code,
            tracking_result, ir_obstacles,
        )

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
            cv2.imshow("Thermal Camera", self._last_thermal)
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
        elif key == ord('p'):
            self.show_grid_overlay = not self.show_grid_overlay
            logger.info("Grid overlay: %s", "ON" if self.show_grid_overlay else "OFF")
        elif key == ord(' '):
            self._drop_pending = True
            logger.info("Drop command triggered via UI")

        return key

    def trigger_drop_flash(self, hold_sec: float = 1.5):
        """Called by pipeline after a drop command is sent; shows a HUD flash."""
        import time as _t
        self._drop_flash_until = _t.monotonic() + hold_sec

    def take_drop_pending(self) -> bool:
        """
        Consume and return the drop_pending flag.
        Returns True once (on the frame Space was pressed), then resets to False.
        """
        pending = self._drop_pending
        self._drop_pending = False
        return pending

    def _draw_drop_flash(self, frame: np.ndarray):
        """Draw a prominent DROP RELEASED overlay for SERVO_HOLD_SEC after a drop."""
        import time as _t
        if _t.monotonic() > self._drop_flash_until:
            return
        h, w = frame.shape[:2]
        # Semi-transparent red/orange banner across top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 80, 220), -1)
        alpha = 0.55
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(
            frame, "\U0001FA82 PAYLOAD DROPPED",
            (w // 2 - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA,
        )

    def _draw_tracked_persons(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
        fusion_result: FusionResult,
    ):
        """Draw tracked persons with persistent track IDs."""
        # Build a map of detection bbox → PersonAnalysis for fire info
        person_analysis_map = {}
        for pa in fusion_result.persons:
            person_analysis_map[pa.detection.bbox] = pa

        for tracked in tracking_result.tracked_persons:
            if not tracked.is_confirmed:
                continue

            # Check if this tracked person is in fire
            in_fire = False
            max_temp = 0.0
            if tracked.detection:
                pa = person_analysis_map.get(tracked.detection.bbox)
                if pa:
                    in_fire = pa.in_fire
                    max_temp = pa.max_temp

            # Choose color based on track ID
            track_color = self.TRACK_COLORS[tracked.track_id % len(self.TRACK_COLORS)]

            if in_fire:
                color = HUMAN_IN_FIRE_COLOR
                label = f"ID:{tracked.track_id} FIRE! {max_temp:.0f}°C"
                thickness = 3

                # Flashing effect
                if int(time.time() * 4) % 2 == 0:
                    x1, y1, x2, y2 = tracked.bbox
                    cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 200), -1)
            else:
                color = track_color
                label = f"ID:{tracked.track_id} ({tracked.confidence:.2f})"
                if max_temp > 0:
                    label += f" {max_temp:.0f}°C"
                thickness = 2

            self._draw_bbox(frame, tracked.bbox, label, color, thickness)

            # Draw small track ID circle at top-left of bbox
            x1, y1, _, _ = tracked.bbox
            cv2.circle(frame, (x1 + 10, y1 - 15), 8, track_color, -1)
            cv2.putText(
                frame, str(tracked.track_id),
                (x1 + 5, y1 - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )

    def _draw_person(self, frame: np.ndarray, person: PersonAnalysis):
        """Draw a person bounding box with thermal info (fallback without tracking)."""
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

    def _draw_ir_indicators(self, frame: np.ndarray, ir_obstacles: ObstaclePacket):
        """
        Draw directional IR obstacle indicators on the frame edges.

        Renders 4 triangular indicators (front/back/left/right) that
        are red when an obstacle is detected and green when clear.
        """
        h, w = frame.shape[:2]
        indicator_size = 30

        indicators = [
            # (obstacle_flag, center_x, center_y, direction_label)
            (ir_obstacles.front, w // 2, indicator_size, "F"),
            (ir_obstacles.back, w // 2, h - indicator_size, "B"),
            (ir_obstacles.left, indicator_size, h // 2, "L"),
            (ir_obstacles.right, w - indicator_size, h // 2, "R"),
        ]

        for obstacle, cx, cy, label in indicators:
            color = (0, 0, 255) if obstacle else (0, 180, 0)
            bg_color = (0, 0, 100) if obstacle else (0, 80, 0)

            # Draw filled circle indicator
            cv2.circle(frame, (cx, cy), indicator_size // 2, bg_color, -1)
            cv2.circle(frame, (cx, cy), indicator_size // 2, color, 2)

            # Draw direction letter
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            cv2.putText(
                frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Status text below/beside indicator
            status = "BLOCKED" if obstacle else "CLEAR"
            if label in ("F", "B"):
                cv2.putText(
                    frame, status,
                    (cx - 25, cy + (25 if label == "F" else -15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA,
                )

    def _draw_grid_overlay(self, frame: np.ndarray, grid_img: np.ndarray):
        """Draw the pathfinding grid in the bottom-right corner of the frame."""
        h, w = frame.shape[:2]
        gh, gw = grid_img.shape[:2]

        # Scale grid to fit in corner (max 300x300)
        max_size = 300
        scale = min(max_size / gw, max_size / gh)
        new_w = int(gw * scale)
        new_h = int(gh * scale)
        grid_resized = cv2.resize(grid_img, (new_w, new_h))

        # Position in bottom-right with padding
        padding = 10
        x1 = w - new_w - padding
        y1 = h - new_h - padding - 40  # Above status bar
        x2 = x1 + new_w
        y2 = y1 + new_h

        if x1 >= 0 and y1 >= 0:
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1 - 2, y1 - 20), (x2 + 2, y2 + 2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw grid
            frame[y1:y2, x1:x2] = grid_resized

            # Label
            cv2.putText(
                frame, "Pathfinder Grid",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
            )

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
        tracking_result: Optional[TrackingResult] = None,
        ir_obstacles: Optional[ObstaclePacket] = None,
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
            5: (0, 200, 255),   # DROP = cyan
        }
        cmd_color = cmd_colors.get(command_code, (200, 200, 200))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Command
        cv2.putText(
            frame, f"CMD: {cmd_name}", (10, bar_y + 28),
            font, 0.6, cmd_color, 2, cv2.LINE_AA,
        )

        # Drop shortcut hint
        cv2.putText(
            frame, "[SPACE]=DROP", (10, bar_y - 6),
            font, 0.38, (120, 120, 120), 1, cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            frame, f"FPS: {self._fps:.0f}", (220, bar_y + 28),
            font, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Frame ID
        cv2.putText(
            frame, f"Frame: {frame_id}", (330, bar_y + 28),
            font, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Detection + tracking counts
        track_count = tracking_result.confirmed_tracks if tracking_result else 0
        stats_text = (
            f"Persons: {len(fusion_result.persons)} | "
            f"Tracks: {track_count} | "
            f"Fire: {len(fusion_result.fire_zones)} | "
            f"Obs: {len(detections.obstacles)} | "
            f"InFire: {fusion_result.humans_in_fire}"
        )
        cv2.putText(
            frame, stats_text, (470, bar_y + 28),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # IR status (compact)
        if ir_obstacles:
            ir_text = (
                f"IR: {'F' if ir_obstacles.front else '-'}"
                f"{'B' if ir_obstacles.back else '-'}"
                f"{'L' if ir_obstacles.left else '-'}"
                f"{'R' if ir_obstacles.right else '-'}"
            )
            ir_color = (0, 0, 255) if ir_obstacles.front else (0, 180, 0)
            cv2.putText(
                frame, ir_text, (950, bar_y + 28),
                font, 0.5, ir_color, 1, cv2.LINE_AA,
            )

        # RF-DETR inference time
        cv2.putText(
            frame, f"RF-DETR: {detections.inference_time_ms:.0f}ms",
            (1080, bar_y + 28),
            font, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
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
        tracking_result: Optional[TrackingResult] = None,
        ir_obstacles: Optional[ObstaclePacket] = None,
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
            "tracked_persons": [
                {
                    "track_id": t.track_id,
                    "bbox": list(t.bbox),
                    "confidence": t.confidence,
                    "confirmed": t.is_confirmed,
                    "age": t.age,
                }
                for t in (tracking_result.tracked_persons if tracking_result else [])
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
            "ir_sensors": {
                "front": ir_obstacles.front if ir_obstacles else None,
                "back": ir_obstacles.back if ir_obstacles else None,
                "left": ir_obstacles.left if ir_obstacles else None,
                "right": ir_obstacles.right if ir_obstacles else None,
            } if ir_obstacles else None,
            "command_sent": command_code,
            "command_name": COMMAND_NAMES.get(command_code, "UNKNOWN"),
            "inference_time_ms": detections.inference_time_ms,
            "tracking_time_ms": tracking_result.tracking_time_ms if tracking_result else 0,
            "active_tracks": tracking_result.active_tracks if tracking_result else 0,
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
        self._close_log()
        logger.info("Visualizer closed")

    def _close_log(self):
        """Close the JSON log file handle if open."""
        if self._log_file is not None:
            try:
                self._log_file.close()
            except IOError:
                pass
            finally:
                self._log_file = None

    def __del__(self):
        """Fallback: ensure log file is closed even if close() was never called."""
        self._close_log()
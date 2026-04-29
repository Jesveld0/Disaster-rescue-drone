"""
pipeline.py — Threaded real-time pipeline for the ground station.

Orchestrates all modules into a 3-thread pipeline:
    Thread 1 (Receive):   UDP recv → decode → queue
    Thread 2 (Inference): queue → RF-DETR + tracker + thermal + depth + pathfinding → fusion → decision
    Thread 3 (Display):   render annotated frame → show + JSON output

All AI processing (RF-DETR detection, DeepSORT tracking, MiDaS depth,
thermal fusion, A* pathfinding) runs on the ground station.

Usage:
    python -m ground_station.pipeline
    python -m ground_station.pipeline --port 5000 --no-depth
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# Add parent directory for imports
sys.path.insert(0, ".")

from config import (
    DATA_PORT, COMMAND_PORT, OBSTACLE_PORT, TARGET_FPS,
    CMD_STOP, CMD_SAFE, COMMAND_NAMES,
    RGB_WIDTH, RGB_HEIGHT,
)
from protocol import FramePacket, ObstaclePacket
from ground_station.receiver import FrameReceiver
from ground_station.decoder import FrameDecoder
from ground_station.thermal_processing import ThermalProcessor
from ground_station.detector import RFDETRDetector, DetectionResult
from ground_station.tracker import HumanTracker, TrackingResult
from ground_station.depth_estimator import DepthEstimator
from ground_station.fusion import ThermalFusion, FusionResult
from ground_station.decision import DecisionEngine
from ground_station.command_sender import CommandSender
from ground_station.visualizer import Visualizer
from ground_station.obstacle_receiver import ObstacleReceiver
from ground_station.pathfinder import OccupancyGrid, AStarPathfinder

logger = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """Intermediate data between receive and inference stages."""
    frame_id: int
    timestamp_ms: int
    rgb_bgr: np.ndarray         # BGR image (720, 1280, 3)
    thermal_gray: np.ndarray    # Grayscale thermal (24, 32)


@dataclass
class InferenceResult:
    """Output from the inference stage for the display stage."""
    frame: DecodedFrame
    detections: DetectionResult
    fusion_result: FusionResult
    command_code: int
    thermal_data: dict                                      # From ThermalProcessor.process()
    depth_map: Optional[np.ndarray] = None
    depth_colormap: Optional[np.ndarray] = None
    tracking_result: Optional[TrackingResult] = None        # DeepSORT tracking
    ir_obstacles: Optional[ObstaclePacket] = None           # IR sensor data
    path: list[tuple[int, int]] = field(default_factory=list)  # A* path
    grid_overlay: Optional[np.ndarray] = None               # Pathfinding grid image


class Pipeline:
    """
    Real-time ground station pipeline with 3 processing threads.

    Architecture:
        [UDP Receiver] → decode_queue → [Inference Thread] → display_queue → [Display Thread]
        [Obstacle Receiver] → latest IR data → [Inference Thread]

    All AI inference (RF-DETR, DeepSORT, MiDaS, A*) runs on the ground station.
    The drone only sends raw frames + IR sensor data.
    """

    def __init__(
        self,
        data_port: int = DATA_PORT,
        command_port: int = COMMAND_PORT,
        obstacle_port: int = OBSTACLE_PORT,
        enable_depth: bool = True,
        enable_display: bool = True,
        enable_tracking: bool = True,
        enable_pathfinding: bool = True,
    ):
        """
        Args:
            data_port: UDP port for receiving frame data.
            command_port: UDP port for sending commands.
            obstacle_port: UDP port for receiving IR obstacle data.
            enable_depth: Enable MiDaS depth estimation (requires GPU memory).
            enable_display: Enable OpenCV display window.
            enable_tracking: Enable DeepSORT human tracking.
            enable_pathfinding: Enable A* pathfinding with occupancy grid.
        """
        self.enable_depth = enable_depth
        self.enable_display = enable_display
        self.enable_tracking = enable_tracking
        self.enable_pathfinding = enable_pathfinding

        # Queues between stages (small to keep latency low)
        self._decode_queue: queue.Queue[DecodedFrame] = queue.Queue(maxsize=3)
        self._display_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=2)

        # Modules
        self.receiver = FrameReceiver(
            port=data_port,
            timeout_callback=self._on_timeout,
        )
        self.decoder = FrameDecoder()
        self.thermal_processor = ThermalProcessor()
        self.detector = RFDETRDetector()
        self.tracker = HumanTracker() if enable_tracking else None
        self.depth_estimator = DepthEstimator() if enable_depth else None
        self.fusion = ThermalFusion(self.thermal_processor)
        self.decision = DecisionEngine(self.depth_estimator)
        self.command_sender = CommandSender(command_port=command_port)
        self.visualizer = Visualizer() if enable_display else None

        # IR obstacle receiver
        self.obstacle_receiver = ObstacleReceiver(port=obstacle_port)

        # Pathfinding
        self.occupancy_grid = OccupancyGrid() if enable_pathfinding else None
        self.pathfinder = AStarPathfinder() if enable_pathfinding else None

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self):
        """Initialize all modules and start the pipeline threads."""
        logger.info("=" * 60)
        logger.info("  Fire Rescue Drone — Ground Station Pipeline")
        logger.info("=" * 60)
        logger.info("Data port: %d | Command port: %d | Obstacle port: %d",
                     DATA_PORT, COMMAND_PORT, OBSTACLE_PORT)
        logger.info("Depth estimation: %s", "ENABLED" if self.enable_depth else "DISABLED")
        logger.info("Human tracking: %s", "ENABLED" if self.enable_tracking else "DISABLED")
        logger.info("Pathfinding: %s", "ENABLED" if self.enable_pathfinding else "DISABLED")
        logger.info("Display: %s", "ENABLED" if self.enable_display else "DISABLED")
        logger.info("RF-DETR loaded: %s | Device: %s",
                     self.detector.is_loaded,
                     self.detector.device if self.detector.is_loaded else "N/A")
        if self.tracker:
            logger.info("DeepSORT tracker: %s", "READY" if self.tracker.is_loaded else "FAILED")
        if self.depth_estimator:
            logger.info("MiDaS loaded: %s", self.depth_estimator.is_loaded)
        logger.info("=" * 60)

        self._running = True

        # Start receivers
        self.receiver.start()
        self.obstacle_receiver.start()

        # Start pipeline threads
        self._threads = [
            threading.Thread(
                target=self._receive_stage, daemon=True, name="ReceiveStage"
            ),
            threading.Thread(
                target=self._inference_stage, daemon=True, name="InferenceStage"
            ),
        ]
        for t in self._threads:
            t.start()

        # Display runs on main thread (OpenCV requirement on macOS)
        if self.enable_display:
            self._display_stage()
        else:
            # Headless mode — just process results
            self._headless_stage()

    def _receive_stage(self):
        """
        Stage 1: Receive and decode frames.

        Pulls frames from the UDP receiver, decodes JPEG + thermal,
        and pushes to the inference queue.
        """
        logger.info("Receive stage started")

        while self._running:
            try:
                # Get latest frame (skips stale ones)
                packet = self.receiver.get_latest()
                if packet is None:
                    time.sleep(0.005)  # 5ms polling interval
                    continue

                # Decode
                rgb_bgr, thermal_gray = self.decoder.decode(packet)
                if rgb_bgr is None or thermal_gray is None:
                    continue

                decoded = DecodedFrame(
                    frame_id=packet.frame_id,
                    timestamp_ms=packet.timestamp_ms,
                    rgb_bgr=rgb_bgr,
                    thermal_gray=thermal_gray,
                )

                # Non-blocking put — drop if queue full
                try:
                    self._decode_queue.put_nowait(decoded)
                except queue.Full:
                    # Drop oldest and insert new
                    try:
                        self._decode_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._decode_queue.put_nowait(decoded)

            except Exception as e:
                logger.error("Receive stage error: %s", e, exc_info=True)

        logger.info("Receive stage stopped")

    def _inference_stage(self):
        """
        Stage 2: Run AI inference on decoded frames.

        Performs RF-DETR detection, DeepSORT tracking, thermal processing,
        depth estimation, A* pathfinding, fusion analysis, and decision-making.
        All processing happens on the ground station.
        """
        logger.info("Inference stage started")

        while self._running:
            try:
                # Get next decoded frame (blocking with timeout)
                try:
                    decoded = self._decode_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check fail-safe
                    failsafe = self.decision.check_failsafe()
                    if failsafe is not None:
                        self.command_sender.send_stop()
                    continue

                # Thermal processing
                thermal_data = self.thermal_processor.process(decoded.thermal_gray)

                # RF-DETR detection (replaces YOLOv8)
                detections = self.detector.detect(decoded.rgb_bgr)

                # DeepSORT human tracking
                tracking_result = None
                if self.tracker and self.tracker.is_loaded:
                    tracking_result = self.tracker.update(detections, decoded.rgb_bgr)

                # Depth estimation (optional)
                depth_map = None
                depth_colormap = None
                if self.depth_estimator and self.depth_estimator.is_loaded:
                    depth_map = self.depth_estimator.estimate(decoded.rgb_bgr)
                    if depth_map is not None:
                        depth_colormap = self.depth_estimator.get_depth_colormap(depth_map)

                # Get latest IR obstacle data
                ir_obstacles = self.obstacle_receiver.get_latest()

                # Pathfinding
                path = []
                grid_overlay = None
                if self.occupancy_grid and self.pathfinder:
                    # Update grid from IR sensors
                    if ir_obstacles:
                        self.occupancy_grid.update_from_ir({
                            "front": ir_obstacles.front,
                            "back": ir_obstacles.back,
                            "left": ir_obstacles.left,
                            "right": ir_obstacles.right,
                        })

                    # Update grid from visual detections
                    self.occupancy_grid.update_from_detections(detections.obstacles)

                    # Compute path
                    path_result = self.pathfinder.find_path(self.occupancy_grid)
                    path = path_result.path

                    # Render grid with path
                    grid_img = self.occupancy_grid.render()
                    if path:
                        grid_overlay = self.pathfinder.render_path(grid_img, path)
                    else:
                        grid_overlay = grid_img

                    # Decay obstacles for next frame
                    self.occupancy_grid.decay()

                # Thermal-RGB fusion
                fusion_result = self.fusion.analyze(
                    detections,
                    thermal_data["temperatures"],
                    thermal_data["fire_mask"],
                )

                # Decision: generate command (now includes IR obstacle data)
                command_code = self.decision.evaluate(
                    fusion_result, detections, depth_map,
                    ir_obstacles=ir_obstacles,
                )

                # Send command to drone
                if self.receiver.sender_address:
                    self.command_sender.set_drone_address(self.receiver.sender_address)
                self.command_sender.send(decoded.frame_id, command_code)

                # Package result for display
                result = InferenceResult(
                    frame=decoded,
                    detections=detections,
                    fusion_result=fusion_result,
                    command_code=command_code,
                    thermal_data=thermal_data,
                    depth_map=depth_map,
                    depth_colormap=depth_colormap,
                    tracking_result=tracking_result,
                    ir_obstacles=ir_obstacles,
                    path=path,
                    grid_overlay=grid_overlay,
                )

                # Push to display queue
                try:
                    self._display_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self._display_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._display_queue.put_nowait(result)

            except Exception as e:
                logger.error("Inference stage error: %s", e, exc_info=True)

        logger.info("Inference stage stopped")

    def _display_stage(self):
        """
        Stage 3: Render and display annotated frames.

        MUST run on the main thread (OpenCV GUI requirement on macOS/Linux).
        """
        logger.info("Display stage started (main thread)")

        try:
            while self._running:
                try:
                    result = self._display_queue.get(timeout=0.05)
                except queue.Empty:
                    # Show a "waiting" frame if no data
                    self._show_waiting_frame()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self._running = False
                    continue

                # Render annotated frame
                display = self.visualizer.render(
                    rgb_frame=result.frame.rgb_bgr,
                    fusion_result=result.fusion_result,
                    detections=result.detections,
                    command_code=result.command_code,
                    frame_id=result.frame.frame_id,
                    thermal_colormap=result.thermal_data.get("colormap"),
                    fire_mask=result.thermal_data.get("fire_mask"),
                    depth_colormap=result.depth_colormap,
                    tracking_result=result.tracking_result,
                    ir_obstacles=result.ir_obstacles,
                    grid_overlay=result.grid_overlay,
                )

                # Show and handle keyboard
                key = self.visualizer.show(display)
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    self._running = False

                # Manual drop — Space key consumed from visualizer
                if self.visualizer.take_drop_pending():
                    logger.warning("🪂 Manual DROP triggered by operator (Space key)")
                    self.command_sender.send_drop(frame_id=result.frame_id)
                    self.visualizer.trigger_drop_flash()


        except KeyboardInterrupt:
            logger.info("Display interrupted")
        finally:
            self.stop()

    def _headless_stage(self):
        """Headless mode: consume results and log, no display."""
        logger.info("Running in headless mode (no display)")
        try:
            while self._running:
                try:
                    result = self._display_queue.get(timeout=0.1)
                    cmd_name = COMMAND_NAMES.get(result.command_code, "UNKNOWN")
                    track_count = (
                        result.tracking_result.confirmed_tracks
                        if result.tracking_result else 0
                    )
                    ir_status = "N/A"
                    if result.ir_obstacles:
                        ir_status = (
                            f"F={'Y' if result.ir_obstacles.front else 'N'} "
                            f"B={'Y' if result.ir_obstacles.back else 'N'} "
                            f"L={'Y' if result.ir_obstacles.left else 'N'} "
                            f"R={'Y' if result.ir_obstacles.right else 'N'}"
                        )
                    logger.info(
                        "Frame %d | CMD: %s | Persons: %d | Tracks: %d | "
                        "Fire: %d | Obstacles: %d | IR: %s | "
                        "RF-DETR: %.0fms",
                        result.frame.frame_id,
                        cmd_name,
                        len(result.fusion_result.persons),
                        track_count,
                        len(result.fusion_result.fire_zones),
                        len(result.detections.obstacles),
                        ir_status,
                        result.detections.inference_time_ms,
                    )
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            logger.info("Headless mode interrupted")
        finally:
            self.stop()

    def _show_waiting_frame(self):
        """Display a waiting/standby frame when no data is received."""
        frame = np.zeros((RGB_HEIGHT, RGB_WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for drone connection...",
            (RGB_WIDTH // 2 - 250, RGB_HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2,
        )
        cv2.putText(
            frame,
            f"Listening on port {DATA_PORT}",
            (RGB_WIDTH // 2 - 180, RGB_HEIGHT // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1,
        )
        timeout_status = f"Time since last: {self.decision.seconds_since_inference:.1f}s"
        cv2.putText(
            frame,
            timeout_status,
            (RGB_WIDTH // 2 - 160, RGB_HEIGHT // 2 + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1,
        )
        cv2.imshow("Fire Rescue Drone — Ground Station", frame)

    def _on_timeout(self):
        """Callback when no packet received within timeout period."""
        logger.warning("Receiver timeout — sending STOP command")
        self.command_sender.send_stop()

    def stop(self):
        """Shut down all pipeline components cleanly."""
        logger.info("Shutting down pipeline...")
        self._running = False

        for t in self._threads:
            t.join(timeout=3.0)

        self.receiver.stop()
        self.obstacle_receiver.stop()
        self.command_sender.close()
        if self.visualizer:
            self.visualizer.close()

        logger.info("Pipeline stopped cleanly")


def main():
    """CLI entry point for the ground station pipeline."""
    parser = argparse.ArgumentParser(
        description="Fire Rescue Drone — Ground Station Pipeline"
    )
    parser.add_argument(
        "--port", type=int, default=DATA_PORT,
        help=f"UDP data port (default: {DATA_PORT})",
    )
    parser.add_argument(
        "--command-port", type=int, default=COMMAND_PORT,
        help=f"UDP command port (default: {COMMAND_PORT})",
    )
    parser.add_argument(
        "--obstacle-port", type=int, default=OBSTACLE_PORT,
        help=f"UDP obstacle data port (default: {OBSTACLE_PORT})",
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Disable MiDaS depth estimation (saves GPU memory)",
    )
    parser.add_argument(
        "--no-tracking", action="store_true",
        help="Disable DeepSORT human tracking",
    )
    parser.add_argument(
        "--no-pathfinding", action="store_true",
        help="Disable A* pathfinding and occupancy grid",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without display window (logging only)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Handle Ctrl+C gracefully
    pipeline = Pipeline(
        data_port=args.port,
        command_port=args.command_port,
        obstacle_port=args.obstacle_port,
        enable_depth=not args.no_depth,
        enable_display=not args.headless,
        enable_tracking=not args.no_tracking,
        enable_pathfinding=not args.no_pathfinding,
    )

    def signal_handler(sig, frame):
        logger.info("Signal %d received — shutting down...", sig)
        pipeline.stop()
        sys.exit(0)

    # Register BEFORE start() — start() blocks on the display thread
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.start()


if __name__ == "__main__":
    main()

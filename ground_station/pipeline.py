"""
pipeline.py — Threaded real-time pipeline for the ground station.

Orchestrates all modules into a 3-thread pipeline:
    Thread 1 (Receive):   UDP recv → decode → queue
    Thread 2 (Inference): queue → YOLO + thermal + depth → fusion → decision
    Thread 3 (Display):   render annotated frame → show + JSON output

Display is DECOUPLED from inference: raw video streams at camera FPS
and the last-known bounding boxes / annotations are overlaid. This keeps
the video smooth even when inference is slow (CPU mode).

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
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, ".")

from config import (
    DATA_PORT, COMMAND_PORT, TARGET_FPS,
    CMD_STOP, CMD_SAFE, COMMAND_NAMES,
    RGB_WIDTH, RGB_HEIGHT,
)
from protocol import FramePacket
from ground_station.receiver import FrameReceiver
from ground_station.decoder import FrameDecoder
from ground_station.thermal_processing import ThermalProcessor
from ground_station.detector import YOLODetector, DetectionResult
from ground_station.depth_estimator import DepthEstimator
from ground_station.fusion import ThermalFusion, FusionResult
from ground_station.decision import DecisionEngine
from ground_station.command_sender import CommandSender
from ground_station.visualizer import Visualizer

logger = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """Intermediate data between receive and inference stages."""
    frame_id: int
    timestamp_ms: int
    rgb_bgr: np.ndarray         # BGR image (720, 1280, 3)
    thermal_heatmap: np.ndarray # Heatmap BGR thermal (24, 32, 3)


@dataclass
class InferenceResult:
    """Output from the inference stage for the display stage."""
    frame: DecodedFrame
    detections: DetectionResult
    fusion_result: FusionResult
    command_code: int
    thermal_data: dict           # From ThermalProcessor.process()
    depth_map: Optional[np.ndarray] = None
    depth_colormap: Optional[np.ndarray] = None


class Pipeline:
    """
    Real-time ground station pipeline with 3 processing threads.

    Architecture:
        [UDP Receiver] → decode_queue → [Inference Thread] → (cached result)
                       ↘ raw_display_queue → [Display Thread] ← uses cached inference

    Display is decoupled from inference: the display thread shows raw
    frames at camera FPS with the last-known inference annotations overlaid.
    This keeps video smooth even when YOLO runs at 3-5 FPS on CPU.
    """

    def __init__(
        self,
        data_port: int = DATA_PORT,
        command_port: int = COMMAND_PORT,
        enable_depth: bool = True,
        enable_display: bool = False,
    ):
        """
        Args:
            data_port: UDP port for receiving frame data.
            command_port: UDP port for sending commands.
            enable_depth: Enable MiDaS depth estimation (requires GPU memory).
            enable_display: Enable OpenCV display window.
        """
        self.enable_display = enable_display

        # Auto-disable depth on CPU (too slow: ~500ms per frame)
        has_cuda = torch.cuda.is_available()
        if enable_depth and not has_cuda:
            logger.warning(
                "⚡ No CUDA GPU detected — disabling MiDaS depth estimation "
                "(saves ~500ms per frame). Use --no-depth to suppress this warning."
            )
            enable_depth = False
        self.enable_depth = enable_depth

        # Queues between stages (small to keep latency low)
        self._decode_queue: queue.Queue[DecodedFrame] = queue.Queue(maxsize=3)
        self._raw_display_queue: queue.Queue[DecodedFrame] = queue.Queue(maxsize=2)

        # Cached inference result (shared between inference and display threads)
        self._cached_inference: Optional[InferenceResult] = None
        self._inference_lock = threading.Lock()

        # Modules
        self.receiver = FrameReceiver(
            port=data_port,
            timeout_callback=self._on_timeout,
        )
        self.decoder = FrameDecoder()
        self.thermal_processor = ThermalProcessor()
        self.detector = YOLODetector()
        self.depth_estimator = DepthEstimator() if enable_depth else None
        self.fusion = ThermalFusion(self.thermal_processor)
        self.decision = DecisionEngine(self.depth_estimator)
        self.command_sender = CommandSender(command_port=command_port)
        self.visualizer = Visualizer() if enable_display else None

        # Control
        self._running = False
        self._threads: list[threading.Thread] = []

    def start(self):
        """Initialize all modules and start the pipeline threads."""
        logger.info("=" * 60)
        logger.info("  Fire Rescue Drone — Ground Station Pipeline")
        logger.info("=" * 60)
        logger.info("Data port: %d | Command port: %d", DATA_PORT, COMMAND_PORT)
        logger.info("Depth estimation: %s", "ENABLED" if self.enable_depth else "DISABLED")
        logger.info("Display: %s", "ENABLED" if self.enable_display else "DISABLED")
        logger.info("YOLO loaded: %s | Device: %s | imgsz: %d",
                     self.detector.is_loaded,
                     self.detector.device if self.detector.is_loaded else "N/A",
                     self.detector.imgsz if self.detector.is_loaded else 0)
        if self.depth_estimator:
            logger.info("MiDaS loaded: %s", self.depth_estimator.is_loaded)
        logger.info("=" * 60)

        self._running = True

        # Start receiver (has its own background thread)
        self.receiver.start()

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
        and pushes to BOTH the inference queue and the raw display queue.
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
                rgb_bgr, thermal_heatmap = self.decoder.decode(packet)
                if rgb_bgr is None or thermal_heatmap is None:
                    continue

                decoded = DecodedFrame(
                    frame_id=packet.frame_id,
                    timestamp_ms=packet.timestamp_ms,
                    rgb_bgr=rgb_bgr,
                    thermal_heatmap=thermal_heatmap,
                )

                # Push to inference queue (non-blocking — drop if full)
                try:
                    self._decode_queue.put_nowait(decoded)
                except queue.Full:
                    try:
                        self._decode_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._decode_queue.put_nowait(decoded)

                # Also push to raw display queue for smooth video
                try:
                    self._raw_display_queue.put_nowait(decoded)
                except queue.Full:
                    try:
                        self._raw_display_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._raw_display_queue.put_nowait(decoded)

            except Exception as e:
                logger.error("Receive stage error: %s", e, exc_info=True)

        logger.info("Receive stage stopped")

    def _inference_stage(self):
        """
        Stage 2: Run AI inference on decoded frames.

        Performs YOLO detection, thermal processing, depth estimation,
        fusion analysis, and decision-making. Results are cached for
        the display thread to pick up.
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
                thermal_data = self.thermal_processor.process(decoded.thermal_heatmap)

                # YOLO detection
                detections = self.detector.detect(decoded.rgb_bgr)

                # Depth estimation (optional)
                depth_map = None
                depth_colormap = None
                if self.depth_estimator and self.depth_estimator.is_loaded:
                    depth_map = self.depth_estimator.estimate(decoded.rgb_bgr)
                    if depth_map is not None:
                        depth_colormap = self.depth_estimator.get_depth_colormap(depth_map)

                # Thermal-RGB fusion
                fusion_result = self.fusion.analyze(
                    detections,
                    thermal_data["temperatures"],
                    thermal_data["fire_mask"],
                )

                # Decision: generate command
                command_code = self.decision.evaluate(
                    fusion_result, detections, depth_map
                )

                # Send command to drone
                if self.receiver.sender_address:
                    self.command_sender.set_drone_address(self.receiver.sender_address)
                self.command_sender.send(decoded.frame_id, command_code)

                # Cache result for display thread
                result = InferenceResult(
                    frame=decoded,
                    detections=detections,
                    fusion_result=fusion_result,
                    command_code=command_code,
                    thermal_data=thermal_data,
                    depth_map=depth_map,
                    depth_colormap=depth_colormap,
                )
                with self._inference_lock:
                    self._cached_inference = result

            except Exception as e:
                logger.error("Inference stage error: %s", e, exc_info=True)

        logger.info("Inference stage stopped")

    def _display_stage(self):
        """
        Stage 3: Render and display frames at camera FPS.

        DECOUPLED from inference: always shows the latest raw frame with
        the most recent inference annotations overlaid. This means:
        - Video streams at camera FPS (~20 FPS) regardless of inference speed
        - Bounding boxes update whenever inference completes (~3-10 FPS on CPU)

        MUST run on the main thread (OpenCV GUI requirement on macOS/Linux).
        """
        logger.info("Display stage started (main thread)")

        last_frame_time = 0.0
        last_raw_frame = None

        try:
            while self._running:
                # Get latest raw frame (short timeout for responsive UI)
                try:
                    raw = self._raw_display_queue.get(timeout=0.03)
                    last_frame_time = time.time()
                    last_raw_frame = raw
                except queue.Empty:
                    if time.time() - last_frame_time > 1.0:
                        # No new frame for > 1s — show waiting frame
                        self._show_waiting_frame()
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self._running = False
                        continue
                    elif last_raw_frame is not None:
                        # Re-render the last available raw frame
                        raw = last_raw_frame
                    else:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self._running = False
                        continue

                # Get cached inference results (if any)
                with self._inference_lock:
                    cached = self._cached_inference

                if cached is not None:
                    # Render raw frame with last-known annotations
                    display = self.visualizer.render(
                        rgb_frame=raw.rgb_bgr,
                        fusion_result=cached.fusion_result,
                        detections=cached.detections,
                        command_code=cached.command_code,
                        frame_id=raw.frame_id,
                        thermal_colormap=cached.thermal_data.get("colormap"),
                        fire_mask=cached.thermal_data.get("fire_mask"),
                        depth_colormap=cached.depth_colormap,
                    )
                else:
                    # No inference yet — show raw frame with init message
                    display = raw.rgb_bgr.copy()
                    cv2.putText(
                        display, "Initializing AI models...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 255), 2, cv2.LINE_AA,
                    )

                # Show and handle keyboard
                key = self.visualizer.show(display)
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    self._running = False

        except KeyboardInterrupt:
            logger.info("Display interrupted")
        finally:
            self.stop()

    def _headless_stage(self):
        """Headless mode: consume inference results and log, no display."""
        logger.info("Running in headless mode (no display)")
        try:
            while self._running:
                with self._inference_lock:
                    result = self._cached_inference

                if result is not None:
                    cmd_name = COMMAND_NAMES.get(result.command_code, "UNKNOWN")
                    logger.info(
                        "Frame %d | CMD: %s | Persons: %d | Fire: %d | Obstacles: %d | "
                        "YOLO: %.0fms",
                        result.frame.frame_id,
                        cmd_name,
                        len(result.fusion_result.persons),
                        len(result.fusion_result.fire_zones),
                        len(result.detections.obstacles),
                        result.detections.inference_time_ms,
                    )
                time.sleep(0.1)
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
        "--no-depth", action="store_true",
        help="Disable MiDaS depth estimation (saves GPU memory)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Enable OpenCV display window (default is headless)",
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
        enable_depth=not args.no_depth,
        enable_display=args.show,
    )

    def signal_handler(sig, frame):
        logger.info("Signal %d received — shutting down...", sig)
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.start()


if __name__ == "__main__":
    main()
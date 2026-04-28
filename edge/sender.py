"""
sender.py — UDP frame sender for the drone payload (Raspberry Pi).

Captures RGB and thermal frames, encodes them into the binary protocol,
and transmits via UDP to the ground station. Also reads IR proximity
sensors and sends obstacle data, and listens for incoming command packets.

Usage:
    python -m edge.sender
    python -m edge.sender --ground-ip 192.168.1.100 --device 0
    python -m edge.sender --no-ir   # Disable IR sensors (dev mode)
"""

import argparse
import logging
import socket
import threading
import time
import sys

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, ".")

from config import (
    GROUND_STATION_IP, DATA_PORT, COMMAND_PORT, OBSTACLE_PORT,
    TARGET_FPS, FRAME_INTERVAL, BUFFER_SIZE,
    THERMAL_WIDTH, THERMAL_HEIGHT, COMMAND_NAMES,
    RGB_WIDTH, RGB_HEIGHT,
    CMD_STOP, CMD_HUMAN_IN_FIRE, CMD_FIRE_ALERT, CMD_SLOW,
)
from protocol import (
    FramePacket, encode_frame_packet, current_timestamp_ms,
    fragment_packet, decode_command, encode_obstacle_packet,
)
from edge.rgb_capture import RGBCamera
from edge.thermal_capture import ThermalCamera
from edge.ir_sensor import IRSensorArray

logger = logging.getLogger(__name__)


class DroneSender:
    """
    Main edge device controller.

    Captures from both cameras, encodes frames into protocol packets,
    and sends them via UDP to the ground station at the target FPS.
    Also reads IR proximity sensors and sends obstacle data.
    Simultaneously listens for command responses.
    """

    def __init__(
        self,
        ground_ip: str = GROUND_STATION_IP,
        data_port: int = DATA_PORT,
        command_port: int = COMMAND_PORT,
        obstacle_port: int = OBSTACLE_PORT,
        camera_index: int = 0,
        enable_ir: bool = True,
    ):
        self.ground_ip = ground_ip
        self.data_port = data_port
        self.command_port = command_port
        self.obstacle_port = obstacle_port

        # Cameras
        self.rgb_camera = RGBCamera(device_index=camera_index)
        self.thermal_camera = ThermalCamera()

        # IR Proximity Sensors
        self.ir_sensors = IRSensorArray(simulated=not enable_ir)
        self.enable_ir = enable_ir

        # Networking
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.obstacle_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # State
        self.frame_id = 0
        self.running = False
        self.last_command = -1
        self._cmd_thread = None

    def start(self):
        """Initialize cameras, sensors, and begin the capture-send loop."""
        logger.info("Initializing drone sender...")

        # Open cameras
        if not self.rgb_camera.open():
            logger.error("Cannot start without RGB camera")
            return
        if not self.thermal_camera.open():
            logger.error("Cannot start without thermal camera")
            return

        # Start IR sensors
        self.ir_sensors.start()
        ir_mode = "HARDWARE" if not self.ir_sensors.is_simulated else "SIMULATED"
        logger.info("IR sensors: %s mode", ir_mode)

        # Bind command listener
        self.cmd_socket.bind(("0.0.0.0", self.command_port))
        self.cmd_socket.settimeout(0.1)

        self.running = True

        # Start command listener thread
        self._cmd_thread = threading.Thread(
            target=self._listen_commands, daemon=True
        )
        self._cmd_thread.start()

        logger.info(
            "Drone sender started. Sending to %s:%d at %d FPS",
            self.ground_ip, self.data_port, TARGET_FPS,
        )
        logger.info(
            "Obstacle data → %s:%d",
            self.ground_ip, self.obstacle_port,
        )

        self._capture_loop()

    def _capture_loop(self):
        """Main capture and send loop, rate-limited to TARGET_FPS."""
        try:
            while self.running:
                loop_start = time.monotonic()

                # Capture RGB
                jpeg_bytes, _ = self.rgb_camera.read()
                if jpeg_bytes is None:
                    logger.warning("Skipping frame %d — RGB capture failed", self.frame_id)
                    time.sleep(FRAME_INTERVAL)
                    continue

                # Capture thermal
                _, thermal_gray = self.thermal_camera.read()

                # Build frame packet
                packet = FramePacket(
                    frame_id=self.frame_id,
                    timestamp_ms=current_timestamp_ms(),
                    rgb_width=RGB_WIDTH,
                    rgb_height=RGB_HEIGHT,
                    thermal_width=THERMAL_WIDTH,
                    thermal_height=THERMAL_HEIGHT,
                    rgb_jpeg=jpeg_bytes,
                    thermal_gray=thermal_gray,
                )

                # Encode and send frame
                raw = encode_frame_packet(packet)
                self._send_packet(raw)

                # Read IR sensors and send obstacle data
                self._send_obstacle_data()

                self.frame_id += 1

                # Rate limiting
                elapsed = time.monotonic() - loop_start
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Periodic stats logging
                if self.frame_id % 100 == 0:
                    ir_state = self.ir_sensors.get_state()
                    logger.info(
                        "Sent frame %d | packet size: %d bytes | loop time: %.1f ms | "
                        "IR: F=%s B=%s L=%s R=%s",
                        self.frame_id, len(raw), elapsed * 1000,
                        ir_state.front, ir_state.back,
                        ir_state.left, ir_state.right,
                    )

        except KeyboardInterrupt:
            logger.info("Capture loop interrupted by user")
        finally:
            self.stop()

    def _send_packet(self, raw: bytes):
        """
        Send a raw packet via UDP, fragmenting if necessary.

        UDP practical limit is ~65507 bytes. Typical JPEG frames at 720p
        with quality 80 are 80-120 KB, so fragmentation is needed.
        """
        fragments = fragment_packet(raw)
        for frag in fragments:
            try:
                self.send_socket.sendto(frag, (self.ground_ip, self.data_port))
            except OSError as e:
                logger.error("Send error: %s", e)

    def _send_obstacle_data(self):
        """Read IR sensors and send obstacle packet to ground station."""
        ir_state = self.ir_sensors.get_state()

        obstacle_packet = encode_obstacle_packet(
            frame_id=self.frame_id,
            front=ir_state.front,
            back=ir_state.back,
            left=ir_state.left,
            right=ir_state.right,
        )

        try:
            self.obstacle_socket.sendto(
                obstacle_packet, (self.ground_ip, self.obstacle_port)
            )
        except OSError as e:
            logger.error("Obstacle send error: %s", e)

    def _listen_commands(self):
        """Background thread: listen for command packets from ground station."""
        logger.info("Command listener started on port %d", self.command_port)
        while self.running:
            try:
                data, addr = self.cmd_socket.recvfrom(BUFFER_SIZE)
                cmd = decode_command(data)
                if cmd is not None:
                    self.last_command = cmd.command_code
                    cmd_name = COMMAND_NAMES.get(cmd.command_code, "UNKNOWN")
                    logger.info(
                        "Received command: %s (code=%d) from %s",
                        cmd_name, cmd.command_code, addr,
                    )
                    self._handle_command(cmd.command_code)
            except socket.timeout:
                continue
            except OSError:
                if self.running:
                    logger.error("Command socket error", exc_info=True)
                break

    def _handle_command(self, command_code: int):
        """
        Process received commands.

        In a real deployment, this would interface with the flight controller
        via MAVLink or a GPIO signal. Here we log the command.
        """
        cmd_name = COMMAND_NAMES.get(command_code, "UNKNOWN")
        if command_code == CMD_STOP:
            logger.critical("⚠️  STOP COMMAND RECEIVED — Halting drone movement!")
        elif command_code == CMD_HUMAN_IN_FIRE:
            logger.critical("🔥 HUMAN IN FIRE DETECTED — Emergency protocol!")
        elif command_code == CMD_FIRE_ALERT:
            logger.warning("🔥 Fire alert — Proceed with caution")
        elif command_code == CMD_SLOW:
            logger.info("⚡ Slow down command received")
        else:
            logger.info("✅ Status: %s", cmd_name)

    def stop(self):
        """Shut down all resources cleanly."""
        self.running = False
        self.rgb_camera.close()
        self.thermal_camera.close()
        self.ir_sensors.stop()
        self.send_socket.close()
        self.obstacle_socket.close()
        self.cmd_socket.close()
        logger.info("Drone sender stopped")


def main():
    """CLI entry point for the drone sender."""
    parser = argparse.ArgumentParser(
        description="Drone Payload UDP Sender — captures RGB + thermal and sends to ground station"
    )
    parser.add_argument(
        "--ground-ip", default=GROUND_STATION_IP,
        help=f"Ground station IP address (default: {GROUND_STATION_IP})",
    )
    parser.add_argument(
        "--data-port", type=int, default=DATA_PORT,
        help=f"Data port (default: {DATA_PORT})",
    )
    parser.add_argument(
        "--command-port", type=int, default=COMMAND_PORT,
        help=f"Command port (default: {COMMAND_PORT})",
    )
    parser.add_argument(
        "--obstacle-port", type=int, default=OBSTACLE_PORT,
        help=f"Obstacle data port (default: {OBSTACLE_PORT})",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--no-ir", action="store_true",
        help="Disable IR proximity sensors (simulated mode)",
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

    sender = DroneSender(
        ground_ip=args.ground_ip,
        data_port=args.data_port,
        command_port=args.command_port,
        obstacle_port=args.obstacle_port,
        camera_index=args.device,
        enable_ir=not args.no_ir,
    )
    sender.start()


if __name__ == "__main__":
    main()

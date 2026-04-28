"""
obstacle_receiver.py — UDP receiver for IR proximity sensor obstacle data.

Receives ObstaclePacket messages from the Raspberry Pi's IR sensors
on a dedicated UDP port, providing real-time obstacle proximity
information to the ground station pipeline.
"""

import logging
import socket
import threading
import time
from typing import Optional

from config import OBSTACLE_PORT, BUFFER_SIZE
from protocol import ObstaclePacket, decode_obstacle_packet

logger = logging.getLogger(__name__)


class ObstacleReceiver:
    """
    Non-blocking UDP receiver for IR obstacle data.

    Runs a background thread that listens for ObstaclePacket messages
    from the drone's IR proximity sensors. Provides thread-safe access
    to the latest obstacle state.

    Usage:
        receiver = ObstacleReceiver()
        receiver.start()
        packet = receiver.get_latest()  # ObstaclePacket or None
        receiver.stop()
    """

    def __init__(self, port: int = OBSTACLE_PORT):
        """
        Args:
            port: UDP port to listen for obstacle data (default: 5002).
        """
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._latest_packet: Optional[ObstaclePacket] = None
        self._lock = threading.Lock()
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._last_receive_time: float = 0.0

        # Stats
        self.stats = {
            "received": 0,
            "dropped_corrupt": 0,
        }

    def start(self):
        """Bind socket and start the receiver thread."""
        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(0.1)
        self._running = True
        self._last_receive_time = time.monotonic()

        self._recv_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="ObstacleReceiver"
        )
        self._recv_thread.start()
        logger.info("ObstacleReceiver started on port %d", self.port)

    def _receive_loop(self):
        """Background thread: continuously receive obstacle packets."""
        while self._running:
            try:
                data, addr = self.socket.recvfrom(BUFFER_SIZE)
                packet = decode_obstacle_packet(data)

                if packet is None:
                    self.stats["dropped_corrupt"] += 1
                    continue

                with self._lock:
                    self._latest_packet = packet
                    self._last_receive_time = time.monotonic()

                self.stats["received"] += 1

                if self.stats["received"] % 100 == 0:
                    logger.info(
                        "ObstacleReceiver stats: received=%d, corrupt=%d",
                        self.stats["received"], self.stats["dropped_corrupt"],
                    )

            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Obstacle receiver socket error", exc_info=True)
                break

    def get_latest(self) -> Optional[ObstaclePacket]:
        """
        Get the latest obstacle packet (thread-safe).

        Returns:
            Most recent ObstaclePacket, or None if no data received.
        """
        with self._lock:
            return self._latest_packet

    @property
    def seconds_since_last(self) -> float:
        """Seconds elapsed since last obstacle packet received."""
        return time.monotonic() - self._last_receive_time

    @property
    def has_recent_data(self) -> bool:
        """Check if obstacle data is recent (within 1 second)."""
        return self.seconds_since_last < 1.0

    def stop(self):
        """Stop receiver and clean up."""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        self.socket.close()
        logger.info("ObstacleReceiver stopped. Stats: %s", self.stats)

"""
receiver.py — UDP frame receiver for the ground station.

Receives fragmented frame packets from the drone, reassembles them,
validates integrity, drops out-of-order / corrupted frames, and detects
communication timeouts.
"""

import logging
import socket
import threading
import time
from collections import deque
from typing import Callable, Optional

from config import DATA_PORT, BUFFER_SIZE, TIMEOUT_SEC
from protocol import (
    FramePacket, decode_frame_packet,
    parse_fragment_header, reassemble_fragments,
)

logger = logging.getLogger(__name__)


class FrameReceiver:
    """
    Non-blocking UDP receiver with background thread.

    Features:
    - Fragment reassembly for large packets
    - Out-of-order frame dropping (only newest frames accepted)
    - Corrupted frame rejection
    - Timeout detection with configurable callback

    Usage:
        receiver = FrameReceiver()
        receiver.start()
        packet = receiver.get_latest()  # non-blocking
        receiver.stop()
    """

    def __init__(
        self,
        port: int = DATA_PORT,
        timeout_callback: Optional[Callable] = None,
        max_queue_size: int = 5,
    ):
        """
        Args:
            port: UDP port to listen on.
            timeout_callback: Called when no packet received within TIMEOUT_SEC.
            max_queue_size: Maximum frames to buffer before dropping oldest.
        """
        self.port = port
        self.timeout_callback = timeout_callback
        self.max_queue_size = max_queue_size

        # Networking
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

        # Frame buffer (thread-safe deque)
        self._frame_queue: deque[FramePacket] = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()

        # State tracking
        self._highest_frame_id = -1
        self._last_receive_time = time.monotonic()
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._timeout_thread: Optional[threading.Thread] = None
        self._sender_address: Optional[tuple] = None

        # Fragment reassembly buffer: {frame_fragments_key: {index: payload}}
        self._fragment_buffer: dict[int, dict[int, bytes]] = {}
        self._fragment_totals: dict[int, int] = {}
        self._fragment_counter = 0

        # Stats
        self.stats = {
            "received": 0,
            "dropped_ooo": 0,
            "dropped_corrupt": 0,
            "timeouts": 0,
        }

    def start(self):
        """Bind socket and start receiver + timeout monitor threads."""
        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(0.1)
        self._running = True
        self._last_receive_time = time.monotonic()

        self._recv_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="FrameReceiver"
        )
        self._timeout_thread = threading.Thread(
            target=self._timeout_monitor, daemon=True, name="TimeoutMonitor"
        )

        self._recv_thread.start()
        self._timeout_thread.start()
        logger.info("FrameReceiver started on port %d", self.port)

    def _receive_loop(self):
        """Background thread: continuously receive UDP datagrams and reassemble."""
        while self._running:
            try:
                data, addr = self.socket.recvfrom(BUFFER_SIZE)
                self._sender_address = addr

                # Parse fragment header
                frag_info = parse_fragment_header(data)
                if frag_info is None:
                    self.stats["dropped_corrupt"] += 1
                    continue

                total_fragments, frag_index, payload = frag_info

                if total_fragments == 1:
                    # Single fragment — process directly
                    self._process_packet(payload)
                else:
                    # Multi-fragment — accumulate and reassemble
                    self._handle_fragment(total_fragments, frag_index, payload)

            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    logger.error("Receive socket error", exc_info=True)
                break

    def _handle_fragment(self, total: int, index: int, payload: bytes):
        """Accumulate fragments and reassemble when complete."""
        key = self._fragment_counter
        # Use simple strategy: accumulate until we get all fragments
        # Reset if we get a fragment 0 (start of new packet)
        if index == 0:
            self._fragment_counter += 1
            key = self._fragment_counter
            self._fragment_buffer[key] = {}
            self._fragment_totals[key] = total

        # Find the most recent key that matches this total
        for k in sorted(self._fragment_buffer.keys(), reverse=True):
            if self._fragment_totals.get(k) == total:
                key = k
                break

        if key not in self._fragment_buffer:
            self._fragment_buffer[key] = {}
            self._fragment_totals[key] = total

        self._fragment_buffer[key][index] = payload

        # Check if complete
        if len(self._fragment_buffer[key]) == total:
            full_data = reassemble_fragments(self._fragment_buffer[key], total)
            del self._fragment_buffer[key]
            del self._fragment_totals[key]

            if full_data is not None:
                self._process_packet(full_data)

            # Clean up old incomplete fragments (keep only last 3)
            keys = sorted(self._fragment_buffer.keys())
            for old_key in keys[:-3]:
                self._fragment_buffer.pop(old_key, None)
                self._fragment_totals.pop(old_key, None)

    def _process_packet(self, data: bytes):
        """Decode a complete packet and add to queue if valid and in-order."""
        packet = decode_frame_packet(data)
        if packet is None:
            self.stats["dropped_corrupt"] += 1
            return

        # Drop out-of-order frames
        if packet.frame_id <= self._highest_frame_id:
            self.stats["dropped_ooo"] += 1
            logger.debug(
                "Dropped OOO frame %d (highest: %d)",
                packet.frame_id, self._highest_frame_id,
            )
            return

        # Accept the frame
        self._highest_frame_id = packet.frame_id
        self._last_receive_time = time.monotonic()
        self.stats["received"] += 1

        with self._queue_lock:
            self._frame_queue.append(packet)

        if self.stats["received"] % 100 == 0:
            logger.info(
                "Receiver stats: received=%d, dropped_ooo=%d, dropped_corrupt=%d",
                self.stats["received"],
                self.stats["dropped_ooo"],
                self.stats["dropped_corrupt"],
            )

    def _timeout_monitor(self):
        """Background thread: monitor for communication timeouts."""
        while self._running:
            elapsed = time.monotonic() - self._last_receive_time
            if elapsed > TIMEOUT_SEC:
                self.stats["timeouts"] += 1
                logger.warning(
                    "Communication timeout: no packet in %.1f s (threshold: %.1f s)",
                    elapsed, TIMEOUT_SEC,
                )
                if self.timeout_callback:
                    self.timeout_callback()
                # Reset timer to avoid repeated rapid callbacks
                self._last_receive_time = time.monotonic()
            time.sleep(0.1)

    def get_latest(self) -> Optional[FramePacket]:
        """
        Get the most recent frame, discarding older buffered frames.

        Returns:
            Latest FramePacket, or None if queue is empty.
        """
        with self._queue_lock:
            if not self._frame_queue:
                return None
            # Take the newest, discard the rest
            latest = self._frame_queue[-1]
            self._frame_queue.clear()
            return latest

    def get_next(self) -> Optional[FramePacket]:
        """
        Get the next frame in order (FIFO).

        Returns:
            Next FramePacket, or None if queue is empty.
        """
        with self._queue_lock:
            if self._frame_queue:
                return self._frame_queue.popleft()
            return None

    @property
    def sender_address(self) -> Optional[tuple]:
        """Return the address of the drone (last packet source)."""
        return self._sender_address

    def stop(self):
        """Stop receiver and clean up resources."""
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._timeout_thread:
            self._timeout_thread.join(timeout=2.0)
        self.socket.close()
        logger.info("FrameReceiver stopped. Stats: %s", self.stats)

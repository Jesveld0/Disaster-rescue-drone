"""
test_loopback.py — Integration test with simulated sender/receiver.

Tests the full data path on localhost:
1. Create synthetic frame data
2. Encode & send via UDP
3. Receive & decode on the other end
4. Verify data integrity
"""

import sys
import socket
import threading
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, ".")

from config import THERMAL_WIDTH, THERMAL_HEIGHT, MAGIC_NUMBER
from protocol import (
    FramePacket, encode_frame_packet, decode_frame_packet,
    fragment_packet, parse_fragment_header, reassemble_fragments,
    encode_command, decode_command,
    current_timestamp_ms,
)


class TestLoopback:
    """Integration tests using loopback UDP communication."""

    LOOPBACK_PORT = 15000  # Use high port to avoid conflicts

    def _create_synthetic_frame(self, frame_id: int = 0) -> FramePacket:
        """Create a synthetic frame packet with valid JPEG data."""
        # Generate a simple test pattern
        pattern = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(pattern, (100, 100), (400, 400), (0, 255, 0), -1)
        cv2.putText(pattern, f"Frame {frame_id}", (500, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # JPEG encode
        _, jpeg_buf = cv2.imencode(".jpg", pattern, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg_bytes = jpeg_buf.tobytes()

        # Thermal data
        thermal = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)

        return FramePacket(
            frame_id=frame_id,
            timestamp_ms=current_timestamp_ms(),
            rgb_width=1280,
            rgb_height=720,
            thermal_width=THERMAL_WIDTH,
            thermal_height=THERMAL_HEIGHT,
            rgb_jpeg=jpeg_bytes,
            thermal_gray=thermal,
        )

    def test_loopback_single_frame(self):
        """Send and receive a single frame via UDP loopback."""
        original = self._create_synthetic_frame(42)
        encoded = encode_frame_packet(original)
        fragments = fragment_packet(encoded)

        received_data = {}
        total_fragments = [0]

        # Set up receiver
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.bind(("127.0.0.1", self.LOOPBACK_PORT))
        recv_sock.settimeout(2.0)

        # Set up sender
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        def send_fragments():
            time.sleep(0.1)  # Brief delay
            for frag in fragments:
                send_sock.sendto(frag, ("127.0.0.1", self.LOOPBACK_PORT))
                time.sleep(0.001)

        # Send in background
        sender_thread = threading.Thread(target=send_fragments, daemon=True)
        sender_thread.start()

        # Receive all fragments
        try:
            for _ in range(len(fragments)):
                data, _ = recv_sock.recvfrom(65535)
                result = parse_fragment_header(data)
                assert result is not None
                total, idx, payload = result
                total_fragments[0] = total
                received_data[idx] = payload
        except socket.timeout:
            pytest.fail("Timed out waiting for fragments")
        finally:
            recv_sock.close()
            send_sock.close()

        # Reassemble and decode
        reassembled = reassemble_fragments(received_data, total_fragments[0])
        assert reassembled is not None

        decoded = decode_frame_packet(reassembled)
        assert decoded is not None
        assert decoded.frame_id == 42
        assert decoded.rgb_width == 1280
        assert decoded.rgb_height == 720
        np.testing.assert_array_equal(decoded.thermal_gray, original.thermal_gray)

    def test_loopback_command(self):
        """Send and receive a command via UDP loopback."""
        port = self.LOOPBACK_PORT + 1

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(2.0)

        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send command
        cmd_data = encode_command(frame_id=10, command_code=2)  # STOP
        send_sock.sendto(cmd_data, ("127.0.0.1", port))

        # Receive command
        try:
            data, _ = recv_sock.recvfrom(65535)
            decoded = decode_command(data)
            assert decoded is not None
            assert decoded.frame_id == 10
            assert decoded.command_code == 2
        except socket.timeout:
            pytest.fail("Timed out waiting for command")
        finally:
            recv_sock.close()
            send_sock.close()

    def test_loopback_multiple_frames_ordering(self):
        """Verify that frames arrive in order when sent sequentially."""
        port = self.LOOPBACK_PORT + 2
        n_frames = 5

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(3.0)

        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Create and send small frames (single fragment each)
        original_ids = []
        for i in range(n_frames):
            packet = self._create_synthetic_frame(i)
            encoded = encode_frame_packet(packet)
            fragments = fragment_packet(encoded)

            for frag in fragments:
                send_sock.sendto(frag, ("127.0.0.1", port))
                time.sleep(0.01)
            original_ids.append(i)
            time.sleep(0.05)

        # Receive and verify ordering
        received_ids = []
        try:
            while len(received_ids) < n_frames:
                # Receive all fragments for this frame
                frag_data = {}
                total = 0
                while True:
                    data, _ = recv_sock.recvfrom(65535)
                    result = parse_fragment_header(data)
                    assert result is not None
                    t, idx, payload = result
                    total = t
                    frag_data[idx] = payload
                    if len(frag_data) == total:
                        break

                reassembled = reassemble_fragments(frag_data, total)
                decoded = decode_frame_packet(reassembled)
                if decoded:
                    received_ids.append(decoded.frame_id)

        except socket.timeout:
            pass  # OK if we got some
        finally:
            recv_sock.close()
            send_sock.close()

        # At least some frames should have arrived in order
        assert len(received_ids) > 0
        for i in range(1, len(received_ids)):
            assert received_ids[i] > received_ids[i - 1], \
                f"Frames out of order: {received_ids}"

    def test_jpeg_integrity(self):
        """Verify JPEG data survives the encode/decode cycle."""
        original = self._create_synthetic_frame(0)
        encoded = encode_frame_packet(original)
        decoded = decode_frame_packet(encoded)

        assert decoded is not None

        # Decode JPEG and verify it's a valid image
        jpeg_array = np.frombuffer(decoded.rgb_jpeg, dtype=np.uint8)
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
        assert image is not None
        assert image.shape == (720, 1280, 3)

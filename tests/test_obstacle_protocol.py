"""
test_obstacle_protocol.py — Tests for ObstaclePacket encoding/decoding.
"""

import struct
import sys
sys.path.insert(0, ".")

from protocol import (
    encode_obstacle_packet, decode_obstacle_packet,
    ObstaclePacket, OBSTACLE_FMT,
)
from config import OBSTACLE_MAGIC


class TestObstaclePacketEncoding:
    """Test ObstaclePacket encode/decode round-trip."""

    def test_encode_all_clear(self):
        """All sensors clear — no obstacles."""
        data = encode_obstacle_packet(42, False, False, False, False)
        assert len(data) == struct.calcsize(OBSTACLE_FMT)

        packet = decode_obstacle_packet(data)
        assert packet is not None
        assert packet.frame_id == 42
        assert packet.front is False
        assert packet.back is False
        assert packet.left is False
        assert packet.right is False

    def test_encode_all_blocked(self):
        """All sensors detect obstacles."""
        data = encode_obstacle_packet(99, True, True, True, True)
        packet = decode_obstacle_packet(data)

        assert packet is not None
        assert packet.frame_id == 99
        assert packet.front is True
        assert packet.back is True
        assert packet.left is True
        assert packet.right is True

    def test_encode_front_only(self):
        """Only front sensor detects obstacle."""
        data = encode_obstacle_packet(1, True, False, False, False)
        packet = decode_obstacle_packet(data)

        assert packet is not None
        assert packet.front is True
        assert packet.back is False
        assert packet.left is False
        assert packet.right is False

    def test_encode_mixed(self):
        """Mixed obstacle pattern."""
        data = encode_obstacle_packet(7, False, True, True, False)
        packet = decode_obstacle_packet(data)

        assert packet is not None
        assert packet.front is False
        assert packet.back is True
        assert packet.left is True
        assert packet.right is False

    def test_timestamp_is_set(self):
        """Timestamp should be a positive value."""
        data = encode_obstacle_packet(0, False, False, False, False)
        packet = decode_obstacle_packet(data)

        assert packet is not None
        assert packet.timestamp_ms > 0


class TestObstaclePacketDecoding:
    """Test ObstaclePacket error handling."""

    def test_reject_too_small(self):
        """Packet smaller than expected should be rejected."""
        result = decode_obstacle_packet(b"\x00" * 5)
        assert result is None

    def test_reject_wrong_magic(self):
        """Packet with wrong magic number should be rejected."""
        # Build a valid-sized packet but with wrong magic
        bad_data = struct.pack(OBSTACLE_FMT, 0xDEADBEEF, 0, 0, 0, 0, 0, 0)
        result = decode_obstacle_packet(bad_data)
        assert result is None

    def test_reject_empty(self):
        """Empty data should be rejected."""
        result = decode_obstacle_packet(b"")
        assert result is None

    def test_accept_extra_bytes(self):
        """Packet with extra trailing bytes should still decode."""
        data = encode_obstacle_packet(10, True, False, True, False)
        data_with_extra = data + b"\x00\x00\x00"
        packet = decode_obstacle_packet(data_with_extra)

        assert packet is not None
        assert packet.frame_id == 10
        assert packet.front is True
        assert packet.left is True

    def test_correct_magic(self):
        """Verify the magic number is correctly set."""
        data = encode_obstacle_packet(0, False, False, False, False)
        magic = struct.unpack("!I", data[:4])[0]
        assert magic == OBSTACLE_MAGIC

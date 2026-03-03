"""
test_protocol.py — Unit tests for the binary communication protocol.

Tests:
- Frame packet encode/decode round-trip
- Corrupted packet handling
- Command packet encode/decode
- Fragment/reassembly logic
"""

import sys
import struct
import time

import numpy as np
import pytest

sys.path.insert(0, ".")

from config import MAGIC_NUMBER, THERMAL_WIDTH, THERMAL_HEIGHT
from protocol import (
    FramePacket, CommandPacket,
    encode_frame_packet, decode_frame_packet,
    encode_command, decode_command,
    fragment_packet, reassemble_fragments, parse_fragment_header,
    current_timestamp_ms,
)


class TestFramePacket:
    """Tests for frame packet encoding and decoding."""

    def _make_sample_packet(self) -> FramePacket:
        """Create a sample FramePacket for testing."""
        # Create a minimal valid JPEG (SOI + EOI markers)
        jpeg_data = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00" * 100 + bytes([0xFF, 0xD9])
        thermal = np.random.randint(0, 255, (THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)

        return FramePacket(
            frame_id=42,
            timestamp_ms=current_timestamp_ms(),
            rgb_width=1280,
            rgb_height=720,
            thermal_width=THERMAL_WIDTH,
            thermal_height=THERMAL_HEIGHT,
            rgb_jpeg=jpeg_data,
            thermal_gray=thermal,
        )

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should produce identical data."""
        original = self._make_sample_packet()
        raw = encode_frame_packet(original)
        decoded = decode_frame_packet(raw)

        assert decoded is not None
        assert decoded.frame_id == original.frame_id
        assert decoded.rgb_width == original.rgb_width
        assert decoded.rgb_height == original.rgb_height
        assert decoded.thermal_width == original.thermal_width
        assert decoded.thermal_height == original.thermal_height
        assert decoded.rgb_jpeg == original.rgb_jpeg
        np.testing.assert_array_equal(decoded.thermal_gray, original.thermal_gray)

    def test_invalid_magic_number(self):
        """Packets with wrong magic number should be rejected."""
        original = self._make_sample_packet()
        raw = encode_frame_packet(original)

        # Corrupt the magic number
        corrupted = b"\x00\x00\x00\x00" + raw[4:]
        decoded = decode_frame_packet(corrupted)
        assert decoded is None

    def test_truncated_packet(self):
        """Truncated packets should be rejected."""
        original = self._make_sample_packet()
        raw = encode_frame_packet(original)

        # Truncate to just header
        truncated = raw[:16]
        decoded = decode_frame_packet(truncated)
        assert decoded is None

    def test_empty_packet(self):
        """Empty data should be rejected."""
        decoded = decode_frame_packet(b"")
        assert decoded is None

    def test_invalid_jpeg(self):
        """Packets with invalid JPEG data should be rejected."""
        thermal = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.uint8)

        # Header + metadata with non-JPEG data
        header = struct.pack("!IIQ", MAGIC_NUMBER, 1, current_timestamp_ms())
        metadata = struct.pack("!HHHH", 1280, 720, THERMAL_WIDTH, THERMAL_HEIGHT)
        bad_jpeg = b"\x00\x00\x00\x00\x00"  # Not a JPEG
        thermal_bytes = thermal.tobytes()

        raw = header + metadata + bad_jpeg + thermal_bytes
        decoded = decode_frame_packet(raw)
        assert decoded is None

    def test_large_frame_id(self):
        """Frame IDs should support large values."""
        packet = self._make_sample_packet()
        packet.frame_id = 2**32 - 1  # Max uint32
        raw = encode_frame_packet(packet)
        decoded = decode_frame_packet(raw)
        assert decoded is not None
        assert decoded.frame_id == 2**32 - 1

    def test_timestamp_preserves_value(self):
        """Timestamps should be preserved exactly."""
        packet = self._make_sample_packet()
        packet.timestamp_ms = 1700000000000  # A specific timestamp
        raw = encode_frame_packet(packet)
        decoded = decode_frame_packet(raw)
        assert decoded is not None
        assert decoded.timestamp_ms == 1700000000000


class TestCommandPacket:
    """Tests for command packet encoding and decoding."""

    def test_encode_decode_roundtrip(self):
        """Command encode/decode should preserve all fields."""
        for code in range(5):
            raw = encode_command(frame_id=100, command_code=code)
            decoded = decode_command(raw)
            assert decoded is not None
            assert decoded.frame_id == 100
            assert decoded.command_code == code

    def test_invalid_magic(self):
        """Command with wrong magic should be rejected."""
        raw = encode_command(1, 0)
        corrupted = b"\x00\x00\x00\x00" + raw[4:]
        decoded = decode_command(corrupted)
        assert decoded is None

    def test_truncated_command(self):
        """Truncated command packets should be rejected."""
        raw = encode_command(1, 0)
        truncated = raw[:10]
        decoded = decode_command(truncated)
        assert decoded is None

    def test_empty_command(self):
        """Empty data should be rejected."""
        decoded = decode_command(b"")
        assert decoded is None


class TestFragmentation:
    """Tests for packet fragmentation and reassembly."""

    def test_single_fragment(self):
        """Small data should produce a single fragment."""
        data = b"small data"
        fragments = fragment_packet(data, max_fragment_size=100)
        assert len(fragments) == 1

    def test_multi_fragment_roundtrip(self):
        """Fragmented data should reassemble to original."""
        original = bytes(range(256)) * 100  # 25.6 KB

        fragments = fragment_packet(original, max_fragment_size=1000)
        assert len(fragments) > 1

        # Parse and reassemble
        parsed: dict[int, bytes] = {}
        total = 0
        for frag in fragments:
            result = parse_fragment_header(frag)
            assert result is not None
            t, idx, payload = result
            total = t
            parsed[idx] = payload

        reassembled = reassemble_fragments(parsed, total)
        assert reassembled == original

    def test_incomplete_fragments(self):
        """Incomplete fragment sets should return None."""
        data = b"x" * 10000
        fragments = fragment_packet(data, max_fragment_size=1000)

        parsed: dict[int, bytes] = {}
        for frag in fragments[:len(fragments) // 2]:
            result = parse_fragment_header(frag)
            _, idx, payload = result
            parsed[idx] = payload

        reassembled = reassemble_fragments(parsed, len(fragments))
        assert reassembled is None

    def test_fragment_header_parse_error(self):
        """Too-short data should fail to parse."""
        result = parse_fragment_header(b"\x00\x01")
        assert result is None

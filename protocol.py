"""
protocol.py — Binary packet encoding/decoding for drone ↔ ground station communication.

Frame Packet (Drone → Ground Station):
    HEADER  (16 bytes): magic(4) + frame_id(4) + timestamp_ms(8)
    METADATA (8 bytes): rgb_w(2) + rgb_h(2) + therm_w(2) + therm_h(2)
    DATA:               JPEG RGB frame (variable) + thermal heatmap (768 * 3 bytes)

Command Packet (Ground Station → Drone):
    HEADER  (16 bytes): magic(4) + frame_id(4) + timestamp_ms(8)
    COMMAND  (1 byte):  command code
"""

import struct
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    MAGIC_NUMBER, HEADER_SIZE, METADATA_SIZE,
    THERMAL_PIXELS, THERMAL_WIDTH, THERMAL_HEIGHT,
    MAX_PACKET_SIZE,
)

logger = logging.getLogger(__name__)

# Struct formats (network byte order = big-endian '!')
HEADER_FMT = "!IIQ"           # magic(uint32) + frame_id(uint32) + timestamp(uint64)
METADATA_FMT = "!HHHH"        # rgb_w, rgb_h, therm_w, therm_h (uint16 each)
COMMAND_FMT = "!IIQB"         # header + command_code(uint8)


@dataclass
class FramePacket:
    """Represents a single frame transmitted from the drone."""
    frame_id: int
    timestamp_ms: int
    rgb_width: int
    rgb_height: int
    thermal_width: int
    thermal_height: int
    rgb_jpeg: bytes             # JPEG-compressed RGB frame
    thermal_heatmap: np.ndarray # uint8 array, shape (thermal_height, thermal_width, 3)


@dataclass
class CommandPacket:
    """Represents a command sent from ground station to drone."""
    frame_id: int
    timestamp_ms: int
    command_code: int


def current_timestamp_ms() -> int:
    """Return current time as milliseconds since epoch."""
    return int(time.time() * 1000)


# =============================================================================
# Frame Packet Encoding / Decoding
# =============================================================================

def encode_frame_packet(packet: FramePacket) -> bytes:
    """
    Serialize a FramePacket to bytes for UDP transmission.

    Layout:
        [HEADER 16B][METADATA 8B][JPEG DATA variable][THERMAL 768*3B]
    """
    header = struct.pack(
        HEADER_FMT,
        MAGIC_NUMBER,
        packet.frame_id,
        packet.timestamp_ms,
    )
    metadata = struct.pack(
        METADATA_FMT,
        packet.rgb_width,
        packet.rgb_height,
        packet.thermal_width,
        packet.thermal_height,
    )
    thermal_bytes = packet.thermal_heatmap.astype(np.uint8).tobytes()
    return header + metadata + packet.rgb_jpeg + thermal_bytes


def decode_frame_packet(data: bytes) -> Optional[FramePacket]:
    """
    Deserialize raw bytes into a FramePacket.

    Returns None if the packet is corrupted or malformed.
    """
    min_size = HEADER_SIZE + METADATA_SIZE + THERMAL_PIXELS
    if len(data) < min_size:
        logger.warning("Packet too small: %d bytes (min %d)", len(data), min_size)
        return None

    try:
        # Parse header
        magic, frame_id, timestamp_ms = struct.unpack_from(HEADER_FMT, data, 0)
        if magic != MAGIC_NUMBER:
            logger.warning("Invalid magic number: 0x%08X (expected 0x%08X)", magic, MAGIC_NUMBER)
            return None

        # Parse metadata
        rgb_w, rgb_h, therm_w, therm_h = struct.unpack_from(
            METADATA_FMT, data, HEADER_SIZE
        )

        # Validate thermal dimensions
        expected_thermal_pixels = therm_w * therm_h
        if expected_thermal_pixels == 0:
            logger.warning("Invalid thermal dimensions: %dx%d", therm_w, therm_h)
            return None

        # Extract JPEG and thermal data
        payload_start = HEADER_SIZE + METADATA_SIZE
        expected_thermal_bytes = expected_thermal_pixels * 3
        jpeg_size = len(data) - payload_start - expected_thermal_bytes

        if jpeg_size <= 0:
            logger.warning("No JPEG data in packet (jpeg_size=%d)", jpeg_size)
            return None

        rgb_jpeg = data[payload_start:payload_start + jpeg_size]
        thermal_bytes = data[payload_start + jpeg_size:]

        # Validate JPEG markers (SOI = 0xFFD8)
        if len(rgb_jpeg) < 2 or rgb_jpeg[0] != 0xFF or rgb_jpeg[1] != 0xD8:
            logger.warning("Invalid JPEG data (missing SOI marker)")
            return None

        thermal_heatmap = np.frombuffer(thermal_bytes, dtype=np.uint8).reshape(
            (therm_h, therm_w, 3)
        )

        return FramePacket(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            rgb_width=rgb_w,
            rgb_height=rgb_h,
            thermal_width=therm_w,
            thermal_height=therm_h,
            rgb_jpeg=rgb_jpeg,
            thermal_heatmap=thermal_heatmap,
        )

    except (struct.error, ValueError) as e:
        logger.warning("Failed to decode frame packet: %s", e)
        return None


# =============================================================================
# Command Packet Encoding / Decoding
# =============================================================================

def encode_command(frame_id: int, command_code: int) -> bytes:
    """Encode a command packet for transmission to the drone."""
    return struct.pack(
        COMMAND_FMT,
        MAGIC_NUMBER,
        frame_id,
        current_timestamp_ms(),
        command_code,
    )


def decode_command(data: bytes) -> Optional[CommandPacket]:
    """
    Decode a command packet from raw bytes.

    Returns None if the packet is corrupted.
    """
    expected_size = struct.calcsize(COMMAND_FMT)
    if len(data) < expected_size:
        logger.warning("Command packet too small: %d bytes", len(data))
        return None

    try:
        magic, frame_id, timestamp_ms, command_code = struct.unpack(COMMAND_FMT, data)
        if magic != MAGIC_NUMBER:
            logger.warning("Invalid magic in command packet")
            return None
        return CommandPacket(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            command_code=command_code,
        )
    except struct.error as e:
        logger.warning("Failed to decode command packet: %s", e)
        return None


# =============================================================================
# Large Packet Fragmentation (for packets exceeding UDP limit)
# =============================================================================

def fragment_packet(data: bytes, max_fragment_size: int = MAX_PACKET_SIZE) -> list[bytes]:
    """
    Split a large packet into numbered fragments for UDP transmission.

    Each fragment has a 6-byte fragment header:
        total_fragments(2) + fragment_index(2) + payload_length(2)
    """
    frag_header_size = 6
    max_payload = max_fragment_size - frag_header_size
    total_fragments = (len(data) + max_payload - 1) // max_payload

    fragments = []
    for i in range(total_fragments):
        start = i * max_payload
        end = min(start + max_payload, len(data))
        payload = data[start:end]
        frag_header = struct.pack("!HHH", total_fragments, i, len(payload))
        fragments.append(frag_header + payload)

    return fragments


def reassemble_fragments(fragments: dict[int, bytes], total: int) -> Optional[bytes]:
    """
    Reassemble fragments into the original packet.

    Args:
        fragments: dict mapping fragment_index → payload bytes
        total: expected total number of fragments

    Returns:
        Reassembled bytes, or None if incomplete.
    """
    if len(fragments) != total:
        return None
    return b"".join(fragments[i] for i in range(total))


def parse_fragment_header(data: bytes) -> Optional[tuple[int, int, bytes]]:
    """
    Parse the fragment header from a received UDP datagram.

    Returns:
        (total_fragments, fragment_index, payload) or None on error.
    """
    if len(data) < 6:
        return None
    total, index, length = struct.unpack("!HHH", data[:6])
    payload = data[6:6 + length]
    if len(payload) != length:
        return None
    return total, index, payload
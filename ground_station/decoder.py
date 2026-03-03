"""
decoder.py — Frame decoding for the ground station.

Decodes JPEG RGB frames and unpacks thermal grayscale arrays
from received FramePacket objects.
"""

import logging
import cv2
import numpy as np

from config import RGB_WIDTH, RGB_HEIGHT, THERMAL_WIDTH, THERMAL_HEIGHT
from protocol import FramePacket

logger = logging.getLogger(__name__)


class FrameDecoder:
    """
    Decodes raw frame packet data into OpenCV images and numpy arrays.

    Handles:
    - JPEG decompression to BGR OpenCV image
    - Thermal grayscale array reshaping
    - Resolution validation and correction
    """

    def __init__(self):
        self._decode_count = 0
        self._error_count = 0

    def decode(self, packet: FramePacket) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Decode a FramePacket into usable image arrays.

        Args:
            packet: Received FramePacket with JPEG and thermal data.

        Returns:
            (rgb_bgr, thermal_gray):
                rgb_bgr — BGR image as numpy array (H, W, 3), or None on error
                thermal_gray — Grayscale thermal image (24, 32) uint8, or None on error
        """
        rgb_bgr = self._decode_jpeg(packet.rgb_jpeg)
        thermal_gray = self._decode_thermal(packet.thermal_gray)

        if rgb_bgr is not None and thermal_gray is not None:
            self._decode_count += 1
        else:
            self._error_count += 1

        return rgb_bgr, thermal_gray

    def _decode_jpeg(self, jpeg_bytes: bytes) -> np.ndarray | None:
        """
        Decompress JPEG bytes to a BGR OpenCV image.

        Returns:
            BGR numpy array (H, W, 3) or None on failure.
        """
        try:
            # Convert bytes to numpy array for OpenCV
            jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr_frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

            if bgr_frame is None:
                logger.warning("JPEG decode returned None")
                return None

            # Ensure correct resolution
            h, w = bgr_frame.shape[:2]
            if w != RGB_WIDTH or h != RGB_HEIGHT:
                logger.debug(
                    "Resizing decoded frame from %dx%d to %dx%d",
                    w, h, RGB_WIDTH, RGB_HEIGHT,
                )
                bgr_frame = cv2.resize(bgr_frame, (RGB_WIDTH, RGB_HEIGHT))

            return bgr_frame

        except Exception as e:
            logger.warning("JPEG decode error: %s", e)
            return None

    def _decode_thermal(self, thermal_gray: np.ndarray) -> np.ndarray | None:
        """
        Validate and reshape thermal grayscale data.

        Args:
            thermal_gray: Raw thermal array from packet.

        Returns:
            Reshaped (THERMAL_HEIGHT, THERMAL_WIDTH) uint8 array, or None.
        """
        try:
            if thermal_gray.size != THERMAL_WIDTH * THERMAL_HEIGHT:
                logger.warning(
                    "Thermal size mismatch: got %d, expected %d",
                    thermal_gray.size, THERMAL_WIDTH * THERMAL_HEIGHT,
                )
                return None

            reshaped = thermal_gray.reshape((THERMAL_HEIGHT, THERMAL_WIDTH))
            return reshaped.astype(np.uint8)

        except Exception as e:
            logger.warning("Thermal decode error: %s", e)
            return None

    @property
    def decode_stats(self) -> dict:
        """Return decode success/error counts."""
        return {
            "decoded": self._decode_count,
            "errors": self._error_count,
        }

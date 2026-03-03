"""
rgb_capture.py — USB RGB camera capture module for Raspberry Pi.

Captures 1280x720 frames at 30 FPS from a USB camera and provides
JPEG-compressed output for efficient UDP transmission.
"""

import logging
import cv2
import numpy as np

from config import RGB_WIDTH, RGB_HEIGHT, RGB_FPS, JPEG_QUALITY

logger = logging.getLogger(__name__)


class RGBCamera:
    """
    USB camera handler that captures RGB frames and produces JPEG-compressed output.

    Usage:
        camera = RGBCamera(device_index=0)
        camera.open()
        jpeg_bytes, frame = camera.read()
        camera.close()
    """

    def __init__(self, device_index: int = 0):
        """
        Args:
            device_index: Video device index (0 = /dev/video0).
        """
        self.device_index = device_index
        self.cap = None
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def open(self) -> bool:
        """
        Open the camera device and configure resolution/FPS.

        Returns:
            True if camera opened successfully.
        """
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera at device index %d", self.device_index)
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RGB_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RGB_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, RGB_FPS)

        # Verify actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Camera opened: device=%d, resolution=%dx%d",
            self.device_index, actual_w, actual_h,
        )

        if actual_w != RGB_WIDTH or actual_h != RGB_HEIGHT:
            logger.warning(
                "Camera resolution mismatch: requested %dx%d, got %dx%d. "
                "Frames will be resized.",
                RGB_WIDTH, RGB_HEIGHT, actual_w, actual_h,
            )
        return True

    def read(self) -> tuple[bytes | None, np.ndarray | None]:
        """
        Capture a single frame and compress it to JPEG.

        Returns:
            (jpeg_bytes, bgr_frame) on success, (None, None) on failure.
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not open")
            return None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            return None, None

        # Ensure correct resolution
        h, w = frame.shape[:2]
        if w != RGB_WIDTH or h != RGB_HEIGHT:
            frame = cv2.resize(frame, (RGB_WIDTH, RGB_HEIGHT))

        # JPEG compression
        success, jpeg_buf = cv2.imencode(".jpg", frame, self._encode_params)
        if not success:
            logger.warning("JPEG encoding failed")
            return None, None

        return jpeg_buf.tobytes(), frame

    def close(self):
        """Release the camera device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

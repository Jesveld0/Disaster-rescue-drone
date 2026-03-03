"""
calibrate_homography.py — Homography calibration tool for thermal→RGB alignment.

Interactive tool to compute the 3x3 homography matrix that maps
thermal image coordinates to RGB image coordinates.

Supports two modes:
1. INTERACTIVE: Click corresponding points in both images
2. AUTOMATIC: Use a checkerboard pattern visible in both cameras

Usage:
    python -m calibration.calibrate_homography --rgb sample_rgb.jpg --thermal sample_thermal.jpg
    python -m calibration.calibrate_homography --auto --device 0
"""

import argparse
import json
import logging
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from config import RGB_WIDTH, RGB_HEIGHT, THERMAL_WIDTH, THERMAL_HEIGHT

logger = logging.getLogger(__name__)


class HomographyCalibrator:
    """
    Interactive homography calibration between thermal and RGB cameras.

    Click 4+ corresponding points in both images to compute the
    perspective transformation matrix.
    """

    def __init__(self):
        self.rgb_points: list[tuple[int, int]] = []
        self.thermal_points: list[tuple[int, int]] = []
        self.homography_matrix: np.ndarray | None = None

    def calibrate_interactive(
        self, rgb_image: np.ndarray, thermal_image: np.ndarray
    ) -> np.ndarray | None:
        """
        Run interactive point-picking calibration.

        User clicks corresponding points in both images (minimum 4 pairs).
        Press 'c' to compute homography, 'r' to reset, 'q' to quit.

        Args:
            rgb_image: BGR image from RGB camera.
            thermal_image: Grayscale or BGR thermal image (will be upscaled).

        Returns:
            3x3 homography matrix, or None if cancelled.
        """
        # Upscale thermal for easier point picking
        thermal_display = cv2.resize(
            thermal_image, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_NEAREST
        )
        if len(thermal_display.shape) == 2:
            thermal_display = cv2.applyColorMap(thermal_display, cv2.COLORMAP_JET)

        self.rgb_points = []
        self.thermal_points = []

        # Set up windows
        cv2.namedWindow("RGB — Click points", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Thermal — Click points", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("RGB — Click points", self._rgb_click)
        cv2.setMouseCallback("Thermal — Click points", self._thermal_click)

        print("\n=== Homography Calibration ===")
        print("1. Click corresponding points in BOTH images")
        print("2. Click points in the SAME order (RGB first, then thermal)")
        print("3. Minimum 4 point pairs required")
        print("   Keys: [c] compute | [r] reset | [q] quit")
        print()

        while True:
            # Draw points on copies
            rgb_disp = rgb_image.copy()
            thermal_disp = thermal_display.copy()

            for i, pt in enumerate(self.rgb_points):
                cv2.circle(rgb_disp, pt, 8, (0, 255, 0), -1)
                cv2.putText(rgb_disp, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for i, pt in enumerate(self.thermal_points):
                cv2.circle(thermal_disp, pt, 8, (0, 255, 0), -1)
                cv2.putText(thermal_disp, str(i + 1), (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Status text
            status = f"RGB points: {len(self.rgb_points)} | Thermal points: {len(self.thermal_points)}"
            cv2.putText(rgb_disp, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("RGB — Click points", rgb_disp)
            cv2.imshow("Thermal — Click points", thermal_disp)

            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                print("Calibration cancelled.")
                cv2.destroyAllWindows()
                return None

            elif key == ord('r'):
                self.rgb_points = []
                self.thermal_points = []
                print("Points reset.")

            elif key == ord('c'):
                if len(self.rgb_points) < 4 or len(self.thermal_points) < 4:
                    print(f"Need at least 4 point pairs. Have: RGB={len(self.rgb_points)}, "
                          f"Thermal={len(self.thermal_points)}")
                    continue

                n_pairs = min(len(self.rgb_points), len(self.thermal_points))
                src = np.array(self.thermal_points[:n_pairs], dtype=np.float32)
                dst = np.array(self.rgb_points[:n_pairs], dtype=np.float32)

                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                if H is not None:
                    self.homography_matrix = H
                    print(f"\nHomography computed from {n_pairs} points:")
                    print(H)

                    # Show warped result
                    warped = cv2.warpPerspective(
                        thermal_display, H, (RGB_WIDTH, RGB_HEIGHT)
                    )
                    blended = cv2.addWeighted(rgb_image, 0.6, warped, 0.4, 0)
                    cv2.imshow("Alignment Preview", blended)
                    print("\nPress 's' to save, 'r' to reset, 'q' to quit")
                else:
                    print("Failed to compute homography. Try more/different points.")

            elif key == ord('s') and self.homography_matrix is not None:
                self._save_homography(self.homography_matrix)
                cv2.destroyAllWindows()
                return self.homography_matrix

        cv2.destroyAllWindows()
        return None

    def calibrate_default(self) -> np.ndarray:
        """
        Compute a default homography assuming cameras are co-located.

        Uses corner-to-corner mapping from thermal resolution to RGB resolution.
        This is a reasonable starting point for rigidly mounted cameras.

        Returns:
            3x3 homography matrix.
        """
        # Map thermal corners to RGB corners (simple scaling)
        src_pts = np.array([
            [0, 0],
            [THERMAL_WIDTH - 1, 0],
            [THERMAL_WIDTH - 1, THERMAL_HEIGHT - 1],
            [0, THERMAL_HEIGHT - 1],
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [RGB_WIDTH - 1, 0],
            [RGB_WIDTH - 1, RGB_HEIGHT - 1],
            [0, RGB_HEIGHT - 1],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts)
        logger.info("Default homography (scaling) computed")
        return H

    def _rgb_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rgb_points.append((x, y))
            print(f"  RGB point {len(self.rgb_points)}: ({x}, {y})")

    def _thermal_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.thermal_points.append((x, y))
            print(f"  Thermal point {len(self.thermal_points)}: ({x}, {y})")

    def _save_homography(self, matrix: np.ndarray, filepath: str = "homography.json"):
        """Save homography matrix to a JSON file."""
        data = {
            "homography": matrix.tolist(),
            "rgb_resolution": [RGB_WIDTH, RGB_HEIGHT],
            "thermal_resolution": [THERMAL_WIDTH, THERMAL_HEIGHT],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nHomography saved to {filepath}")
        print("Copy matrix to config.py HOMOGRAPHY_MATRIX to use in production.")


def main():
    parser = argparse.ArgumentParser(description="Homography Calibration Tool")
    parser.add_argument("--rgb", help="Path to RGB calibration image")
    parser.add_argument("--thermal", help="Path to thermal calibration image")
    parser.add_argument("--default", action="store_true",
                       help="Generate default scaling homography")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    calibrator = HomographyCalibrator()

    if args.default:
        H = calibrator.calibrate_default()
        print("Default homography matrix:")
        print(H)
        calibrator._save_homography(H)
    elif args.rgb and args.thermal:
        rgb = cv2.imread(args.rgb)
        thermal = cv2.imread(args.thermal, cv2.IMREAD_GRAYSCALE)
        if rgb is None or thermal is None:
            print("Error: could not read input images")
            sys.exit(1)
        calibrator.calibrate_interactive(rgb, thermal)
    else:
        print("Usage:")
        print("  Interactive: --rgb <image> --thermal <image>")
        print("  Default:     --default")


if __name__ == "__main__":
    main()

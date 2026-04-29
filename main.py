import time

import cv2
import torch
from ultralytics import RTDETR


def main() -> None:
    # Force Apple Silicon GPU backend (Metal Performance Shaders).
    # If MPS isn't available, we still keep device='mps' as required and fail loudly.
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. Ensure you're on Apple Silicon and have a PyTorch build with MPS support."
        )

    # RT-DETR pretrained weights from Ultralytics.
    model = RTDETR("rtdetr-l.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open default webcam (index 0).")

    # Try to keep latency low for real-time preview.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "RT-DETR Human Detection (MPS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_t = time.perf_counter()
    fps_ema = None
    fps_alpha = 0.15  # smoothing factor

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Run inference on MPS; restrict to COCO class 0 (person).
            results = model.predict(
                source=frame,
                device="mps",
                classes=[0],
                conf=0.25,
                verbose=False,
            )

            # Ultralytics returns a list (len==1 here since we pass one frame).
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()

                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    color = (0, 220, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"person {c:.2f}"
                    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    y_text = max(y1, th + 10)

                    # Label background for readability.
                    cv2.rectangle(
                        frame,
                        (x1, y_text - th - 8),
                        (x1 + tw + 8, y_text + baseline - 4),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 4, y_text - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # FPS (EMA-smoothed).
            now = time.perf_counter()
            dt = max(now - prev_t, 1e-6)
            inst_fps = 1.0 / dt
            prev_t = now
            fps_ema = inst_fps if fps_ema is None else (fps_alpha * inst_fps + (1.0 - fps_alpha) * fps_ema)

            fps_text = f"FPS: {fps_ema:.1f}"
            cv2.rectangle(frame, (10, 10), (10 + 140, 10 + 34), (0, 0, 0), -1)
            cv2.putText(
                frame,
                fps_text,
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

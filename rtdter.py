import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import RTDETR, YOLO
from logger_utils import log_to_csv


@dataclass
class Detection:
    box_xyxy: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    cls: int


class HumanSegmenter:
    """
    Real-time human detection + segmentation pipeline.

    Modes:
    1) RT-DETR detection only (rtdetr-l.pt / rtdetr-x.pt): draws person boxes.
    2) RT-DETR + external seg model (recommended in Ultralytics): use --seg-model yolo11n-seg.pt
       and fuse person masks with RT-DETR person detections.
    3) If your loaded RT-DETR variant directly outputs masks (e.g., custom rtdetr-ins weights),
       this class will use those masks automatically when available.
    """

    PERSON_CLASS_ID = 0  # COCO 'person'

    def __init__(
        self,
        det_model: str = "rtdetr-l.pt",
        seg_model: Optional[str] = None,
        conf: float = 0.35,
        iou: float = 0.5,
        device: Optional[str] = None,
        alpha: float = 0.45,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.device = device
        self.alpha = alpha

        # Detection backbone/head
        self.detector = RTDETR(det_model)

        # Optional segmentation model (e.g. yolo11n-seg.pt / yolo11s-seg.pt)
        self.segmenter = YOLO(seg_model) if seg_model else None

        self.color_person = (60, 180, 75)  # BGR
        self.text_color = (255, 255, 255)

    @staticmethod
    def _to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    @staticmethod
    def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        # a, b shape: [4] => x1,y1,x2,y2
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter = inter_w * inter_h
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _extract_person_detections(self, result) -> List[Detection]:
        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = self._to_numpy(result.boxes.xyxy)
        confs = self._to_numpy(result.boxes.conf)
        classes = self._to_numpy(result.boxes.cls).astype(int)

        for box, conf, cls in zip(boxes, confs, classes):
            if cls == self.PERSON_CLASS_ID:
                detections.append(Detection(box_xyxy=box, conf=float(conf), cls=int(cls)))
        return detections

    def _extract_person_masks(self, frame: np.ndarray, det_result) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (mask_uint8, box_xyxy) for person class.
        mask shape matches frame HxW, values in {0,255}.
        """
        h, w = frame.shape[:2]
        masks_and_boxes: List[Tuple[np.ndarray, np.ndarray]] = []

        # Case A: detector itself returns masks (e.g., rtdetr-ins custom weights)
        if getattr(det_result, "masks", None) is not None and det_result.masks is not None:
            boxes = self._to_numpy(det_result.boxes.xyxy) if det_result.boxes is not None else []
            classes = self._to_numpy(det_result.boxes.cls).astype(int) if det_result.boxes is not None else []
            masks = self._to_numpy(det_result.masks.data)  # [N, Hm, Wm] maybe resized
            for i in range(len(masks)):
                if i < len(classes) and classes[i] == self.PERSON_CLASS_ID:
                    m = (masks[i] > 0.5).astype(np.uint8) * 255
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    box = boxes[i] if i < len(boxes) else np.array([0, 0, 0, 0], dtype=np.float32)
                    masks_and_boxes.append((m, box))
            return masks_and_boxes

        # Case B: external segmentation model
        if self.segmenter is None:
            return masks_and_boxes

        seg_res = self.segmenter.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )[0]

        if seg_res.masks is None or seg_res.boxes is None or len(seg_res.boxes) == 0:
            return masks_and_boxes

        seg_boxes = self._to_numpy(seg_res.boxes.xyxy)
        seg_cls = self._to_numpy(seg_res.boxes.cls).astype(int)
        seg_masks = self._to_numpy(seg_res.masks.data)  # [N, Hm, Wm]

        for i in range(len(seg_masks)):
            if seg_cls[i] != self.PERSON_CLASS_ID:
                continue
            m = (seg_masks[i] > 0.5).astype(np.uint8) * 255
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks_and_boxes.append((m, seg_boxes[i]))

        return masks_and_boxes

    def _overlay_mask(self, frame: np.ndarray, mask_uint8: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
        overlay = frame.copy()
        color_layer = np.zeros_like(frame, dtype=np.uint8)
        color_layer[:, :] = color_bgr
        mask_bool = mask_uint8.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 1.0 - self.alpha, color_layer[mask_bool], self.alpha, 0
        )
        return overlay

    def process_source(
        self,
        source: Union[int, str] = 0,
        save_path: Optional[str] = None,
        view: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_fps = fps_in if fps_in and fps_in > 0 else 30.0
            writer = cv2.VideoWriter(save_path, fourcc, writer_fps, (width, height))

        prev_t = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1) RT-DETR person detection
            det_res = self.detector.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )[0]
            person_dets = self._extract_person_detections(det_res)

            # 2) Segmentation masks (detector masks if available OR external seg model)
            person_masks = self._extract_person_masks(frame, det_res)

            # 3) Fuse masks to detections by best IoU
            fused_masks = [None] * len(person_dets)
            if person_masks and person_dets:
                seg_boxes = [sb for _, sb in person_masks]
                for di, det in enumerate(person_dets):
                    best_j, best_iou = -1, 0.0
                    for sj, sbox in enumerate(seg_boxes):
                        iou = self._box_iou_xyxy(det.box_xyxy, sbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_j = sj
                    if best_j >= 0 and best_iou > 0.1:
                        fused_masks[di] = person_masks[best_j][0]

            # 4) Draw masks first
            vis = frame.copy()
            for m in fused_masks:
                if m is not None:
                    vis = self._overlay_mask(vis, m, self.color_person)

            # 5) Draw boxes + confidence for person class only
            for det in person_dets:
                x1, y1, x2, y2 = det.box_xyxy.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.color_person, 2)
                label = f"person {det.conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), self.color_person, -1)
                cv2.putText(
                    vis,
                    label,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    self.text_color,
                    2,
                    cv2.LINE_AA,
                )

            # 6) FPS counter
            cur_t = time.time()
            inf_time = (cur_t - prev_t) * 1000.0 # Approximation of total processing time
            fps = 1.0 / max(cur_t - prev_t, 1e-6)
            prev_t = cur_t
            
            # Log metrics
            log_data = {
                "epoch": -1,
                "model_name": "RT-DETR",
                "mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0, "f1_score": 0, "loss": 0,
                "inference_time_ms": inf_time,
                "fps": fps
            }
            log_to_csv("rtdetr_results.csv", log_data)

            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(vis)

            if view:
                cv2.imshow("RT-DETR Human Detection + Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC or q
                    break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time human detection + segmentation using RT-DETR (Ultralytics)")
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index (e.g., 0)")
    parser.add_argument("--det-model", type=str, default="rtdetr-l.pt", help="Detection weights: rtdetr-l.pt or rtdetr-x.pt")
    parser.add_argument(
        "--seg-model",
        type=str,
        default=None,
        help="Optional segmentation weights. Example: yolo11n-seg.pt. "
             "If omitted, pipeline uses detector masks only when available (e.g. rtdetr-ins custom weights).",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / None")
    parser.add_argument("--save-path", type=str, default=None, help="Optional output video path")
    parser.add_argument("--no-view", action="store_true", help="Disable live visualization window")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    app = HumanSegmenter(
        det_model=args.det_model,
        seg_model=args.seg_model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    app.process_source(source=source, save_path=args.save_path, view=not args.no_view)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from ultralytics import YOLO


@dataclass
class TrainingConfig:
    data_yaml: Path
    train_images_dir: Path
    val_images_dir: Path
    project_dir: Path = Path("runs/detect")
    run_name: str = "yolov8m_human_detector"
    epochs: int = 100
    patience: int = 15
    imgsz: int = 640
    workers: int = 8
    batch: int = -1  # Auto-batch uses available VRAM for high throughput.
    save_period: int = 5
    overlap_mask: bool = True


class YoloDetectorTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = self._detect_device()
        self.model = self._initialize_model()

    def _initialize_model(self) -> YOLO:
        """
        Initialize YOLO model and recover automatically from corrupted local weights.
        """
        weights_name = "yolov8m.pt"
        try:
            return YOLO(weights_name)
        except RuntimeError as exc:
            err = str(exc).lower()
            is_corrupt_archive = (
                "pytorchstreamreader" in err
                or "failed finding central directory" in err
                or "not a zip archive" in err
            )
            if not is_corrupt_archive:
                raise

            local_weights = Path(weights_name)
            if local_weights.exists():
                local_weights.unlink()

            # Retry once; ultralytics will fetch a fresh copy.
            return YOLO(weights_name)

    def _detect_device(self) -> str:
        """Prefer CUDA automatically and fall back when unavailable."""
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def validate_paths(self) -> None:
        """Validate required dataset files/directories before training starts."""
        required_paths = [
            self.config.data_yaml,
            self.config.train_images_dir,
            self.config.val_images_dir,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Required dataset paths are missing:\n" + "\n".join(f"- {path}" for path in missing)
            )

    def train(self):
        """Train YOLOv8 medium model with high-throughput settings."""
        self.validate_paths()
        # Increase imgsz to 1280 for higher precision if VRAM/headroom allows.
        return self.model.train(
            data=str(self.config.data_yaml),
            epochs=self.config.epochs,
            patience=self.config.patience,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            workers=self.config.workers,
            device=self.device,
            project=str(self.config.project_dir),
            name=self.config.run_name,
            overlap_mask=self.config.overlap_mask,
            save_period=self.config.save_period,
            pretrained=True,
        )

    def export_best_to_onnx(self) -> Path:
        """Export best checkpoint to ONNX for deployment portability."""
        best_weights = self.config.project_dir / self.config.run_name / "weights" / "best.pt"
        if not best_weights.exists():
            raise FileNotFoundError(f"best.pt not found at expected path: {best_weights}")

        best_model = YOLO(str(best_weights))
        onnx_path = Path(
            best_model.export(
                format="onnx",
                dynamic=True,
                simplify=True,
            )
        )
        return onnx_path

    def run_pipeline(self) -> Path:
        """Run full training + export pipeline."""
        self.train()
        return self.export_best_to_onnx()

    def run_camera(self, camera_index: int = 0, conf: float = 0.25) -> None:
        """
        Run real-time webcam detection and print person-only filtered detections.
        Press 'q' in the video window to stop.
        """
        stream = self.model.predict(
            source=camera_index,
            stream=True,
            show=True,
            conf=conf,
            imgsz=self.config.imgsz,
            device=self.device,
            verbose=False,
        )
        for results in stream:
            persons = filter_person_detections([results])
            if persons:
                print(f"Persons detected: {persons}")


def filter_person_detections(results: Any) -> List[Dict[str, Any]]:
    """
    Keep only class-0 (person) detections with confidence and normalized boxes.
    Returns one dict per detection:
      {
        "confidence": float,
        "bbox_xywhn": [x_center, y_center, width, height]
      }
    """
    persons: List[Dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        cls_tensor = boxes.cls
        conf_tensor = boxes.conf
        xywhn_tensor = boxes.xywhn

        for cls_id, conf, xywhn in zip(cls_tensor.tolist(), conf_tensor.tolist(), xywhn_tensor.tolist()):
            if int(cls_id) != 0:
                continue
            persons.append(
                {
                    "confidence": float(conf),
                    "bbox_xywhn": [float(v) for v in xywhn],
                }
            )
    return persons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 trainer and webcam detector")
    parser.add_argument("--mode", choices=["train", "camera"], default="camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    cfg = TrainingConfig(
        data_yaml=Path("data.yaml"),
        train_images_dir=Path("images/train"),
        val_images_dir=Path("images/val"),
    )

    trainer = YoloDetectorTrainer(cfg)
    if args.mode == "camera":
        trainer.run_camera(camera_index=args.camera_index, conf=args.conf)
    else:
        exported_onnx = trainer.run_pipeline()
        print(f"Training complete. ONNX exported to: {exported_onnx}")

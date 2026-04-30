import csv
import os
from pathlib import Path

def log_to_csv(filename, metrics):
    """
    Logs metrics to a CSV file.
    metrics: dict containing keys: epoch, model_name, mAP50, mAP50_95, precision, recall, f1_score, loss, inference_time_ms, fps
    """
    fieldnames = [
        "epoch", "model_name", "mAP50", "mAP50_95", 
        "precision", "recall", "f1_score", "loss", 
        "inference_time_ms", "fps"
    ]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

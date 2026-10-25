'''infer_prawn.py

A script to run inference using the prawn disease detection model trained with Ultralytics YOLO.

Features
--------
- Accepts a single image, a directory of images, or a glob pattern.
- Saves annotated images with bounding boxes and disease labels.
- Optionally outputs predictions as a JSON file for downstream processing.
- Works on CPU (default) but will automatically use GPU if available.

Usage
-----
```bash
python infer_prawn.py --source path/to/image_or_folder --output results/ --conf 0.5
```
```bash
# Example with a glob pattern (all jpg files in a folder)
python infer_prawn.py --source "path/to/images/*.jpg" --output results/
```

The script checks these trained models, in order:
```
prawn/models/prawn_disease_final/weights/best.pt
prawn/models/prawn_fast_test/weights/best.pt
```
If you use a different checkpoint, update `MODEL_CANDIDATES` below.
''' 

import argparse
import os
import json
from pathlib import Path

from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "prawn" / "models" / "prawn_disease_final" / "weights" / "best.pt",
    BASE_DIR / "prawn" / "models" / "prawn_fast_test" / "weights" / "best.pt",
]

def get_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Trained prawn model checkpoint not found in prawn/models.")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - source: str – image path, directory, or glob pattern
        - output: str – directory where results will be saved
        - conf: float – confidence threshold for detections (default 0.5)
        - save_json: bool – whether to write predictions to a JSON file
    """
    parser = argparse.ArgumentParser(
        description="Run prawn disease detection on images using a trained YOLO model."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image, a directory of images, or a glob pattern (e.g., 'imgs/*.jpg').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Directory where annotated images and optional JSON will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, predictions are saved to a JSON file in the output folder.",
    )
    return parser.parse_args()

def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def load_model() -> YOLO:
    """Load the YOLO model.

    Returns
    -------
    YOLO
        The loaded model ready for inference.
    """
    model_path = get_model_path()
    return YOLO(str(model_path))

def run_inference(model: YOLO, source: str, conf: float, save_dir: Path, save_json: bool) -> None:
    """Run inference on the given source and save results.

    Parameters
    ----------
    model : YOLO
        Loaded YOLO model.
    source : str
        Image path, folder, or glob pattern.
    conf : float
        Confidence threshold.
    save_dir : Path
        Directory where annotated images will be stored.
    save_json : bool
        Whether to write predictions to a JSON file.
    """
    results = model.predict(
        source=source,
        conf=conf,
        save=True,               # Save annotated images.
        project=str(save_dir),   # Output folder for annotated images.
        name="",                # No sub‑folder inside project.
        exist_ok=True,
        imgsz=224,               # Same size used during training.
    )

    if save_json:
        json_data = []
        for r in results:
            image_path = r.path
            boxes = []
            for box in r.boxes:
                boxes.append({
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0],
                })
            json_data.append({"image": image_path, "detections": boxes})
        json_path = save_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    print("🔧 Loading prawn model …")
    model = load_model()
    print(f"✅ Model loaded from {get_model_path()}")

    print(f"🚀 Running inference on '{args.source}' with confidence ≥ {args.conf}")
    run_inference(model, args.source, args.conf, output_dir, args.save_json)
    print("✅ Inference completed. Annotated images are in:")
    print(f"   {output_dir.resolve()}")

if __name__ == "__main__":
    main()

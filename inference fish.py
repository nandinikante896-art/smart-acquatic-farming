'''inference_fish.py

A script to run inference using the fish disease detection model trained with Ultralytics YOLO.

Features
--------
- Accepts a single image, a directory of images, or a glob pattern.
- Saves annotated images with bounding boxes and disease labels.
- Optional JSON output for downstream processing.
- Works on CPU (default) but will automatically use GPU if available.
'''

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

# -----------------------------------------------------------------
# Configuration – adjust only if you move/rename the model folder
# -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    Path(r"C:/Users/NANDINI/Desktop/AquaCare_Complete/fish/models/fish_disease_final_fast/weights/best.pt"),
    Path(r"C:/Users/NANDINI/Desktop/AquaCare_Complete/fish/models/fish_disease_final_fast"),
    BASE_DIR / "models" / "fish_disease_final_fast" / "weights" / "best.pt",
    BASE_DIR / "models" / "fish_disease_final_fast",
    BASE_DIR / "models" / "fish_disease_final" / "weights" / "best.pt",
    BASE_DIR / "models" / "fish_disease_final",
    BASE_DIR / "models" / "fish_disease" / "weights" / "best.pt",
]

def get_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Trained fish model folder or checkpoint not found.")

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Run fish disease detection on images using a trained YOLO model."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file, folder, or glob pattern (e.g. 'imgs/*.jpg').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Folder where annotated images / optional JSON will be saved.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default 0.5).",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="If set, predictions are saved to a JSON file.",
    )
    return parser.parse_args()

def ensure_dir(p: Path) -> None:
    """Create a directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

def load_model() -> YOLO:
    """Load the YOLO model."""
    model_path = get_model_path()
    return YOLO(str(model_path))

def run_inference(
    model: YOLO,
    source: str,
    conf: float,
    out_dir: Path,
    save_json: bool,
) -> None:
    """Run inference and optionally write predictions to JSON."""
    results = model.predict(
        source=source,
        conf=conf,
        save=True,              # write annotated images
        project=str(out_dir),   # output folder for images
        name="",                # no extra sub‑folder
        exist_ok=True,
        imgsz=224,              # same size as used during training
    )

    if save_json:
        data = []
        for r in results:
            detections = []
            for b in r.boxes:
                detections.append({
                    "class_id": int(b.cls),
                    "class_name": model.names[int(b.cls)],
                    "confidence": float(b.conf),
                    "bbox": b.xyxy.tolist()[0],   # [x1, y1, x2, y2]
                })
            data.append({"image": r.path, "detections": detections})
        json_path = out_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    print("🔧 Loading fish model …")
    model = load_model()
    print(f"✅ Model loaded from {get_model_path()}")

    print(f"🚀 Running inference on '{args.source}' (conf ≥ {args.conf})")
    run_inference(model, args.source, args.conf, out_dir, args.save_json)

    print("✅ Inference finished. Results are in:")
    print(f"   {out_dir.resolve()}")

if __name__ == "__main__":
    main()

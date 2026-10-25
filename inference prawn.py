'''
infer_prawn.py

Run inference with the prawn disease detection model trained with Ultralytics YOLO.

Features
--------
- Handles a single image, a folder of images, or a glob pattern.
- Saves annotated images with bounding boxes and disease labels.
- Optional JSON output for downstream processing.
- Works on CPU by default; will use GPU automatically if available.
'''

import argparse
import json
from pathlib import Path
from ultralytics import YOLO

# -------------------------------------------------
# Configuration – adjust if you move/rename the model
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    Path(r"C:/Users/NANDINI/Desktop/AquaCare_Complete/prawn/models/prawn_disease_final/weights/best.pt"),
    Path(r"C:/Users/NANDINI/Desktop/AquaCare_Complete/prawn/models/prawn_disease_final"),
    BASE_DIR / "models" / "prawn_disease_final" / "weights" / "best.pt",
    BASE_DIR / "models" / "prawn_disease_final",
    BASE_DIR / "models" / "prawn_disease_final_fast",
]

def get_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Trained prawn model checkpoint not found in prawn/models.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prawn disease detection on images using a trained YOLO model."
    )
    parser.add_argument("--source", type=str, required=True,
        help="Image file, folder, or glob pattern (e.g., 'imgs/*.jpg').")
    parser.add_argument("--output", type=str, default="inference_results",
        help="Folder where annotated images / JSON will be saved.")
    parser.add_argument("--conf", type=float, default=0.5,
        help="Confidence threshold for detections.")
    parser.add_argument("--save_json", action="store_true",
        help="If set, predictions are saved to a JSON file.")
    return parser.parse_args()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_model() -> YOLO:
    model_path = get_model_path()
    return YOLO(str(model_path))

def run_inference(model: YOLO, source: str, conf: float,
                  out_dir: Path, save_json: bool) -> None:
    results = model.predict(
        source=source,
        conf=conf,
        save=True,               # Save annotated images
        project=str(out_dir),    # Output folder
        name="",                 # No sub‑folder inside project
        exist_ok=True,
        imgsz=224,               # Same size used during training
    )
    if save_json:
        data = []
        for r in results:
            boxes = []
            for box in r.boxes:
                boxes.append({
                    "class_id": int(box.cls),
                    "class_name": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0],
                })
            data.append({"image": r.path, "detections": boxes})
        json_path = out_dir / "predictions.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Predictions saved to {json_path}")

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    ensure_dir(out_dir)

    print("🔧 Loading prawn model …")
    model = load_model()
    print(f"✅ Model loaded from {MODEL_PATH}")

    print(f"🚀 Running inference on '{args.source}' (conf ≥ {args.conf})")
    run_inference(model, args.source, args.conf, out_dir, args.save_json)

    print("✅ Inference finished. Results are in:")
    print(f"   {out_dir.resolve()}")

if __name__ == "__main__":
    main()

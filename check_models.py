"""
AquaCare - Model Path Update Script
=====================================
After retraining, run this script to verify new models exist
and show paths to update in app_final.py
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

models_to_check = {
    "Fish - final fast": os.path.join(BASE, "fish", "models", "fish_disease_final_fast", "weights", "best.pt"),
    "Fish - disease": os.path.join(BASE, "fish", "models", "fish_disease", "weights", "best.pt"),
    "Fish - retrained": os.path.join(BASE, "fish", "models", "fish_retrain_v2", "weights", "best.pt"),
    "Prawn - final": os.path.join(BASE, "prawn", "models", "prawn_disease_final", "weights", "best.pt"),
    "Prawn - fast test": os.path.join(BASE, "prawn", "models", "prawn_fast_test", "weights", "best.pt"),
}

print("=" * 70)
print("AquaCare Model Status Check")
print("=" * 70)
for name, path in models_to_check.items():
    exists = os.path.exists(path)
    size_mb = os.path.getsize(path) / (1024*1024) if exists else 0
    status = f"EXISTS ({size_mb:.1f} MB)" if exists else "MISSING"
    print(f"  {name:<30} {status}")
    if exists:
        print(f"    Path: {path}")
print()

# Check what app_final.py currently loads
print("=" * 70)
print("Searching app_final.py for model discovery paths...")
app_path = os.path.join(BASE, "app_final.py")
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    if "fish_disease" in line.lower() or "prawn_disease" in line.lower() or "discover_species" in line.lower():
        print(f"  Line {i:4d}: {line.rstrip()}")

print()
print("=" * 70)
print("Instructions:")
print("  Training notebooks: fish/AquaCare_Fish_DO_Training.ipynb")
print("                     prawn/AquaCare_Prawn_DO_Training.ipynb")
print("  Disease-detection notebooks: fish/fish yolo.ipynb, prawn/prawn yolo.ipynb")
print("  Start the application with: python app.py")
print("=" * 70)

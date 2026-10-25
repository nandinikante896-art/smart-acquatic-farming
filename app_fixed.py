"""
Run from the AquaCare_Complete folder:
    python apply_fixes.py app_final.py
Creates app_final.py.bak first, then applies each fix.
Prints OK / SKIP for every patch so you can see what changed.
"""
import os
import re
import shutil
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "app_final.py"
if not os.path.exists(target):
    sys.exit(f"File not found: {target}")

shutil.copyfile(target, target + ".bak")
src = open(target, "r", encoding="utf-8").read()


def rep(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"SKIP  {label}  (text not found)")
        return
    if count == 1 and n != 1:
        print(f"SKIP  {label}  (found {n} times, expected 1)")
        return
    src = src.replace(old, new)
    print(f"OK    {label}")


def rep_func(label, start_marker, end_marker, new_code):
    """Replace everything from start_marker up to (not including) end_marker."""
    global src
    pat = re.compile(re.escape(start_marker) + r".*?(?=" + re.escape(end_marker) + r")", re.S)
    if len(pat.findall(src)) != 1:
        print(f"SKIP  {label}  (function boundaries not found uniquely)")
        return
    src = pat.sub(lambda m: new_code, src, count=1)
    print(f"OK    {label}")


# 1. static folder -------------------------------------------------------------
rep("static folder", "app = Flask(__name__)\n",
    'app = Flask(__name__, static_folder="static_ui", static_url_path="/static")\n')
rep("remove custom static route", '@app.route("/static/<path:filepath>")\n', "")

# 2. primary model selection (cached, prefers retrain) -------------------------
NEW_SELECT = '''_PRIMARY_CACHE = {}


def _select_primary_disease_model(models, class_map, species):
    key = (species, tuple(id(m) for m in (models or [])))
    if key in _PRIMARY_CACHE:
        return _PRIMARY_CACHE[key]

    profiles = [_model_class_profile(m, class_map) for m in (models or []) if m is not None]
    profiles = [p for p in profiles if p["recognized_count"] > 0]

    if not profiles:
        result = (None, [], "No checkpoint exposes recognized disease classes.")
    else:
        profiles.sort(
            key=lambda p: (
                p["recognized_count"],
                "retrain" in p.get("path", "").lower(),
                "best.pt" in p.get("path", "").lower(),
                "fast" not in p.get("path", "").lower(),
                "test" not in p.get("path", "").lower(),
                -len(p.get("path", "")),
            ),
            reverse=True,
        )
        result = (profiles[0]["model"], profiles, "")
        print(f"[MODEL PROFILE] {species} PRIMARY = {profiles[0].get('path')}")

    _PRIMARY_CACHE[key] = result
    return result


'''
rep_func("select primary model", "def _select_primary_disease_model(",
         "def _model_supports_species_classes(", NEW_SELECT)

# 3. stricter false-positive filter --------------------------------------------
NEW_FILTER = '''def _filter_disease_false_positives(image_path, detections, species, allow_closeup=False):
    out = []
    for det in detections or []:
        cls = str(det.get("class_name", "")).strip()
        conf = float(det.get("confidence", 0.0))
        if cls in {"Healthy Fish", "Healthy"}:
            if conf >= HEALTHY_MIN_CONF:
                out.append(det)
            continue
        min_conf = 0.55 if cls in {"Black_Gill", "WSSV_Black_Gill"} else 0.50
        if conf < min_conf:
            continue
        if _box_visual_consistency(image_path, det, species, allow_closeup=False):
            out.append(det)
    return out


'''
rep_func("false-positive filter", "def _filter_disease_false_positives(",
         "def _dedupe_detection_boxes(", NEW_FILTER)

# 4. stability check + healthy-vs-disease margin -------------------------------
rep("stability check",
    "                if max_conf >= 0.25 or (len(views_seen) >= MIN_DISEASE_VIEW_SUPPORT and max_conf >= 0.18):\n"
    "                    stable.extend(group)\n\n"
    "            disease_candidates = stable if stable else disease_candidates\n",
    "                if max_conf >= 0.45 or (len(views_seen) >= MIN_DISEASE_VIEW_SUPPORT and max_conf >= 0.35):\n"
    "                    stable.extend(group)\n\n"
    "            disease_candidates = stable\n")

rep("healthy beats weak disease",
    "        if top_disease is None and top_healthy is not None:\n",
    "        if top_disease is not None and top_healthy is not None:\n"
    "            if float(top_healthy.get(\"confidence\", 0)) >= float(top_disease.get(\"confidence\", 0)) + 0.20:\n"
    "                top_disease = None\n\n"
    "        if top_disease is None and top_healthy is not None:\n")

# 5. augment off ---------------------------------------------------------------
if "augment=True" in src:
    src = src.replace("augment=True", "augment=False")
    print("OK    augment off")
else:
    print("SKIP  augment off")

# 6. disable colour-heuristic fallback -----------------------------------------
rep("disable colour fallback",
    "        # Work on HSV/Lab for robust colour/texture measurements.\n",
    "        return None  # colour heuristic disabled: it produced wrong diseases\n\n"
    "        # Work on HSV/Lab for robust colour/texture measurements.\n")

# 7. no more permanent close-up mode -------------------------------------------
rep("allow_closeup false",
    "allow_closeup=(not bool(species_boxes)),", "allow_closeup=False,")
rep("disease_min_conf",
    'disease_min_conf = 0.20 if (yolo_res.get("crop_mode") or not species_boxes) else 0.30',
    "disease_min_conf = 0.45")
rep("closeup conf",
    "best_disease_conf >= 0.22", "best_disease_conf >= 0.45")

# 8. species classifier gate ---------------------------------------------------
GATE_FUNC = '''SPECIES_CLS_PATH = os.path.join(BASE_DIR, "models", "species_cls.pt")
SPECIES_CLS = None
if os.path.exists(SPECIES_CLS_PATH):
    try:
        SPECIES_CLS = YOLO(SPECIES_CLS_PATH)
        print(f"[LOAD] Species classifier: {SPECIES_CLS_PATH}")
    except Exception as exc:
        print(f"[WARN] Species classifier load failed: {exc}")


def species_classifier_gate(image_path: str, selected: str):
    """Returns (ok, predicted_species, confidence). ok=True when no classifier is present."""
    if SPECIES_CLS is None:
        return True, None, 0.0
    try:
        r = SPECIES_CLS.predict(source=image_path, imgsz=224, verbose=False)[0]
        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf.item())
        pred = str(r.names[top1]).strip().lower()
        print(f"[SPECIES CLS] pred={pred} conf={conf:.3f} selected={selected}")
        if pred in {"fish", "prawn"} and pred != selected and conf >= 0.70:
            return False, pred, conf
        return True, pred, conf
    except Exception as exc:
        print(f"[SPECIES CLS] error {exc}")
        return True, None, 0.0


'''
rep("classifier function",
    "def _internal_opposite_species_check(",
    GATE_FUNC + "def _internal_opposite_species_check(")

GATE_CALL = '''    _ok, _pred, _pc = species_classifier_gate(image_path, species)
    if not _ok:
        _w = "Prawn" if _pred == "prawn" else "Fish"
        _s = "Fish" if species == "fish" else "Prawn"
        return _make_species_rejection(
            species,
            f"❌ This looks like a {_w} image ({_pc*100:.1f}%), but you selected {_s}. Please switch the species.",
            f"❌ ఇది {('రొయ్య' if _pred == 'prawn' else 'చేప')} image లా ఉంది ({_pc*100:.1f}%). Species మార్చండి."
        )

'''
rep("classifier call",
    '    species_model = FISH_OBJECT_MODEL if species == "fish" else PRAWN_OBJECT_MODEL\n',
    GATE_CALL + '    species_model = FISH_OBJECT_MODEL if species == "fish" else PRAWN_OBJECT_MODEL\n')

# 9. vision width/height NameError ---------------------------------------------
rep("vision width/height",
    '    bbox = vision.get("bbox")\n    annotated_b64 = _annotate_vision_bbox',
    '    bbox = vision.get("bbox")\n'
    '    try:\n'
    '        with Image.open(image_path) as _im:\n'
    '            width, height = _im.size\n'
    '    except Exception:\n'
    '        width, height = 1, 1\n'
    '    annotated_b64 = _annotate_vision_bbox')
src = src.replace("            # pyrefly: ignore [unknown-name]\n", "")

open(target, "w", encoding="utf-8").write(src)
print(f"\nDone. Backup saved as {target}.bak")

# 10. frontend loadSample fix ---------------------------------------------------
html_path = os.path.join("static_ui", "index.html")
if os.path.exists(html_path):
    shutil.copyfile(html_path, html_path + ".bak")
    h = open(html_path, "r", encoding="utf-8").read()
    a = "if (!silent) loader.style.display = 'block';"
    if a in h:
        h = h.replace(a, "loader.style.display = 'block';")
        print("OK    index.html: silent crash")
    else:
        print("SKIP  index.html: silent crash (already fixed or not found)")
    b = ("setTimeout(() => {\n"
         "          document.getElementById('disease-form').dispatchEvent(new Event('submit'));\n"
         "        }, 120);")
    if b in h:
        h = h.replace(b, "setTimeout(() => submitDiseaseDetection(null, false), 120);")
        print("OK    index.html: sample submit")
    else:
        print("SKIP  index.html: sample submit (text differs, change it by hand)")
    open(html_path, "w", encoding="utf-8").write(h)
else:
    print("SKIP  static_ui/index.html not found")
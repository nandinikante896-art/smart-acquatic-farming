import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "app.py"
DST = "app_fixed.py"
code = open(SRC, "r", encoding="utf-8").read().replace("\r\n", "\n")

GATE_BLOCK = '''# ==== STRICT POSITIVE GATE: ONLY REAL FISH / PRAWN IMAGES ====
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_LOAD_TRIED = False
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_AQUATIC_MIN = 0.60

CLIP_AQUATIC_PROMPTS = [
    "a photo of a fish", "a close-up photo of a fish",
    "a photo of a diseased fish with ulcers or wounds", "a photo of a fish in a pond",
    "a photo of a koi fish", "a photo of a prawn", "a photo of a shrimp",
    "a close-up photo of a vannamei shrimp", "a photo of prawns in an aquaculture pond",
]
CLIP_NON_AQUATIC_PROMPTS = [
    "a photo of a person", "a selfie of a human face", "a photo of a dog or a cat",
    "a photo of a bird", "a photo of a cow or a goat", "a photo of a car or a motorcycle",
    "a photo of a room interior", "a photo of a building", "a photo of a landscape or sky",
    "a photo of a tree, flower or plant", "a photo of cooked food on a plate",
    "a photo of a vegetable or fruit", "a screenshot of a phone or computer",
    "a photo of a document or text", "a cartoon or drawing",
    "a photo of a toy or object on a table",
    "an empty pond or water surface without any animal",
    "a photo of an insect or a spider",
]

def _load_clip_once():
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_LOAD_TRIED
    if CLIP_LOAD_TRIED:
        return CLIP_MODEL is not None
    CLIP_LOAD_TRIED = True
    try:
        from transformers import CLIPModel, CLIPProcessor
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        CLIP_MODEL.eval()
        print(f"[LOAD] CLIP aquatic gate loaded: {CLIP_MODEL_NAME}")
        return True
    except Exception as exc:
        MODEL_LOAD_ERRORS["clip_gate"] = str(exc)
        print(f"[WARN] CLIP gate NOT available: {exc}")
        return False

def clip_is_aquatic(image_path: str):
    if not _load_clip_once():
        return None, {}
    try:
        import torch
        prompts = CLIP_AQUATIC_PROMPTS + CLIP_NON_AQUATIC_PROMPTS
        with Image.open(image_path) as im:
            img = im.convert("RGB")
        inputs = CLIP_PROCESSOR(text=prompts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = CLIP_MODEL(**inputs).logits_per_image[0].softmax(dim=0).tolist()
        n_aq = len(CLIP_AQUATIC_PROMPTS)
        aquatic_total = float(sum(probs[:n_aq]))
        top_idx = int(np.argmax(probs))
        ok = (aquatic_total >= CLIP_AQUATIC_MIN) and (top_idx < n_aq)
        print(f"[CLIP GATE] aquatic={aquatic_total:.3f} top='{prompts[top_idx]}' ok={ok}")
        return ok, {"aquatic_prob": round(aquatic_total, 3), "top_label": prompts[top_idx]}
    except Exception as exc:
        print(f"[CLIP GATE] error: {type(exc).__name__}: {exc}")
        return None, {}

def strict_species_image_gate(image_path: str, species: str):
    sp_te = "చేప" if species == "fish" else "రొయ్య"
    msg_en = f"❌ This is not a {species} image. Only real fish or prawn images are accepted. Detection stopped."
    msg_te = f"❌ ఇది {sp_te} ఫోటో కాదు. కేవలం నిజమైన చేప లేదా రొయ్య ఫోటోలు మాత్రమే అనుమతించబడతాయి."
    obj_model = FISH_OBJECT_MODEL if species == "fish" else PRAWN_OBJECT_MODEL
    if obj_model is not None:
        if not _run_species_object_gate(obj_model, image_path, species):
            return _make_species_rejection(species, msg_en, msg_te)
    ok, _info = clip_is_aquatic(image_path)
    if ok is False:
        return _make_species_rejection(species, msg_en, msg_te)
    return None


'''

marker = "def analyze_species_disease("
assert code.count(marker) == 1, "analyze_species_disease not found exactly once"
assert "strict_species_image_gate" not in code, "Gate already applied"
code = code.replace(marker, GATE_BLOCK + marker, 1)

call_marker = "    yolo_res = detect_disease_spots("
assert code.count(call_marker) == 1, "YOLO call marker not found"
GATE_CALL = '''    gate_reject = strict_species_image_gate(image_path, species)
    if gate_reject:
        return gate_reject

'''
code = code.replace(call_marker, GATE_CALL + call_marker, 1)
old = "      if (!silent) loader.style.display = 'block';\n\n      try {\n        const res = await fetch(`/api/sample/"
if old in code:
    code = code.replace(old, old.replace("if (!silent) loader", "loader"), 1)

for a, b in [('"fish_weights": FISH_MODEL_PATH,', '"fish_weights": FISH_MODEL_PATHS,'),
             ('"prawn_weights": PRAWN_MODEL_PATH,', '"prawn_weights": PRAWN_MODEL_PATHS,')]:
    code = code.replace(a, b)

open(DST, "w", encoding="utf-8", newline="\n").write(code)
print("OK -> app_fixed.py created")
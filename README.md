# AquaCare Fish & Prawn Disease Detection

AquaCare is a Flask web application for fish and prawn image screening and dissolved-oxygen (DO) estimation. The project includes trained YOLO checkpoints, fish/prawn image datasets, DO prediction models, and training notebooks.

## Run the application

Use Python 3.13 (64-bit; tested) and PowerShell from this project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The launcher checks ports 5001-5099 and opens the first available URL in your browser. It reuses an existing server only if that server's `/health` confirms the current AquaCare version and loaded Fish/Prawn species gate; older servers are skipped automatically. Use the URL printed in the terminal (and opened by the launcher), not an older browser tab. To verify it, open `<printed URL>/health` and check for `AquaCare AI Platform v4.7 contextual-assistant` and `species_gate_model: true`.

If PowerShell blocks virtual-environment activation, run the commands using the virtual environment interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Features

- Fish disease screening: Healthy Fish, Bacterial Diseases, Parasitic Diseases, White Tail Diseases, and Fungal Diseases.
- Prawn disease screening: Healthy, WSSV, Black Gill, and WSSV + Black Gill.
- Fish/Prawn species verification: the selected tab is checked before disease analysis; low-confidence species matches are rejected instead of being analyzed as the wrong animal. Keep the root-level `species_gate.joblib` file beside `app_final.py`.
- Disease results include precautions, the model's medicine and food/feed guidance, and affected-image regions when available. Healthy results include indicative selling-rate ranges for the selected species; these are reference ranges, not live quotes.
- The assistant is scoped to AquaCare application help and the current AquaCare result. It supports English, Telugu, and Hindi. Without a Gemini key it provides built-in application help; configuring a valid Google AI Studio Gemini API key as `GEMINI_API_KEY` enables more flexible answers grounded in the app's documented features and current result. If Gemini returns HTTP 400, replace the key in `.env` with a valid Gemini API key, keep `GEMINI_MODEL=gemini-2.5-flash`, and restart the server. Never share the API key.
- DO estimates for fish and prawn using the included model checkpoints and water-quality inputs.
- Local inspection history, farmer Q&A, sample images, and Telugu/Hindi/English UI support.

Model outputs are screening estimates, not a veterinary diagnosis. Confirm disease and treatment decisions with a qualified fisheries or aquaculture professional.

## Optional Gemini assistant

To enable flexible Gemini-backed replies in the web assistant and `/api/assistant`, copy `.env.example` to `.env`, add a Google AI Studio Gemini API key as `GEMINI_API_KEY`, and set `GEMINI_MODEL=gemini-2.5-flash`. Without a valid key, the UI uses built-in AquaCare help and result-specific guidance. Do not share or commit `.env`.

## Run image inference from the command line

```powershell
python "fish\inference fish.py" --source "path\to\fish.jpg" --output fish_results --conf 0.5 --save_json
python infer_prawn.py --source "path\to\prawn.jpg" --output prawn_results --conf 0.5 --save_json
```

Both commands also accept a directory or an image glob as `--source`.

## Project layout

- `app.py`, `app_final.py` — Flask entry point and application.
- `static_ui/` — web interface and image assets.
- `fish/models/`, `prawn/models/` — trained disease-detection and DO model files.
- `fish/fish_yolo/`, `prawn/prawn_yolo/` — YOLO datasets and class configuration.
- `fish/*.ipynb`, `prawn/*.ipynb`, `DO/` — training and exploration notebooks.
- `requirements.txt` — application dependencies.

The model and dataset files are included in the project directory and are required for offline inference. The exported YOLO training argument files and dataset configuration may contain paths from the original training computer; update those paths before retraining on another computer. Running the web app does not require retraining.

The included DO checkpoints were saved with scikit-learn 1.7.2. Keep the version pinned in `requirements.txt`; a newer scikit-learn release may not load those serialized models correctly.

import os
import io
import uuid
import json
import sqlite3
import base64
import hashlib
import re
import socket
import secrets
import hmac
import threading
import webbrowser
from functools import wraps
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, jsonify, render_template_string, request, send_from_directory, session
from ultralytics import YOLO
import urllib.request
import urllib.error
from urllib.parse import quote_plus
import time
import requests

try:
    import cv2
except Exception:
    cv2 = None

# ==============================================================================
# 1. SETUP & DIRECTORIES
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass
app.secret_key = os.getenv("AQUACARE_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_SECURE=os.getenv("AQUACARE_COOKIE_SECURE", "").strip() == "1",
)

DB_PATH = os.path.join(BASE_DIR, "aquacare.db")
MARKETPLACE_CONTACT_PHONE = "9959269947"
FARMER_PASSWORD_HASH_ITERATIONS = 310_000


def hash_farmer_password(password: str) -> Tuple[str, str]:
    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, FARMER_PASSWORD_HASH_ITERATIONS
    )
    return salt.hex(), password_hash.hex()


def verify_farmer_password(password: str, salt_hex: str, password_hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(password_hash_hex)
    except ValueError:
        return False
    actual_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, FARMER_PASSWORD_HASH_ITERATIONS
    )
    return hmac.compare_digest(actual_hash, expected_hash)


def farmer_login_required(handler):
    @wraps(handler)
    def wrapped(*args, **kwargs):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                has_database_account = conn.execute(
                    "SELECT 1 FROM farmer_accounts LIMIT 1"
                ).fetchone() is not None
        except sqlite3.Error as exc:
            print(f"[FARMER AUTH] Could not check configured accounts: {exc}")
            return jsonify({
                "success": False,
                "message": "Could not verify farmer account setup.",
            }), 500
        has_legacy_account = bool(
            os.getenv("FARMER_USERNAME") and os.getenv("FARMER_PASSWORD")
        )
        if not has_database_account and not has_legacy_account:
            return jsonify({
                "success": False,
                "message": "Farmer account setup is required on this computer.",
            }), 503
        if not session.get("farmer_authenticated"):
            return jsonify({"success": False, "message": "Farmer login is required."}), 401
        return handler(*args, **kwargs)
    return wrapped


@app.route("/api/farmer/login", methods=["POST"])
def farmer_login():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Send farmer credentials as JSON."}), 400

    submitted_username = payload.get("username")
    submitted_password = payload.get("password")
    if not isinstance(submitted_username, str) or not isinstance(submitted_password, str):
        return jsonify({"success": False, "message": "Enter both username and password."}), 400

    username = submitted_username.strip()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            account = conn.execute("""
                SELECT username, password_salt, password_hash
                FROM farmer_accounts
                WHERE username = ? COLLATE NOCASE
            """, (username,)).fetchone()
    except sqlite3.Error as exc:
        print(f"[FARMER LOGIN] Could not load farmer account: {exc}")
        return jsonify({"success": False, "message": "Could not verify farmer login."}), 500

    valid_credentials = bool(
        account and verify_farmer_password(submitted_password, account[1], account[2])
    )
    if not account:
        configured_username = os.getenv("FARMER_USERNAME", "")
        configured_password = os.getenv("FARMER_PASSWORD", "")
        if configured_username and configured_password:
            valid_credentials = (
                hmac.compare_digest(username.casefold(), configured_username.casefold())
                and hmac.compare_digest(
                    submitted_password.encode("utf-8"),
                    configured_password.encode("utf-8"),
                )
            )
    if not valid_credentials:
        return jsonify({"success": False, "message": "Incorrect farmer username or password."}), 401

    session.clear()
    session["farmer_authenticated"] = True
    session["farmer_username"] = account[0] if account else username
    return jsonify({"success": True, "message": "Farmer login successful."})


@app.route("/api/farmer/setup", methods=["POST"])
def farmer_setup():
    if request.remote_addr not in {"127.0.0.1", "::1"}:
        return jsonify({"success": False, "message": "Farmer account setup is available only on this computer."}), 403

    username = os.getenv("FARMER_USERNAME", "")
    password = os.getenv("FARMER_PASSWORD", "")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            has_database_account = conn.execute(
                "SELECT 1 FROM farmer_accounts LIMIT 1"
            ).fetchone() is not None
    except sqlite3.Error as exc:
        print(f"[FARMER SETUP] Could not check existing accounts: {exc}")
        return jsonify({"success": False, "message": "Could not verify farmer account setup."}), 500
    if has_database_account or (username and password):
        return jsonify({
            "success": False,
            "message": "Farmer setup is already complete. Log in with the existing farmer account, then use Add another farmer login in the dashboard.",
        }), 409

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Send farmer account details as JSON."}), 400

    submitted_username = payload.get("username")
    submitted_password = payload.get("password")
    if not isinstance(submitted_username, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", submitted_username):
        return jsonify({
            "success": False,
            "message": "Choose a username of 3-64 letters, numbers, dots, underscores, or hyphens.",
        }), 400
    if not isinstance(submitted_password, str) or not re.fullmatch(
        r"[A-Za-z0-9!@%^&*()+=._-]{12,128}", submitted_password
    ):
        return jsonify({
            "success": False,
            "message": "Choose a password of at least 12 characters using letters, numbers, or !@%^&*()+=._-.",
        }), 400

    dotenv_path = os.path.join(BASE_DIR, ".env")
    generated_secret = os.getenv("AQUACARE_SECRET_KEY") or secrets.token_urlsafe(48)
    replace_partial_legacy_credentials = not (username and password)
    try:
        with open(dotenv_path, "a+", encoding="utf-8") as dotenv_file:
            dotenv_file.seek(0)
            existing_content = dotenv_file.read()
            settings = {
                "FARMER_USERNAME": submitted_username,
                "FARMER_PASSWORD": submitted_password,
                "AQUACARE_SECRET_KEY": generated_secret,
            }
            existing_keys = set()
            updated_lines = []
            for line in existing_content.splitlines():
                if not line.lstrip().startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key in settings:
                        if key in {"FARMER_USERNAME", "FARMER_PASSWORD"} and replace_partial_legacy_credentials:
                            line = f"{key}={settings[key]}"
                            existing_keys.add(key)
                        elif value.strip().strip("\"'"):
                            existing_keys.add(key)
                        else:
                            line = f"{key}={settings[key]}"
                            existing_keys.add(key)
                updated_lines.append(line)
            missing_settings = [key for key in settings if key not in existing_keys]
            new_content = "\n".join(updated_lines)
            if existing_content and not existing_content.endswith(("\n", "\r")):
                new_content += "\n"
            for key in missing_settings:
                new_content += f"{key}={settings[key]}\n"
            if new_content != existing_content:
                dotenv_file.seek(0)
                dotenv_file.write(new_content)
                dotenv_file.truncate()
            dotenv_file.flush()
    except OSError as exc:
        print(f"[FARMER SETUP] Could not save local credentials: {exc}")
        return jsonify({"success": False, "message": "Could not save farmer setup to the local .env file."}), 500

    os.environ["FARMER_USERNAME"] = submitted_username
    os.environ["FARMER_PASSWORD"] = submitted_password
    os.environ["AQUACARE_SECRET_KEY"] = generated_secret
    app.secret_key = generated_secret
    salt, password_hash = hash_farmer_password(submitted_password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO farmer_accounts (username, password_salt, password_hash)
                VALUES (?, ?, ?)
            """, (submitted_username, salt, password_hash))
    except sqlite3.Error as exc:
        print(f"[FARMER SETUP] Could not save account record: {exc}")
        return jsonify({"success": False, "message": "Could not save farmer account."}), 500
    session.clear()
    session["farmer_authenticated"] = True
    session["farmer_username"] = submitted_username
    return jsonify({"success": True, "message": "Farmer account created and signed in."}), 201


@app.route("/api/farmer/accounts", methods=["POST"])
@farmer_login_required
def create_farmer_account():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Send farmer account details as JSON."}), 400

    username = payload.get("username")
    password = payload.get("password")
    if not isinstance(username, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", username):
        return jsonify({
            "success": False,
            "message": "Choose a username of 3-64 letters, numbers, dots, underscores, or hyphens.",
        }), 400
    if not isinstance(password, str) or not re.fullmatch(
        r"[A-Za-z0-9!@%^&*()+=._-]{12,128}", password
    ):
        return jsonify({
            "success": False,
            "message": "Choose a password of at least 12 characters using letters, numbers, or !@%^&*()+=._-.",
        }), 400

    salt, password_hash = hash_farmer_password(password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO farmer_accounts (username, password_salt, password_hash)
                VALUES (?, ?, ?)
            """, (username, salt, password_hash))
        return jsonify({"success": True, "username": username}), 201
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "That farmer username is already in use."}), 409
    except sqlite3.Error as exc:
        print(f"[FARMER ACCOUNTS] Could not create account: {exc}")
        return jsonify({"success": False, "message": "Could not create farmer account."}), 500


@app.route("/api/farmer/logout", methods=["POST"])
@farmer_login_required
def farmer_logout():
    session.clear()
    return jsonify({"success": True})


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS farmer_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        configured_username = os.getenv("FARMER_USERNAME", "").strip()
        configured_password = os.getenv("FARMER_PASSWORD", "")
        if configured_username and configured_password:
            salt, password_hash = hash_farmer_password(configured_password)
            cur.execute("""
                INSERT OR IGNORE INTO farmer_accounts (username, password_salt, password_hash)
                VALUES (?, ?, ?)
            """, (configured_username, salt, password_hash))
        cur.execute("""
            CREATE TABLE IF NOT EXISTS inspection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                inspection_type TEXT,
                species TEXT,
                predicted_class TEXT,
                confidence REAL,
                status TEXT,
                notes TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS purchase_inquiries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                farmer_name TEXT,
                phone TEXT,
                location TEXT,
                product_name TEXT,
                quantity TEXT,
                status TEXT DEFAULT 'Pending'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS harvest_listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                seller_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                location TEXT NOT NULL,
                species TEXT NOT NULL CHECK (species IN ('fish', 'prawn')),
                variety TEXT NOT NULL,
                quantity_kg REAL NOT NULL CHECK (quantity_kg > 0),
                quantity_sold_kg REAL NOT NULL DEFAULT 0 CHECK (quantity_sold_kg >= 0),
                quantity_reserved_kg REAL NOT NULL DEFAULT 0 CHECK (quantity_reserved_kg >= 0),
                price_per_kg REAL NOT NULL CHECK (price_per_kg > 0),
                notes TEXT NOT NULL DEFAULT ''
            )
        """)
        listing_columns = {
            row[1] for row in cur.execute("PRAGMA table_info(harvest_listings)")
        }
        if "quantity_sold_kg" not in listing_columns:
            cur.execute("""
                ALTER TABLE harvest_listings
                ADD COLUMN quantity_sold_kg REAL NOT NULL DEFAULT 0
            """)
        reservation_column_added = "quantity_reserved_kg" not in listing_columns
        if reservation_column_added:
            cur.execute("""
                ALTER TABLE harvest_listings
                ADD COLUMN quantity_reserved_kg REAL NOT NULL DEFAULT 0
            """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                buyer_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                delivery_location TEXT NOT NULL,
                order_type TEXT NOT NULL CHECK (order_type IN ('harvest', 'feed', 'medicine')),
                product_name TEXT NOT NULL,
                seller_listing_id INTEGER,
                quantity REAL NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price >= 0),
                estimated_total REAL NOT NULL CHECK (estimated_total >= 0),
                status TEXT NOT NULL DEFAULT 'Placed',
                FOREIGN KEY (seller_listing_id) REFERENCES harvest_listings(id)
            )
        """)
        if reservation_column_added:
            pending_harvest_orders = cur.execute("""
                SELECT seller_listing_id, SUM(quantity)
                FROM marketplace_orders
                WHERE order_type = 'harvest'
                  AND status = 'Placed'
                  AND seller_listing_id IS NOT NULL
                GROUP BY seller_listing_id
            """).fetchall()
            for listing_id, reserved_quantity in pending_harvest_orders:
                cur.execute("""
                    UPDATE harvest_listings
                    SET quantity_sold_kg = MAX(0, quantity_sold_kg - ?),
                        quantity_reserved_kg = ?
                    WHERE id = ?
                """, (reserved_quantity, reserved_quantity, listing_id))
        cur.execute("""
            UPDATE marketplace_orders
            SET status = 'Accepted'
            WHERE status = 'Done'
        """)
        cur.execute("""
            UPDATE marketplace_orders
            SET status = 'Placed'
            WHERE status NOT IN ('Placed', 'Accepted')
        """)
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[DB] Init warning: {exc}")

init_db()

# ==============================================================================
# 2. MODEL PATHS & AUTO-DISCOVERY
# ==============================================================================
def find_model_file(*relative_candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in relative_candidates:
        full_path = os.path.join(BASE_DIR, *candidate)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            return full_path
    return None

def find_all_model_files(*candidates) -> List[str]:
    found = []
    for candidate in candidates:
        full_path = os.path.join(BASE_DIR, *candidate)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0 and full_path not in found:
            found.append(full_path)
    return found

def discover_species_disease_models(species: str) -> List[str]:
    """Find trained disease checkpoints even when project folder names differ."""
    species = str(species).strip().lower()
    if species not in {"fish", "prawn"}:
        return []
    explicit = [
        # ---- RETRAINED MODELS (highest priority - better accuracy) ----
        (species, "models", "fish_retrain_v2", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_retrain_v2", "weights", "best.pt"),
        # ---- ORIGINAL MODELS (fallback) ----
        (species, "models", "fish_disease_final", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_disease_final", "weights", "best.pt"),
        (species, "models", "fish_disease_final_fast", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_disease_final_fast", "weights", "best.pt"),
        (species, "models", "fish_disease", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_disease", "weights", "best.pt"),
        (species, "models", "fish_disease-2", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_fast_test", "weights", "best.pt"),
        (species, "models", "fish_fast_test", "weights", "best.pt") if species == "fish" else (species, "models", "prawn_disease-2", "weights", "best.pt"),
    ]
    found = find_all_model_files(*explicit)
    if found:
        return found

    candidates = []
    seen = set()
    skip_dirs = {"uploads", "__pycache__", ".git", "node_modules", ".venv", "venv"}
    scan_roots = [os.path.join(BASE_DIR, species, "models"), BASE_DIR]
    for root in scan_roots:
        if not os.path.isdir(root):
            continue
        try:
            for current_root, dirnames, files in os.walk(root):
                dirnames[:] = [d for d in dirnames if d.lower() not in skip_dirs]
                for fn in files:
                    low = fn.lower()
                    if not low.endswith((".pt", ".onnx")):
                        continue
                    if low in {"yolo11n.pt", "yolo26n.pt", "yolov8n.pt", "yolov8s.pt"}:
                        continue
                    full = os.path.abspath(os.path.join(current_root, fn))
                    if full in seen:
                        continue
                    seen.add(full)
                    path_low = full.lower().replace("\\", "/")
                    if root == BASE_DIR and (f"/{species}/" not in path_low and species not in os.path.basename(path_low)):
                        continue
                    score = 0
                    for token, weight in (("disease", 100), ("final", 80), ("trained", 60), ("best.pt", 40), ("fast", 20)):
                        if token in path_low:
                            score += weight
                    if path_low.endswith("last.pt"):
                        score -= 30
                    candidates.append((score, full))
        except Exception as exc:
            print(f"[MODEL DISCOVERY] {species}: {type(exc).__name__}: {exc}")
    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    return [p for _, p in candidates[:6]]

FISH_MODEL_PATHS = discover_species_disease_models("fish")
PRAWN_MODEL_PATHS = discover_species_disease_models("prawn")

COCO_MODEL_PATH = find_model_file(
    ("fish", "yolo11n.pt"),
    ("prawn", "yolo11n.pt"),
    ("fish", "yolo26n.pt"),
)

FISH_DO_PATH = find_model_file(
    ("fish", "models", "fish_do_best_model.pkl"),
    ("fish", "models", "fish_do_model.pkl"),
)

PRAWN_DO_PATH = find_model_file(
    ("prawn", "models", "prawn_do_best_model.pkl"),
    ("prawn", "models", "prawn_do_model.pkl"),
)

# Load YOLO & ML Models safely
ALL_FISH_MODELS: List[Any] = []
ALL_PRAWN_MODELS: List[Any] = []
FISH_DISEASE_MODEL = None
PRAWN_DISEASE_MODEL = None
COCO_DETECTOR = None
FISH_DO_MODEL = None
PRAWN_DO_MODEL = None
MODEL_LOAD_ERRORS = {}

for p in FISH_MODEL_PATHS:
    try:
        m = YOLO(p)
        ALL_FISH_MODELS.append(m)
        print(f"[LOAD] Fish YOLO model loaded: {p}")
    except Exception as exc:
        MODEL_LOAD_ERRORS[f"fish_{os.path.basename(p)}"] = str(exc)
        print(f"[WARN] Fish YOLO load warning for {p}: {exc}")

for p in PRAWN_MODEL_PATHS:
    try:
        m = YOLO(p)
        ALL_PRAWN_MODELS.append(m)
        print(f"[LOAD] Prawn YOLO model loaded: {p}")
    except Exception as exc:
        MODEL_LOAD_ERRORS[f"prawn_{os.path.basename(p)}"] = str(exc)
        print(f"[WARN] Prawn YOLO load warning for {p}: {exc}")

FISH_DISEASE_MODEL = ALL_FISH_MODELS[0] if ALL_FISH_MODELS else None
PRAWN_DISEASE_MODEL = ALL_PRAWN_MODELS[0] if ALL_PRAWN_MODELS else None

if COCO_MODEL_PATH:
    try:
        COCO_DETECTOR = YOLO(COCO_MODEL_PATH)
        print(f"[LOAD] Base COCO Detector loaded for negative validation: {COCO_MODEL_PATH}")
    except Exception as exc:
        print(f"[WARN] Loading COCO Detector: {exc}")

# ------------------------------------------------------------------------------
# OPTIONAL TRUE SPECIES OBJECT GATES
# A dedicated fish/prawn detect checkpoint is used only when its class names
# actually contain an organism class such as 'fish' or 'prawn'. Generic YOLO
# checkpoints and disease labels such as 'Healthy Fish' are never treated as
# organism classes.
# ------------------------------------------------------------------------------
FISH_OBJECT_MODEL = None
PRAWN_OBJECT_MODEL = None
FISH_OBJECT_MODEL_PATH = None
PRAWN_OBJECT_MODEL_PATH = None

def _model_name_list(model):
    names = getattr(model, "names", {}) if model is not None else {}
    if isinstance(names, dict):
        return [str(v) for _, v in sorted(names.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []

def _has_exact_species_object_class(model, species: str) -> bool:
    vals = {re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_") for x in _model_name_list(model)}
    if species == "fish":
        allowed = {"fish", "fishes", "carp", "tilapia", "rohu", "catla", "mrigal", "pangasius", "murrel"}
    else:
        allowed = {"prawn", "prawns", "shrimp", "vannamei", "tiger_shrimp", "black_tiger"}
    return any(v in allowed for v in vals)

def _discover_true_species_object_models():
    global FISH_OBJECT_MODEL, PRAWN_OBJECT_MODEL, FISH_OBJECT_MODEL_PATH, PRAWN_OBJECT_MODEL_PATH
    candidates = []
    # Start with explicit project object checkpoints if present.
    preferred = [
        os.path.join(BASE_DIR, "fish", "models", "fish_object", "weights", "best.pt"),
        os.path.join(BASE_DIR, "fish", "models", "fish_detection", "weights", "best.pt"),
        os.path.join(BASE_DIR, "prawn", "models", "prawn_object", "weights", "best.pt"),
        os.path.join(BASE_DIR, "prawn", "models", "prawn_detection", "weights", "best.pt"),
        os.path.join(BASE_DIR, "models", "fish_object", "best.pt"),
        os.path.join(BASE_DIR, "models", "prawn_object", "best.pt"),
        os.path.join(BASE_DIR, "models", "aquatic_object", "best.pt"),
        os.path.join(BASE_DIR, "best.pt"),
        os.path.join(BASE_DIR, "fish_best.pt"),
        os.path.join(BASE_DIR, "prawn_best.pt"),
        os.path.join(BASE_DIR, "fish_object.pt"),
        os.path.join(BASE_DIR, "prawn_object.pt"),
    ]
    for p in preferred:
        if os.path.exists(p) and os.path.getsize(p) > 0 and p not in candidates:
            candidates.append(p)
    try:
        for root, _, files in os.walk(BASE_DIR):
            for fn in files:
                if not fn.lower().endswith((".pt", ".onnx")):
                    continue
                p = os.path.join(root, fn)
                low = p.lower()
                # Avoid generic pretrained base checkpoints.
                if os.path.basename(low) in {"yolo11n.pt", "yolo26n.pt", "yolov8n.pt", "yolov8s.pt"}:
                    continue
                if p not in candidates:
                    candidates.append(p)
    except Exception:
        pass

    candidates.sort(key=lambda p: ("best.pt" not in p.lower(), "last.pt" in p.lower(), len(p)))
    for p in candidates:
        try:
            model = YOLO(p)
            if str(getattr(model, "task", "")).lower() != "detect":
                continue
            has_fish = _has_exact_species_object_class(model, "fish")
            has_prawn = _has_exact_species_object_class(model, "prawn")

            if FISH_OBJECT_MODEL is None and has_fish:
                FISH_OBJECT_MODEL = model
                FISH_OBJECT_MODEL_PATH = p
                print(f"[LOAD] TRUE fish object model: {p} | classes={_model_name_list(model)}")

            if PRAWN_OBJECT_MODEL is None and has_prawn:
                PRAWN_OBJECT_MODEL = model
                PRAWN_OBJECT_MODEL_PATH = p
                print(f"[LOAD] TRUE prawn object model: {p} | classes={_model_name_list(model)}")

            if FISH_OBJECT_MODEL is not None and PRAWN_OBJECT_MODEL is not None:
                break
        except Exception as exc:
            MODEL_LOAD_ERRORS[f"species_object_{os.path.basename(p)}"] = str(exc)

def _load_do_checkpoint(path: Optional[str], label: str):
    if not path:
        return None, {}
    obj = joblib.load(path)
    model = obj.get("pipeline") if isinstance(obj, dict) else obj
    if not hasattr(model, "predict"):
        raise TypeError(f"{label} DO checkpoint does not contain a predict() model.")
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    return model, meta

def _load_meta_json(species_name: str, meta_file: str) -> dict:
    p = os.path.join(BASE_DIR, species_name, "models", meta_file)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

FISH_DO_META = _load_meta_json("fish", "fish_do_metadata.json")
PRAWN_DO_META = _load_meta_json("prawn", "prawn_do_metadata.json")

if FISH_DO_PATH:
    try:
        FISH_DO_MODEL, _m = _load_do_checkpoint(FISH_DO_PATH, "Fish")
        if _m:
            FISH_DO_META.update(_m)
        print(f"[LOAD] Fish DO model loaded: {FISH_DO_PATH}")
    except Exception as exc:
        MODEL_LOAD_ERRORS["fish_do"] = str(exc)
        print(f"[ERROR] Loading Fish DO: {exc}")

if PRAWN_DO_PATH:
    try:
        PRAWN_DO_MODEL, _pm = _load_do_checkpoint(PRAWN_DO_PATH, "Prawn")
        if _pm:
            PRAWN_DO_META.update(_pm)
        print(f"[LOAD] Prawn DO model loaded: {PRAWN_DO_PATH}")
    except Exception as exc:
        MODEL_LOAD_ERRORS["prawn_do"] = str(exc)
        print(f"[ERROR] Loading Prawn DO: {exc}")

# ==============================================================================
# 3. CLINICAL, SELLING & BUYING DATABASE (TRILINGUAL + TELENGLISH)
# ==============================================================================
FISH_CLASSES = {
    0: "Healthy Fish",
    1: "Bacterial Diseases",
    2: "Parasitic Diseases",
    3: "White Tail Diseases",
    4: "Fungal Diseases",
}

PRAWN_CLASSES = {
    0: "Healthy",
    1: "WSSV",
    2: "Black_Gill",
    3: "WSSV_Black_Gill",
}

SPECIES_GATE_MODEL = None
SPECIES_GATE_PATH = find_model_file(("species_gate.joblib",))
if SPECIES_GATE_PATH:
    try:
        _species_gate_artifact = joblib.load(SPECIES_GATE_PATH)
        if (
            isinstance(_species_gate_artifact, dict)
            and _species_gate_artifact.get("feature_version") == 1
            and _species_gate_artifact.get("species_order") == ["fish", "prawn"]
            and _species_gate_artifact.get("feature_count") == 13
            and hasattr(_species_gate_artifact.get("classifier"), "predict_proba")
            and getattr(_species_gate_artifact["classifier"], "n_features_in_", None) == 13
        ):
            SPECIES_GATE_MODEL = _species_gate_artifact["classifier"]
            print(f"[LOAD] Fish/Prawn species gate loaded: {SPECIES_GATE_PATH}")
        else:
            raise ValueError("Species gate artifact has an unsupported or invalid format.")
    except Exception as exc:
        MODEL_LOAD_ERRORS["species_gate"] = str(exc)
        print(f"[ERROR] Loading Fish/Prawn species gate: {type(exc).__name__}: {exc}")

# Live market selling benchmark pricing (Andhra Pradesh & National Markets)
MARKET_SELLING_RATES = {
    "fish": {
        "Rohu / Catla (1.5 - 2.5 kg)": {"price_range": "₹150 - ₹185 / kg", "avg_price": 165},
        "Catla Large (3.0+ kg)": {"price_range": "₹170 - ₹205 / kg", "avg_price": 185},
        "Pangasius / Basa": {"price_range": "₹110 - ₹130 / kg", "avg_price": 120},
        "Tilapia (GIFT Monosex)": {"price_range": "₹125 - ₹145 / kg", "avg_price": 135},
        "Murrel / Korameenu": {"price_range": "₹480 - ₹560 / kg", "avg_price": 520},
    },
    "prawn": {
        "30 Count (Large Export)": {"price_range": "₹400 - ₹430 / kg", "avg_price": 415},
        "40 Count (Medium-Large)": {"price_range": "₹340 - ₹370 / kg", "avg_price": 355},
        "50 Count (Standard)": {"price_range": "₹290 - ₹320 / kg", "avg_price": 305},
        "60 Count (Medium)": {"price_range": "₹250 - ₹280 / kg", "avg_price": 265},
        "100 Count (Small)": {"price_range": "₹190 - ₹220 / kg", "avg_price": 205},
    }
}

BUYING_CATALOG = [
    {
        "id": "med_kmno4",
        "category": "medicine",
        "title_en": "Potassium Permanganate (KMnO4 99%)",
        "title_te": "పొటాషియం పర్మాంగనేట్ (KMnO4 99%)",
        "title_hi": "पोटैशियम परमैंगनेट (KMnO4 99%)",
        "title_tel": "Potassium Permanganate (KMnO4 99%)",
        "price": "₹280 / kg (5kg pack ₹1,250)",
        "unit_price": 280,
        "purpose_en": "Pond water disinfectant, bacterial ulcers, fin rot & fungal dip.",
        "purpose_te": "చెరువు నీటి శానిటైజర్, బ్యాక్టీరియల్ పుండ్లు, తోక కుళ్ళు నివారణ.",
        "purpose_hi": "तालाब जल कीटाणुनाशक, बैक्टीरियल घाव और फंगस उपचार।",
        "purpose_tel": "Neellalo pundlu, toka kullu mariyu fungal rogalu nivarana."
    },
    {
        "id": "med_otc",
        "category": "medicine",
        "title_en": "Oxytetracycline (OTC 50% Feed Grade)",
        "title_te": "ఆక్సిటెట్రాసైక్లిన్ (OTC 50% ఫీడ్ గ్రేడ్)",
        "title_hi": "ऑक्सीटेट्रासाइक्लिन (OTC 50% फीड ग्रेड)",
        "title_tel": "Oxytetracycline (OTC 50% Feed Grade)",
        "price": "₹750 / kg",
        "unit_price": 750,
        "purpose_en": "Broad-spectrum oral antibiotic for bacterial dropsy & ulcers.",
        "purpose_te": "చేపలకు మేతలో కలిపే శక్తివంతమైన యాంటీబయాటిక్.",
        "purpose_hi": "बैक्टीरियल घावों और पूंछ सड़न के लिए प्रभावी एंटीबायोटिक।",
        "purpose_tel": "Chepala pundlu, toka kullu taggadaniki meethalo kalise antibiotic."
    },
    {
        "id": "med_zeolite",
        "category": "medicine",
        "title_en": "Zeolite Granules + Yucca Extract",
        "title_te": "జియోలైట్ గుళికలు + యుక్కా లిక్విడ్",
        "title_hi": "जियोलाइट दाने + युक्का लिक्विड",
        "title_tel": "Zeolite Granules + Yucca Extract",
        "price": "₹420 / 25 kg bag",
        "unit_price": 420,
        "purpose_en": "Bottom sludge detoxifier, removes toxic ammonia & cures black gill.",
        "purpose_te": "చెరువు అడుగున విషవాయువుల తొలగింపు, నల్ల మొప్ప నివారణ.",
        "purpose_hi": "तालाब के तल की गैसों को सोखता है, अमोनिया और ब्लैक गिल रोकता है।",
        "purpose_tel": "Adugu chetta shuddhi, ammonia thagginchadam, nalla moppa nivarana."
    },
    {
        "id": "med_bkc",
        "category": "medicine",
        "title_en": "BKC 80% Aquaculture Sanitizer",
        "title_te": "BKC 80% శానిటైజర్",
        "title_hi": "BKC 80% सैनिटाइज़र",
        "title_tel": "BKC 80% Sanitizer",
        "price": "₹380 / Liter",
        "unit_price": 380,
        "purpose_en": "Potent water disinfectant against protozoan and bacterial blooms.",
        "purpose_te": "నీటిలోని హానికర బాక్టీరియాను అరికట్టే శానిటైజర్.",
        "purpose_hi": "हानिकारक बैक्टीरिया और प्रोटोजोआ को नष्ट करने वाला सैनिटाइज़र।",
        "purpose_tel": "Neetiloni bacterial & protozoan load thagginche sanitizer."
    },
    {
        "id": "med_probiotic",
        "category": "medicine",
        "title_en": "Soil & Water Probiotics (Bacillus complex)",
        "title_te": "సాయిల్ & వాటర్ ప్రోబయోటిక్స్",
        "title_hi": "मिट्टी और पानी के प्रोबायोटिक्स",
        "title_tel": "Soil & Water Probiotics",
        "price": "₹650 / kg",
        "unit_price": 650,
        "purpose_en": "Digests organic sludge, improves water transparency & prevents gill rot.",
        "purpose_te": "అడుగున మురికిని కరిగించి మేత వినియోగాన్ని పెంచుతుంది.",
        "purpose_hi": "तल के कचरे को घोलकर पानी की गुणवत्ता और ग्रोथ सुधारता है।",
        "purpose_tel": "Adugu mud clean chesi rogalu rakunda chese manchi bacteria."
    },
    {
        "id": "seed_vannamei",
        "category": "seed",
        "title_en": "SPF Vannamei Post Larvae (PL-10)",
        "title_te": "SPF వనామి రొయ్య పిల్లలు (PL-10)",
        "title_hi": "SPF वेनामेई झींगा बीज (PL-10)",
        "title_tel": "SPF Vannamei Prawn Seed (PL-10)",
        "price": "₹0.38 / seed",
        "unit_price": 0.38,
        "purpose_en": "Certified pathogen-free vigorous post larvae with >98% PCR clearance.",
        "purpose_te": "వ్యాధి రహిత నాణ్యమైన హేచరీ వనామి రొయ్య సీడ్.",
        "purpose_hi": "रोग-मुक्त, प्रमाणित उच्च उत्तरजीविता वाला प्रीमियम बीज।",
        "purpose_tel": "Rogalu leni premium quality SPF hatchery seed."
    },
    {
        "id": "seed_rohu",
        "category": "seed",
        "title_en": "Rohu / Catla 3-4 inch Fingerlings",
        "title_te": "రోహు / బొచ్చె 3-4 అంగుళాల పిల్లలు",
        "title_hi": "रोहू / कतला 3-4 इंच फिंगरलिंग्स",
        "title_tel": "Rohu / Catla 3-4 inch Fingerlings",
        "price": "₹1.60 / fingerling",
        "unit_price": 1.60,
        "purpose_en": "Sturdy riverine broodstock fingerlings ready for stocking.",
        "purpose_te": "వేగంగా పెరిగే నాణ్యమైన చేప పిల్లలు.",
        "purpose_hi": "तेजी से बढ़ने वाले स्वस्थ नदीय मूल के मछली बच्चे।",
        "purpose_tel": "Vegamga perige arogyakaramaina chepa pillalu."
    },
    {
        "id": "feed_fish",
        "category": "feed",
        "title_en": "Floating Fish Feed (28% Protein, 4mm)",
        "title_te": "ఫ్లోటింగ్ చేపల మేత (28% ప్రోటీన్, 4mm)",
        "title_hi": "फ्लोटिंग फिश फीड (28% प्रोटीन, 4mm)",
        "title_tel": "Floating Chepala Meetha (28% Protein)",
        "price": "₹1,480 / 40 kg bag",
        "unit_price": 1480,
        "purpose_en": "High FCR floating extruded pellets for maximum weight gain.",
        "purpose_te": "నీటిపై తేలియాడే అధిక పోషకాల చేపల మేత.",
        "purpose_hi": "पानी पर तैरने वाला उच्च FCR मछली आहार, तेज विकास के लिए।",
        "purpose_tel": "Neellapai thele quality chepala feed, fast growth kosam."
    },
    {
        "id": "feed_prawn",
        "category": "feed",
        "title_en": "Vannamei Shrimp Grower Feed (35% Protein)",
        "title_te": "వనామి రొయ్యల మేత (35% ప్రోటీన్)",
        "title_hi": "वेनामेई झींगा फीड (35% प्रोटीन)",
        "title_tel": "Vannamei Royyala Meetha (35% Protein)",
        "price": "₹2,850 / 25 kg bag",
        "unit_price": 2850,
        "purpose_en": "Water-stable premium shrimp pellets for rapid count growth.",
        "purpose_te": "నీటిలో త్వరగా కరగని మేలైన రొయ్యల మేత.",
        "purpose_hi": "पानी में स्थिर रहने वाला प्रीमियम झींगा आहार।",
        "purpose_tel": "Water stable quality shrimp feed, fast count weight peragadaniki."
    }
]

# ==============================================================================
# 3A. DETECTION SAFETY THRESHOLDS
# ==============================================================================
# Disease predictions need real visual/model evidence. These thresholds are used
# only to reject weak false-positive disease boxes; they do not invent classes.
DISEASE_MIN_CONF = 0.50
HEALTHY_MIN_CONF = 0.45
DISEASE_HEALTHY_MARGIN = 0.12
SINGLE_VIEW_DISEASE_CONF = 0.50
MIN_DISEASE_VIEW_SUPPORT = 2

# Hard species-consistency thresholds. These are used only to prevent
# Fish<->Prawn mix-ups; the selected species is never auto-switched.
SPECIES_GATE_ACCEPT_CONF = 0.80
SPECIES_GATE_FISH_BORDERLINE_CONF = 0.78
SPECIES_GATE_FISH_BORDERLINE_DETECTION_CONF = 0.25
SPECIES_GATE_OTHER_CONFIRM_CONF = 0.30
SPECIES_GATE_PRAWN_BORDERLINE_CONF = 0.60
SPECIES_GATE_PRAWN_BORDERLINE_DETECTION_CONF = 0.40
SPECIES_GATE_HEALTHY_CONFLICT_CONF = 0.70
SPECIES_GATE_TARGET_HEALTHY_CONFLICT_CONF = 0.95
BLACK_GILL_MIN_CONF = 0.55
HEALTHY_OVERRIDE_CONF = 0.45
STRONG_DISEASE_CONF = 0.84

CLINICAL_DATABASE: Dict[str, Dict[str, Any]] = {
    "Healthy Fish": {
        "is_healthy": True,
        "species": "fish",
        "title_en": "Healthy Fish (Grade-A Commercial Quality)",
        "title_te": "ఆరోగ్యకరమైన చేప (గ్రేడ్-ఏ మార్కెట్ నాణ్యత)",
        "title_hi": "स्वस्थ मछली (ग्रेड-ए व्यावसायिक गुणवत्ता)",
        "title_tel": "Arogyakaramaina Chepa (Grade-A Market Quality)",
        "severity": "Safe",
        "severity_color": "#10b981",
        "symptoms_en": "Active swimming reflex, smooth undamaged scales, intact clear fins, bright red gills.",
        "symptoms_te": "చేపలు చురుకుగా ఈదుతున్నాయి, రెక్కలు మరియు పొలుసులు బలంగా ఉన్నాయి, మొప్పలు ఎర్రగా ఉన్నాయి.",
        "symptoms_hi": "सक्रिय तैराकी, मजबूत पंख और तराजू, साफ चमकदार आंखें और स्वस्थ लाल गलफड़े।",
        "symptoms_tel": "Chepalu active ga eeduthunnayi, rekkalu polusulu bagunnayi.",
        "market_status_en": "Grade-A Harvest Ready. Maximum commercial trading demand.",
        "market_status_te": "మార్కెట్ అమ్మకానికి మరియు హార్వెస్టింగ్ కు సిద్ధంగా ఉంది. గ్రేడ్-ఏ ధర లభిస్తుంది.",
        "market_status_hi": "ग्रेड-ए कटाई के लिए तैयार। उच्च बाजार मांग और उत्तम दरें।",
        "market_status_tel": "Market lo ammadaniki ready ga undi. Manchi Grade-A rate vasthundi.",
        "selling_rates": MARKET_SELLING_RATES["fish"],
        "harvest_guide_en": "1. Harvest early dawn (2:00 AM - 6:00 AM) to maintain firm muscle texture.\n2. Keep 1:1 ice slurry ready for immediate cold-chain transport.\n3. Contact 2-3 local traders/auctioneers for maximum bidding rate.",
        "harvest_guide_te": "1. తెల్లవారుజామున (2:00 AM - 6:00 AM) చల్లగా ఉన్నప్పుడు పట్టండి.\n2. చేపలు తాజాగా ఉండటానికి 1:1 ఐస్ మిశ్రమం సిద్ధంగా ఉంచుకోండి.\n3. మంచి ధర కోసం కనీసం 2-3 మంది వ్యాపారులతో మాట్లాడి నిర్ణయించండి.",
        "harvest_guide_hi": "1. सुबह के समय (2:00 AM - 6:00 AM) कटाई करें ताकि ताजगी बनी रहे।\n2. 1:1 बर्फ के पानी में ठंडा रखें।\n3. बेहतर मूल्य के लिए स्थानीय व्यापारियों से संपर्क करें।",
        "harvest_guide_tel": "1. Poddunne 2:00 AM - 6:00 AM madhyalo harvest cheyandi.\n2. Neellu mariyu ice 1:1 ratio lo chilling cheyandi.\n3. Ekkuva rate kosam 2-3 buyers tho matladi fix cheyandi.",
        "food_guide_en": "1. Normal Feeding: Continue standard feeding schedule (2-3% of body biomass daily).\n2. Feed Quality: Use floating extruded pellets (28% protein) for optimum growth.\n3. Check-trays: Inspect trays 2 hours post broadcast to ensure zero feed wastage.",
        "food_guide_te": "1. సాధారణ మేత: శరీర బరువులో 2-3% సాధారణ ఫ్లోటింగ్ మేత వేయండి.\n2. నాణ్యత: అధిక పెరుగుదల కోసం 28% ప్రోటీన్ ఉన్న మేత వాడండి.\n3. ట్రేలు: మేత వేసిన 2 గంటల తర్వాత ఫీడ్ ట్రేలను తనిఖీ చేయండి.",
        "food_guide_hi": "1. सामान्य आहार: दैनिक बायोमास का 2-3% फ्लोटिंग चारा देना जारी रखें।\n2. गुणवत्ता: अच्छी ग्रोथ के लिए 28% प्रोटीन युक्त चारा उपयोग करें।\n3. ट्रे जाँच: आहार देने के 2 घंटे बाद फीड ट्रे की जाँच करें।",
        "food_guide_tel": "1. Normal feed schedule continue cheyandi (2-3% body weight).\n2. Quality floating feed vadandi.\n3. Feed trays 2 hours tarvatha check cheyandi.",
        "food_links": [
            {
                "title_en": "AquaCare Store: Floating Fish Feed (28% Protein, 40kg)",
                "title_te": "AquaCare స్టోర్: ఫ్లోటింగ్ చేపల మేత (28% ప్రోటీన్)",
                "title_hi": "एक्वाकेयर स्टोर: फ्लोटिंग मछली चारा (28% प्रोटीन)",
                "title_tel": "AquaCare Store: Floating Fish Feed (28% Protein)",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:feed_fish",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "Godrej Aquafeed (GrowMax Floating Pellets)",
                "title_te": "గోద్రెజ్ ఆగ్రోవెట్ ఆక్వాఫీడ్ (అధికారిక వెబ్‌సైట్)",
                "title_hi": "गोदरेज एग्रोवेट एक्वाफीड (आधिकारिक वेबसाइट)",
                "title_tel": "Godrej Agrovet Aquafeed (Official Website)",
                "badge_en": "Manufacturer",
                "badge_te": "తయారీదారు",
                "badge_hi": "निर्माता",
                "badge_tel": "Manufacturer",
                "url": "https://www.godrejagrovet.com/businesses/aquafeed",
                "type": "external",
                "icon": "fa-solid fa-wheat-awn"
            },
            {
                "title_en": "IndiaMART: Floating Fish Feed Wholesalers",
                "title_te": "ఇండియామార్ట్: చేపల మేత సరఫరాదారులు",
                "title_hi": "इंडियामार्ट: फ्लोटिंग फिश फीड आपूर्तिकर्ता",
                "title_tel": "IndiaMART: Fish Feed Suppliers",
                "badge_en": "Verified Sellers",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "Verified Sellers",
                "url": "https://dir.indiamart.com/search.mp?ss=floating+fish+feed",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            }
        ],
        "market_links": [
            {
                "title_en": "NFDB Aqua Bazar (Govt of India Trading Hub)",
                "title_te": "NFDB ఆక్వా బజార్ (కేంద్ర ప్రభుత్వ మార్కెట్ పోర్టల్)",
                "title_hi": "एनएफडीबी एक्वा बाज़ार (सरकारी व्यापार मंच)",
                "title_tel": "NFDB Aqua Bazar (Govt Market Portal)",
                "badge_en": "Govt Portal",
                "badge_te": "ప్రభుత్వ పోర్టల్",
                "badge_hi": "सरकारी पोर्टल",
                "badge_tel": "Govt Portal",
                "url": "https://aquabazar.nfdb.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-building-columns"
            },
            {
                "title_en": "National Fisheries Development Board (NFDB)",
                "title_te": "జాతీయ మత్స్య అభివృద్ధి మండలి (NFDB)",
                "title_hi": "राष्ट्रीय मत्स्य विकास बोर्ड (NFDB)",
                "title_tel": "National Fisheries Development Board",
                "badge_en": "Official",
                "badge_te": "అధికారికం",
                "badge_hi": "आधिकारिक",
                "badge_tel": "Official",
                "url": "https://nfdb.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-landmark"
            }
        ],
        "voice_script_te": "చేప చాలా ఆరోగ్యంగా ఉంది. ఎలాంటి వ్యాధులు లేవు. మార్కెట్లో గ్రేడ్-ఏ ధర లభిస్తుంది.",
        "voice_script_hi": "मछली पूरी तरह से स्वस्थ है। कोई बीमारी नहीं है। बाजार में ग्रेड-ए दरें मिलेंगी।",
        "voice_script_en": "The fish is completely healthy with premium Grade-A market quality."
    },
    "Bacterial Diseases": {
        "is_healthy": False,
        "species": "fish",
        "title_en": "Bacterial Diseases (Ulcers, Dropsy & Fin Rot)",
        "title_te": "బాక్టీరియల్ వ్యాధి (పుండ్లు, ఉబ్బిన పొట్ట, తోక కుళ్ళు)",
        "title_hi": "बैक्टीरियल रोग (घाव, फूला हुआ पेट, पूंछ सड़न)",
        "title_tel": "Bacterial Rogam (Pundlu, Toka Kullu)",
        "severity": "High",
        "severity_color": "#ef4444",
        "symptoms_en": "Red hemorrhagic ulcers on body, fraying fin margins, abdominal dropsy swelling, pop-eye.",
        "symptoms_te": "శరీరంపై ఎర్రటి రక్తపు పుండ్లు, తోక మరియు రెక్కలు కుళ్ళిపోవడం, ఉబ్బిన కళ్ళు.",
        "symptoms_hi": "शरीर पर लाल रक्तस्त्रावी घाव, पूंछ और पंख गलना, फूला हुआ पेट, उभरी हुई आंखें।",
        "symptoms_tel": "Shareeram meeda erra pundlu, toka kullu mariyu pop-eye lakshanalu.",
        "medicine_en": "💊 MANDHULU (MEDICINES & TREATMENT):\n1. Pond Water Sanitizer: Apply Potassium Permanganate (KMnO4 99%) @ 2.5 - 3.0 ppm OR BKC 80% @ 1 Liter per acre to disinfect pond water.\n2. Medicated Feed Antibiotic: Mix Oxytetracycline (OTC 50% Feed Grade) @ 65 mg per kg fish body weight for 7 consecutive days.\n3. Binder Application: Use egg white or commercial feed binder gel to coat the antibiotic onto pellets, shade dry for 20 minutes before broadcasting.",
        "medicine_te": "💊 మందులు & రసాయన చికిత్స (MANDHULU):\n1. నీటి చికిత్స: ఎకరాకు పొటాషియం పర్మాంగనేట్ (KMnO4) 2.5 ppm లేదా BKC 80% 1 లీటరు నీటిలో చల్లండి.\n2. మేతలో మందు: ఆక్సిటెట్రాసైక్లిన్ (OTC 50%) కేజీ చేపల బరువుకు 65 mg చొప్పున 7 రోజులు మేతలో కలిపి ఇవ్వండి.\n3. బైండర్ పద్ధతి: కోడిగుడ్డు తెల్లసొన లేదా బైండర్ జెల్ ఉపయోగించి మందును మేతకు పట్టించి, 20 నిమిషాలు నీడలో ఆరబెట్టి చెరువులో వేయండి.",
        "medicine_hi": "💊 दवाइयां और उपचार (MANDHULU):\n1. जल उपचार: प्रति एकड़ 2.5 ppm पोटैशियम परमैंगनेट (KMnO4) या 1 लीटर BKC 80% का छिड़काव करें।\n2. आहार एंटीबायोटिक: ऑक्सीटेट्रासाइक्लिन (OTC 50%) 65 मिलीग्राम प्रति किलो मछली वजन के हिसाब से 7 दिन चारे में मिलाकर दें।\n3. बाइंडर प्रयोग: अंडे की सफेदी या बाइंडर जेल से दवा को चारे पर चिपकाएं और छांव में सुखाकर डालें।",
        "medicine_tel": "💊 MANDHULU (MEDICINES):\n1. Neellalo Potassium Permanganate 2.5 ppm leda BKC 1L/acre veyandi.\n2. Meethalo OTC powder 65mg per kg fish weight 7 rojulu ivvandi.\n3. Guddu thella sonatho kalipi needalo aarabetti ivvandi.",
        "medicine_links": [
            {
                "title_en": "AquaCare Store: Potassium Permanganate (KMnO4 99%)",
                "title_te": "AquaCare స్టోర్: పొటాషియం పర్మాంగనేట్ (KMnO4 99%)",
                "title_hi": "एक्वाकेयर स्टोर: पोटैशियम परमैंगनेट (KMnO4 99%)",
                "title_tel": "AquaCare Store: Potassium Permanganate (KMnO4 99%)",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:med_kmno4",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "AquaCare Store: Oxytetracycline (OTC 50% Feed Grade)",
                "title_te": "AquaCare స్టోర్: ఆక్సిటెట్రాసైక్లిన్ (OTC 50% యాంటీబయాటిక్)",
                "title_hi": "एक्वाकेयर स्टोर: ऑक्सीटेट्रासाइक्लिन (OTC 50%)",
                "title_tel": "AquaCare Store: Oxytetracycline (OTC 50%)",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:med_otc",
                "type": "store",
                "icon": "fa-solid fa-pills"
            },
            {
                "title_en": "AquaCare Store: BKC 80% Aquaculture Sanitizer",
                "title_te": "AquaCare స్టోర్: BKC 80% శానిటైజర్",
                "title_hi": "एक्वाकेयर स्टोर: BKC 80% सैनिटाइज़र",
                "title_tel": "AquaCare Store: BKC 80% Sanitizer",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:med_bkc",
                "type": "store",
                "icon": "fa-solid fa-shield-virus"
            },
            {
                "title_en": "IndiaMART: Oxytetracycline 50% Feed Grade Sellers",
                "title_te": "ఇండియామార్ట్: ఆక్సిటెట్రాసైక్లిన్ 50% సరఫరాదారులు",
                "title_hi": "इंडियामार्ट: ऑक्सीटेट्रासाइक्लिन 50% विक्रेता",
                "title_tel": "IndiaMART: OTC 50% Feed Grade Sellers",
                "badge_en": "IndiaMART",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "IndiaMART",
                "url": "https://dir.indiamart.com/search.mp?ss=oxytetracycline+veterinary+powder",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            },
            {
                "title_en": "IndiaMART: Potassium Permanganate Crystals",
                "title_te": "ఇండియామార్ట్: పొటాషియం పర్మాంగనేట్ డీలర్లు",
                "title_hi": "इंडियामार्ट: पोटैशियम परमैंगनेट विक्रेता",
                "title_tel": "IndiaMART: KMnO4 Crystals Dealers",
                "badge_en": "IndiaMART",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "IndiaMART",
                "url": "https://dir.indiamart.com/search.mp?ss=potassium+permanganate",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. Immediate Feed Reduction: CUT DAILY FEED BY 40% IMMEDIATELY. Infected fish have weak digestion; unconsumed decaying feed causes toxic ammonia spikes.\n2. Healing Supplements: Mix Vitamin C (5g/kg feed) and Liver Tonic (10ml/kg feed) to boost tissue repair and regenerate damaged skin/fins.\n3. Tray Monitoring: Inspect feed check-trays 2 hours post broadcasting. If feed remains, cut by another 10%.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. మేతను తగ్గించండి: మేతను వెంటనే 40% తగ్గించండి! జబ్బు పడ్డ చేపలు తక్కువ తింటాయి; మిగిలిన మేత కుళ్ళిపోయి విషపూరిత అమ్మోనియా పెరుగుతుంది.\n2. పోషక సప్లిమెంట్స్: గాయాలు, పుండ్లు త్వరగా మానడానికి మేతలో విటమిన్-సి (5g/kg) మరియు లివర్ టానిక్ (10ml/kg) కలిపి ఇవ్వండి.\n3. ట్రేల తనిఖీ: మేత వేసిన 2 గంటల తర్వాత ఫీడ్ ట్రేలు పరిశీలించి, మిగిలిపోతే మరింత తగ్గించండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. आहार में कटौती: दैनिक आहार तुरंत 40% कम करें। बीमार मछलियां कम खाती हैं; बचा चारा सड़कर अमोनिया बढ़ाता है।\n2. उपचारात्मक पोषण: घाव भरने के लिए चारे में विटामिन-सी (5 ग्राम/किग्रा) और लिवर टॉनिक (10 मिली/किग्रा) मिलाएं।\n3. ट्रे निगरानी: चारा डालने के 2 घंटे बाद फीड ट्रे की जाँच करें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meethanu ventane 40% thagginchandi. Rogam unna chepalu feed thinalevu.\n2. Vitamin C 5g per kg mariyu Liver tonic 10ml kalipi ivvandi.\n3. 2 hours tarvatha trays check cheyandi.",
        "food_links": [
            {
                "title_en": "AquaCare Store: Floating Fish Pellets (40kg bag)",
                "title_te": "AquaCare స్టోర్: ఫ్లోటింగ్ చేపల మేత (40kg సంచి)",
                "title_hi": "एक्वाकेयर स्टोर: फ्लोटिंग मछली चारा",
                "title_tel": "AquaCare Store: Floating Fish Pellets",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:feed_fish",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "IndiaMART: Vitamin C Coated (Feed Grade Immunity)",
                "title_te": "ఇండియామార్ట్: కోటెడ్ విటమిన్-సి (మేత సప్లిమెంట్)",
                "title_hi": "इंडियामार्ट: कोटेड विटामिन-सी फीड ग्रेड",
                "title_tel": "IndiaMART: Coated Vitamin C Feed Grade",
                "badge_en": "Immunity",
                "badge_te": "రోగనిరోధకత",
                "badge_hi": "इम्युनिटी",
                "badge_tel": "Immunity",
                "url": "https://dir.indiamart.com/search.mp?ss=coated+vitamin+c+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-wheat-awn"
            },
            {
                "title_en": "IndiaMART: Aquaculture Liver Tonic & Stress Care",
                "title_te": "ఇండియామార్ట్: ఆక్వా లివర్ టానిక్ సరఫరాదారులు",
                "title_hi": "इंडियामार्ट: एक्वाकल्चर लिवर टॉनिक विक्रेता",
                "title_tel": "IndiaMART: Aqua Liver Tonic Suppliers",
                "badge_en": "Liver Care",
                "badge_te": "లివర్ కేర్",
                "badge_hi": "लिवर केयर",
                "badge_tel": "Liver Care",
                "url": "https://dir.indiamart.com/search.mp?ss=aquaculture+liver+tonic",
                "type": "external",
                "icon": "fa-solid fa-bottle-droplet"
            }
        ],
        "voice_script_te": "చేపలకు బాక్టీరియల్ తోక కుళ్ళు లేదా పుండ్ల వ్యాధి సోకింది. నీటిలో పొటాషియం పర్మాంగనేట్ వేయండి. మేతను 40 శాతం తగ్గించి, మేతలో ఆక్సిటెట్రాసైక్లిన్ మరియు విటమిన్ సి కలిపి ఏడు రోజులు ఇవ్వండి.",
        "voice_script_hi": "मछलियों में बैक्टीरियल संक्रमण पाया गया है। पानी में पोटैशियम परमैंगनेट डालें, चारा 40% कम करें और 7 दिन ऑक्सीटेट्रासाइक्लिन दें।",
        "voice_script_en": "Bacterial infection detected. Apply potassium permanganate to pond, reduce feed by 40%, and administer OTC with Vitamin C."
    },
    "Parasitic Diseases": {
        "is_healthy": False,
        "species": "fish",
        "title_en": "Parasitic Diseases (Argulus Fish Lice / Lernaea Anchor Worm)",
        "title_te": "పరాన్నజీవుల వ్యాధి (పేలు - ఆర్గులస్ / యాంకర్ వార్మ్స్)",
        "title_hi": "परजीवी रोग (मछली की जूं - आर्गुलस / एंकर वर्म)",
        "title_tel": "Parasitic Rogam (Chepala Pelu - Argulus)",
        "severity": "Moderate",
        "severity_color": "#f59e0b",
        "symptoms_en": "Visible transparent lice disks clinging to skin, flashing/rubbing against pond dykes, excess mucus.",
        "symptoms_te": "చేప శరీరంపై గుండ్రని పేలు కనిపించడం, దురదతో ఒడ్డుకు రాపిడి చేసుకోవడం.",
        "symptoms_hi": "त्वचा पर चिपकी हुई पारदर्शी जूं, खुजली के कारण किनारों से रगड़ना, अत्यधिक बलगम।",
        "symptoms_tel": "Chepala meeda pelu antukoni undadam, chepalu odduki rasi eedadam.",
        "medicine_en": "💊 MANDHULU (MEDICINES & TREATMENT):\n1. Parasite Eradication: Apply Emamectin Benzoate 0.2% @ 350-400 ml per acre OR Deltamethrin @ 100 ml per acre evenly across the pond.\n2. MANDATORY Second Dose: Repeat single dose after exactly 14 days to kill freshly hatched larval cysts.\n3. Continuous Aeration: Run all aerators during and after chemical treatment to prevent dissolved oxygen dips.",
        "medicine_te": "💊 మందులు & రసాయన చికిత్స (MANDHULU):\n1. పరాన్నజీవుల నివారణ: ఎకరాకు ఎమామెక్టిన్ బెంజోయేట్ 0.2% 350-400 ml లేదా డెల్టామెత్రిన్ 100 ml నీటిలో కలపండి.\n2. తప్పనిసరి రెండో డోస్: గుడ్ల నుండి వచ్చే పిల్ల పురుగుల కోసం 14 రోజుల తర్వాత ఖచ్చితంగా రెండో డోస్ వేయండి.\n3. ఎరేషన్: మందు వేసిన తర్వాత చెరువులో ఆక్సిజన్ తగ్గకుండా ఎరేటర్లను తప్పనిసరిగా నడపండి.",
        "medicine_hi": "💊 दवाइयां और उपचार (MANDHULU):\n1. परजीवी नियंत्रण: एमामेक्टिन बेंजोएट 0.2% 350-400 मिली प्रति एकड़ या डेल्टामेथ्रिन 100 मिली प्रति एकड़ तालाब में डालें।\n2. दूसरी खुराक: अंडों से निकलने वाले कीड़ों को मारने के लिए 14 दिन बाद दूसरी खुराक अवश्य दें।\n3. एरेटर संचालन: दवा डालने के बाद एरेटर लगातार चलाएं।",
        "medicine_tel": "💊 MANDHULU (MEDICINES):\n1. Emamectin Benzoate 350 ml/acre leda Deltamethrin 100 ml veyandi.\n2. 14 rojula tharwatha malli compulsory second dose veyandi.\n3. Aerators on chesi unchandi.",
        "medicine_links": [
            {
                "title_en": "IndiaMART: Emamectin Benzoate 0.2% Aqua Liquid",
                "title_te": "ఇండియామార్ట్: ఎమామెక్టిన్ బెంజోయేట్ 0.2% (పేల నివారణ)",
                "title_hi": "इंडियामार्ट: एमामेक्टिन बेंजोएट 0.2% एक्वा लिक्विड",
                "title_tel": "IndiaMART: Emamectin Benzoate 0.2% Aqua",
                "badge_en": "Anti-Parasitic",
                "badge_te": "పేల నివారణ",
                "badge_hi": "परजीवी नाशक",
                "badge_tel": "Anti-Parasite",
                "url": "https://dir.indiamart.com/search.mp?ss=emamectin+benzoate+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-bug-slash"
            },
            {
                "title_en": "IndiaMART: Deltamethrin Aquaculture Grade Solution",
                "title_te": "ఇండియామార్ట్: డెల్టామెత్రిన్ ఆక్వా ద్రావణం",
                "title_hi": "इंडियामार्ट: डेल्टामेथ्रिन एक्वाकल्चर ग्रेड",
                "title_tel": "IndiaMART: Deltamethrin Aqua Grade",
                "badge_en": "IndiaMART",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "IndiaMART",
                "url": "https://dir.indiamart.com/search.mp?ss=deltamethrin+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            },
            {
                "title_en": "AquaCare Store: Potassium Permanganate (Antiseptic Dip)",
                "title_te": "AquaCare స్టోర్: పొటాషియం పర్మాంగనేట్ (KMnO4)",
                "title_hi": "एक्वाकेयर स्टोर: पोटैशियम परमैंगनेट",
                "title_tel": "AquaCare Store: Potassium Permanganate",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_kmno4",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "AquaCare Store: Soil & Water Probiotics (Complex)",
                "title_te": "AquaCare స్టోర్: సాయిల్ & వాటర్ ప్రోబయోటిక్స్",
                "title_hi": "एक्वाकेयर स्टोर: सॉइल व वाटर प्रोबायोटिक्स",
                "title_tel": "AquaCare Store: Soil & Water Probiotics",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_probiotic",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. Feeding Volume: Maintain normal feeding volume, but split into 3-4 smaller meals daily.\n2. Anti-Stress Nutrition: Add Liver Tonic (10 ml/kg feed) and Multivitamins to relieve irritation and epidermal stress.\n3. Probiotics: Broadcast soil probiotics 3 days post-chemical to restore healthy pond microbes.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. మేత పరిమాణం: మేత సాధారణంగా ఇవ్వండి, అయితే రోజుకు 3-4 సార్లు విభజించి వేయండి.\n2. ఒత్తిడి నివారణ: దురద, ఒత్తిడి తగ్గడానికి మేతలో లివర్ టానిక్ (10 ml/kg) మరియు మల్టీవిటమిన్లు కలిపి ఇవ్వండి.\n3. ప్రోబయోటిక్స్: మందు వేసిన 3 రోజుల తర్వాత మట్టి ప్రోబయోటిక్స్ వేయండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. चारा मात्रा: सामान्य मात्रा दें, लेकिन दिन में 3-4 बार में बांटकर खिलाएं।\n2. तनाव मुक्ति: खुजली और तनाव कम करने के लिए लिवर टॉनिक (10 मिली/किग्रा) और मल्टीविटामिन मिलाएं।\n3. प्रोबायोटिक्स: दवा के 3 दिन बाद तालाब में प्रोबायोटिक्स डालें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Normal feed quantity ivvandi, 3-4 times divide chesi veyandi.\n2. Liver tonic 10ml mariyu Multivitamins meethalo kalipi ivvandi.\n3. Feed trays clean ga unchandi.",
        "food_links": [
            {
                "title_en": "AquaCare Store: Floating Fish Pellets (28% Protein)",
                "title_te": "AquaCare స్టోర్: ఫ్లోటింగ్ చేపల మేత",
                "title_hi": "एक्वाकेयर स्टोर: फ्लोटिंग मछली चारा",
                "title_tel": "AquaCare Store: Floating Fish Pellets",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:feed_fish",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "IndiaMART: Aquaculture Liver Tonic & Multivitamin",
                "title_te": "ఇండియామార్ట్: ఆక్వా లివర్ టానిక్ & మల్టీవిటమిన్",
                "title_hi": "इंडियामार्ट: एक्वा लिवर टॉनिक व मल्टीविटामिन",
                "title_tel": "IndiaMART: Liver Tonic & Multivitamin",
                "badge_en": "Recovery",
                "badge_te": "పునరుద్ధరణ",
                "badge_hi": "रिकवरी",
                "badge_tel": "Recovery",
                "url": "https://dir.indiamart.com/search.mp?ss=aquaculture+liver+tonic",
                "type": "external",
                "icon": "fa-solid fa-bottle-droplet"
            }
        ],
        "voice_script_te": "చేపలకు పేలు లేదా పరాన్నజీవుల వ్యాధి సోకింది. ఎమామెక్టిన్ బెంజోయేట్ వేయండి. పద్నాలుగు రోజుల తర్వాత రెండో డోస్ వేయండి. మేతలో లివర్ టానిక్ ఇవ్వండి.",
        "voice_script_hi": "मछलियों में परजीवी जूं का संक्रमण है। एमामेक्टिन बेंजोएट डालें और 14 दिन बाद दोबारा दोहराएं। चारे में लिवर टॉनिक दें।",
        "voice_script_en": "Parasitic infection detected. Apply emamectin benzoate, repeat in fourteen days, and supplement liver tonic in feed."
    },
    "White Tail Diseases": {
        "is_healthy": False,
        "species": "fish",
        "title_en": "White Tail Disease (MrNV / Muscle Necrosis)",
        "title_te": "తోక తెల్లబడే వ్యాధి (వైట్ టెయిల్ నెక్రోసిస్)",
        "title_hi": "सफेद पूंछ रोग (व्हाइट टेल नेक्रोसिस)",
        "title_tel": "White Tail Rogam (Toka Thellabadadam)",
        "severity": "High",
        "severity_color": "#ea580c",
        "symptoms_en": "Opaque milky white discoloration starting at caudal tail and progressing towards body, erratic swimming.",
        "symptoms_te": "తోక భాగం తెల్లగా పాల రంగులోకి మారడం, ఈతలో బ్యాలెన్స్ తప్పడం.",
        "symptoms_hi": "पूंछ का दूधिया सफेद होना, मांसपेशियों का कमजोर पड़ना, तैरने में असंतुलन।",
        "symptoms_tel": "Toka thellaga paala lanti color lo maradam, swimming balance thappadam.",
        "medicine_en": "💊 MANDHULU (MEDICINES & TREATMENT):\n1. Emergency Biosecurity: Immediately STOP drag netting or sampling to minimize physical shock.\n2. Pond Disinfection: Apply Iodine 20% complex @ 1 Liter per acre to curb viral & secondary bacterial load.\n3. Electrolyte Balancing: Apply Magnesium Sulphate @ 15 kg/acre and Calcium Chloride @ 20 kg/acre to restore osmotic muscle balance.",
        "medicine_te": "💊 మందులు & రసాయన చికిత్స (MANDHULU):\n1. చేపలకు ఒత్తిడి తగ్గించండి: వలలు లాగడం లేదా శాంప్లింగ్ చేయడం పూర్తిగా ఆపివేయండి.\n2. నీటి శుద్ధి: బ్యాక్టీరియా లోడ్ తగ్గించడానికి ఎకరాకు అయోడిన్ 20% 1 లీటరు వేయండి.\n3. ఖనిజ లవణాలు: ఎకరాకు మెగ్నీషియం సల్ఫేట్ 15 కేజీలు, కాల్షియం క్లోరైడ్ 20 కేజీలు వేసి కండరాల సమతుల్యతను కాపాడండి.",
        "medicine_hi": "💊 दवाइयां और उपचार (MANDHULU):\n1. तनाव नियंत्रण: मछलियों को झटके से बचाने के लिए जाल खींचना तुरंत रोकें।\n2. जल शुद्धिकरण: प्रति एकड़ 1 लीटर आयोडीन 20% कॉम्प्लेक्स डालें।\n3. मिनरल्स संतुलन: प्रति एकड़ मैग्नीशियम सल्फेट 15 किलो और कैल्शियम क्लोराइड 20 किलो डालें।",
        "medicine_tel": "💊 MANDHULU (MEDICINES):\n1. Valalu laagakandi, stress thagginchandi.\n2. Iodine 20% 1 liter per acre veyandi.\n3. Magnesium mariyu Calcium minerals cheruvulo veyandi.",
        "medicine_links": [
            {
                "title_en": "IndiaMART: Iodine 20% Aquaculture Sanitizer",
                "title_te": "ఇండియామార్ట్: అయోడిన్ 20% శానిటైజర్",
                "title_hi": "इंडियामार्ट: आयोडीन 20% एक्वा सैनिटाइज़र",
                "title_tel": "IndiaMART: Iodine 20% Aqua Sanitizer",
                "badge_en": "Disinfectant",
                "badge_te": "శానిటైజర్",
                "badge_hi": "कीटाणुनाशक",
                "badge_tel": "Sanitizer",
                "url": "https://dir.indiamart.com/search.mp?ss=iodine+20+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-shield-virus"
            },
            {
                "title_en": "IndiaMART: Magnesium & Calcium Aqua Minerals",
                "title_te": "ఇండియామార్ట్: మెగ్నీషియం & కాల్షియం ఆక్వా మినరల్స్",
                "title_hi": "इंडियामार्ट: मैग्नीशियम सल्फेट व कैल्शियम मिनरल्स",
                "title_tel": "IndiaMART: Magnesium & Calcium Minerals",
                "badge_en": "Electrolytes",
                "badge_te": "ఖనిజ లవణాలు",
                "badge_hi": "खनिज लवण",
                "badge_tel": "Minerals",
                "url": "https://dir.indiamart.com/search.mp?ss=aquaculture+minerals+magnesium+calcium",
                "type": "external",
                "icon": "fa-solid fa-gem"
            },
            {
                "title_en": "AquaCare Store: BKC 80% Sanitizer",
                "title_te": "AquaCare స్టోర్: BKC 80% శానిటైజర్",
                "title_hi": "एक्वाकेयर स्टोर: BKC 80% सैनिटाइज़र",
                "title_tel": "AquaCare Store: BKC 80% Sanitizer",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_bkc",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. Cut Feed by 30%: Lower daily feeding by 30% to ease metabolic burden.\n2. Muscle Regeneration: Mix Beta-Glucan (5 g/kg feed) and Vitamin E (3 g/kg feed) with binder for cellular muscle healing.\n3. Feed Timing: Broadcast feed during cool dawn and dusk hours only.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. మేతను 30% తగ్గించండి: జీర్ణక్రియపై భారం తగ్గించడానికి మేతను వెంటనే 30% తగ్గించండి.\n2. కండరాల రికవరీ: తోక కండరాలు పునరుద్ధరించబడటానికి మేతలో బీటా-గ్లూకాన్ (5g/kg) మరియు విటమిన్-ఈ (3g/kg) కలిపి ఇవ్వండి.\n3. చల్లని వేళల్లో మేత: కేవలం ఉదయం, సాయంత్రం వేళల్లోనే మేత వేయండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. चारा 30% घटाएं: पाचन पर भार कम करने के लिए दैनिक आहार तुरंत 30% कम करें।\n2. मांसपेशी पुनर्जनन: चारे में बीटा-ग्लूकन (5 ग्राम/किग्रा) और विटामिन-ई (3 ग्राम/किग्रा) मिलाएं।\n3. ठंडे समय पर आहार: केवल सुबह और शाम के ठंडे समय पर चारा डालें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meetha 30% ventane thagginchandi.\n2. Beta-glucan 5g mariyu Vitamin E 3g meethalo kalipi ivvandi.\n3. Challaga unnapude feed veyandi.",
        "food_links": [
            {
                "title_en": "IndiaMART: Beta-Glucan 70% Feed Supplement",
                "title_te": "ఇండియామార్ట్: బీటా-గ్లూకాన్ 70% (కండరాల రికవరీ)",
                "title_hi": "इंडियामार्ट: बीटा-ग्लूकन 70% फीड सप्लीमेंट",
                "title_tel": "IndiaMART: Beta-Glucan 70% Feed Grade",
                "badge_en": "Muscle Repair",
                "badge_te": "కండరాల బలం",
                "badge_hi": "मांसपेशी मरम्मत",
                "badge_tel": "Muscle Healing",
                "url": "https://dir.indiamart.com/search.mp?ss=beta+glucan+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-dumbbell"
            },
            {
                "title_en": "IndiaMART: Vitamin E 50% Feed Grade Powder",
                "title_te": "ఇండియామార్ట్: విటమిన్-ఈ 50% మేత పౌడర్",
                "title_hi": "इंडियामार्ट: विटामिन-ई 50% फीड पाउडर",
                "title_tel": "IndiaMART: Vitamin E 50% Feed Grade",
                "badge_en": "Tissue Care",
                "badge_te": "కణజాల రక్షణ",
                "badge_hi": "कोशिका सुरक्षा",
                "badge_tel": "Tissue Care",
                "url": "https://dir.indiamart.com/search.mp?ss=vitamin+e+feed+grade",
                "type": "external",
                "icon": "fa-solid fa-capsules"
            },
            {
                "title_en": "AquaCare Store: Floating Fish Feed (28% Protein)",
                "title_te": "AquaCare స్టోర్: ఫ్లోటింగ్ చేపల మేత",
                "title_hi": "एक्वाकेयर स्टोर: फ्लोटिंग मछली चारा",
                "title_tel": "AquaCare Store: Floating Fish Feed",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:feed_fish",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            }
        ],
        "voice_script_te": "చేపలకు వైట్ టెయిల్ వ్యాధి సోకింది. వలలు లాగవద్దు. నీటిలో అయోడిన్ మరియు మినరల్స్ వేయండి. మేతను 30 శాతం తగ్గించి బీటా గ్లూకాన్ ఇవ్వండి.",
        "voice_script_hi": "मछलियों में सफेद पूंछ रोग है। जाल न खींचे। पानी में आयोडीन और मिनरल्स डालें। चारा 30% कम करें।",
        "voice_script_en": "White tail disease detected. Halt netting, disinfect with iodine, dose minerals, and reduce feed by 30% with beta-glucan."
    },
    "Fungal Diseases": {
        "is_healthy": False,
        "species": "fish",
        "title_en": "Fungal Diseases (Saprolegnia / Cotton Wool)",
        "title_te": "శిలీంధ్ర వ్యాధి (దూది వంటి ఫంగస్ - సాప్రోలెగ్నియా)",
        "title_hi": "फंगल रोग (कपास जैसा फंगस - सैप्रोलिग्निया)",
        "title_tel": "Fungal Rogam (Doodi lanti patches)",
        "severity": "Moderate",
        "severity_color": "#f59e0b",
        "symptoms_en": "White/grey cotton-like tufts on body, fins, and eyes; slow lethargic movement near banks.",
        "symptoms_te": "శరీరం మరియు రెక్కలపై తెల్లటి దూది వంటి ఫంగస్ మచ్చలు రావడం, నీరసంగా తేలడం.",
        "symptoms_hi": "शरीर, पंखों और आंखों पर सफेद/भूरे कपास जैसे गुच्छे, किनारों पर सुस्त तैरना।",
        "symptoms_tel": "Shareeram meeda thellati doodi patches kanapadadam.",
        "medicine_en": "💊 MANDHULU (MEDICINES & TREATMENT):\n1. Pond Antifungal: Copper Sulphate (CuSO4) @ 0.5 - 0.8 ppm (Ensure water alkalinity > 60 ppm) OR Methylene Blue @ 2.0 ppm.\n2. Salt Bath Dip: 2% rock salt dip for 5-8 minutes in small holding tanks before releasing.\n3. Gentle Handling: Handle drag nets smoothly to prevent further skin abrasions and open wounds.",
        "medicine_te": "💊 మందులు & రసాయన చికిత్స (MANDHULU):\n1. ఫంగస్ చికిత్స: కాపర్ సల్ఫేట్ (0.5 - 0.8 ppm) లేదా మిథిలీన్ బ్లూ (2 ppm) నీటిలో కలపండి.\n2. ఉప్పు స్నానం: 2% ఉప్పు నీటి ద్రావణంలో చేపలను 5 నిమిషాలు ఉంచి వదలండి.\n3. గాయాలు కాకుండా జాగ్రత్త: వలలు లాగేటప్పుడు చేపల పొలుసులు ఊడిపోకుండా సున్నితంగా వ్యవహరించండి.",
        "medicine_hi": "💊 दवाइयां और उपचार (MANDHULU):\n1. फंगस नियंत्रण: कॉपर सल्फेट (0.5 - 0.8 ppm) या मेथिलीन ब्लू 2 ppm पानी में डालें।\n2. नमक स्नान: 2% नमक के घोल में मछलियों को 5 मिनट का स्नान दें।\n3. चोट से बचाव: जाल खींचते समय मछलियों की त्वचा को छिलने से बचाएं।",
        "medicine_tel": "💊 MANDHULU (MEDICINES):\n1. Copper sulphate 0.5 ppm leda Methylene blue 2 ppm veyandi.\n2. 2% uppu neellalo 5 minutes dip cheyandi.\n3. Chepalaki debbalu thagilinchakandi.",
        "medicine_links": [
            {
                "title_en": "AquaCare Store: Potassium Permanganate (KMnO4 99%)",
                "title_te": "AquaCare స్టోర్: పొటాషియం పర్మాంగనేట్ (KMnO4 99%)",
                "title_hi": "एक्वाकेयर स्टोर: पोटैशियम परमैंगनेट (KMnO4 99%)",
                "title_tel": "AquaCare Store: Potassium Permanganate",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_kmno4",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "IndiaMART: Copper Sulphate (CuSO4) Aqua Grade",
                "title_te": "ఇండియామార్ట్: కాపర్ సల్ఫేట్ (మైలుతుత్తం) ఆక్వా గ్రేడ్",
                "title_hi": "इंडियामार्ट: कॉपर सल्फेट एक्वा ग्रेड",
                "title_tel": "IndiaMART: Copper Sulphate Aqua Grade",
                "badge_en": "Antifungal",
                "badge_te": "ఫంగస్ నాశిని",
                "badge_hi": "एंटीफंगल",
                "badge_tel": "Antifungal",
                "url": "https://dir.indiamart.com/search.mp?ss=copper+sulphate+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-flask"
            },
            {
                "title_en": "IndiaMART: Methylene Blue Pure Aquaculture Grade",
                "title_te": "ఇండియామార్ట్: మిథిలీన్ బ్లూ ఫంగస్ చికిత్స",
                "title_hi": "इंडियामार्ट: मेथिलीन ब्लू एक्वाकल्चर ग्रेड",
                "title_tel": "IndiaMART: Methylene Blue Aqua Grade",
                "badge_en": "Fungal Cure",
                "badge_te": "ఫంగస్ చికిత్స",
                "badge_hi": "फंगस उपचार",
                "badge_tel": "Fungal Cure",
                "url": "https://dir.indiamart.com/search.mp?ss=methylene+blue+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-droplet"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. Maintain Feeding with Skin Repair Supplements: Add Vitamin E (2 g/kg feed) and Spirulina for epidermal tissue healing.\n2. Clean Feed Trays: Scrub feed trays daily to eliminate adhering fungal hyphae.\n3. Water Temperature: Fungal outbreaks peak during cold weather; monitor pond temperature.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. చర్మం రికవరీ: చర్మం, పొలుసులు త్వరగా కోలుకోవడానికి మేతలో విటమిన్-ఈ (2 g/kg) మరియు స్పిరులినా కలిపి ఇవ్వండి.\n2. ఫీడ్ ట్రేల పరిశుభ్రత: ఫంగస్ వ్యాపించకుండా ఫీడ్ ట్రేలను రోజూ శుభ్రం చేయండి.\n3. చలికాలపు జాగ్రత్తలు: చల్లటి వాతావరణంలో ఫంగస్ వేగంగా పెరుగుతుంది కాబట్టి ఉష్ణోగ్రతను గమనించండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. त्वचा रिकवरी: चारे में विटामिन-ई (2 ग्राम/किग्रा) और स्पिरुलिना मिलाकर दें।\n2. फीड ट्रे स्वच्छता: फंगल बीजाणुओं को रोकने के लिए ट्रे रोजाना साफ करें।\n3. तापमान ध्यान: ठंड में फंगस बढ़ता है, पानी का तापमान नियमित जांचें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meethalo Vitamin E 2g mariyu Spirulina kalipi ivvandi.\n2. Feed trays roju clean cheyandi.\n3. Challadanaani batti feed adjust cheyandi.",
        "food_links": [
            {
                "title_en": "IndiaMART: Spirulina Powder Feed Grade (Skin Immunity)",
                "title_te": "ఇండియామార్ట్: స్పిరులినా పౌడర్ (చర్మ సంరక్షణ మేత)",
                "title_hi": "इंडियामार्ट: स्पिरुलिना पाउडर फीड ग्रेड",
                "title_tel": "IndiaMART: Spirulina Powder Feed Grade",
                "badge_en": "Skin Immunity",
                "badge_te": "చర్మ రక్షణ",
                "badge_hi": "त्वचा सुरक्षा",
                "badge_tel": "Skin Care",
                "url": "https://dir.indiamart.com/search.mp?ss=spirulina+feed+grade",
                "type": "external",
                "icon": "fa-solid fa-seedling"
            },
            {
                "title_en": "AquaCare Store: Floating Fish Pellets (28% Protein)",
                "title_te": "AquaCare స్టోర్: ఫ్లోటింగ్ చేపల మేత",
                "title_hi": "एक्वाकेयर स्टोर: फ्लोटिंग मछली चारा",
                "title_tel": "AquaCare Store: Floating Fish Pellets",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:feed_fish",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            }
        ],
        "voice_script_te": "చేపలకు ఫంగస్ వ్యాధి సోకింది. కాపర్ సల్ఫేట్ లేదా మిథిలీన్ బ్లూ వేయండి. మేతలో విటమిన్ ఈ కలిపి ఇవ్వండి.",
        "voice_script_hi": "मछलियों में फंगस का संक्रमण है। पानी में कॉपर सल्फेट डालें और चारे में विटामिन ई दें।",
        "voice_script_en": "Fungal infection detected. Treat with copper sulphate or methylene blue, and add Vitamin E to feed."
    },
    "Healthy": {
        "is_healthy": True,
        "species": "prawn",
        "title_en": "Healthy Prawn (Grade-A Export Quality)",
        "title_te": "ఆరోగ్యకరమైన రొయ్య (గ్రేడ్-ఏ ఎగుమతి నాణ్యత)",
        "title_hi": "स्वस्थ झींगा (ग्रेड-ए निर्यात गुणवत्ता)",
        "title_tel": "Arogyakaramaina Royya (Grade-A Export Quality)",
        "severity": "Safe",
        "severity_color": "#10b981",
        "symptoms_en": "Translucent cuticle, full dark digestive gut tract, active jumping reflex, clean translucent gills.",
        "symptoms_te": "రొయ్య శరీరం పారదర్శకంగా ఉంది, మేత పేగు నిండుగా ఉంది, చురుకైన ఈత ఉంది.",
        "symptoms_hi": "पारदर्शी शरीर, भरी हुई आहार नली, तेज उछलने की क्षमता, साफ गलफड़े।",
        "symptoms_tel": "Royya body transparent ga undi, gut line full ga undi, active jumping undi.",
        "market_status_en": "Export Grade Vannamei Shrimp. Zero melanosis, high count weight.",
        "market_status_te": "ఎగుమతి ప్రీమియం గ్రేడ్ వనామి రొయ్యలు. అధిక ధర మరియు గరిష్ట లాభం లభిస్తుంది.",
        "market_status_hi": "निर्यात ग्रेड वेनामेई झींगा। कोई कालापन नहीं, उत्कृष्ट वजन और उच्चतम दरें।",
        "market_status_tel": "Export Premium Grade Vannamei. Manchi rate mariyu high profit vasthundi.",
        "selling_rates": MARKET_SELLING_RATES["prawn"],
        "harvest_guide_en": "1. Check molt cycle: Ensure <5% soft-shell before drag netting.\n2. Night chilling harvest: Use minimum 1 kg ice per 1 kg shrimp to avoid black spot melanosis.",
        "harvest_guide_te": "1. కుబుసం రాలిన వెంటనే పట్టవద్దు (మెత్తటి రొయ్య 5% కంటే తక్కువ ఉండాలి).\n2. రాత్రి చల్లగా ఉన్నప్పుడు 1:1 ఐస్ నిష్పత్తితో హార్వెస్ట్ చేయండి.",
        "harvest_guide_hi": "1. मोल्टिंग चेक करें (नरम झींगा 5% से कम होना चाहिए)।\n2. रात में ठंडे तापमान में 1:1 बर्फ के अनुपात के साथ कटाई करें।",
        "harvest_guide_tel": "1. Molt cycle check cheyandi, soft shell thakkuva undali.\n2. Ice slurry lo harvest chesi export rate pondandi.",
        "food_guide_en": "1. Standard Feeding: Maintain standard feeding by body weight count.\n2. Feed Trays: Inspect check-trays 2 hours post broadcast to ensure optimum FCR.\n3. Protein: Use 35% protein water-stable extruded shrimp feed.",
        "food_guide_te": "1. కౌంట్ మేత: రొయ్యల కౌంట్ ఆధారంగా సాధారణ మేత కొనసాగించండి.\n2. చెక్ ట్రేలు: మేత వేసిన 2 గంటల తర్వాత చెక్ ట్రేలు తనిఖీ చేయండి.\n3. నాణ్యత: 35% ప్రోటీన్ ఉన్న నాణ్యమైన మేత వాడండి.",
        "food_guide_hi": "1. सामान्य आहार: काउंट के अनुसार संतुलित आहार जारी रखें।\n2. चेक ट्रे: 2 घंटे बाद चेक ट्रे की जाँच करें।\n3. गुणवत्ता: 35% प्रोटीन युक्त पानी में स्थिर रहने वाला चारा दें।",
        "food_guide_tel": "1. Count batti normal feed schedule continue cheyandi.\n2. Check trays roju 2 hours tarvatha chudandi.\n3. Quality 35% protein feed veyandi.",
        "food_links": [
            {
                "title_en": "AquaCare Store: Vannamei Shrimp Grower Feed (35% Protein, 25kg)",
                "title_te": "AquaCare స్టోర్: వనామి రొయ్యల మేత (35% ప్రోటీన్)",
                "title_hi": "एक्वाकेयर स्टोर: वेनामेई झींगा चारा (35% प्रोटीन)",
                "title_tel": "AquaCare Store: Vannamei Shrimp Feed (35% Protein)",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "स्टोर ऑर्डर",
                "badge_tel": "Direct Order",
                "url": "internal:feed_prawn",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "Avanti Feeds Official Portal (India's No.1 Shrimp Feed)",
                "title_te": "అవంతి ఫీడ్స్ అధికారిక పోర్టల్ (రొయ్యల మేత దిగ్గజం)",
                "title_hi": "अवंती फीड्स आधिकारिक पोर्टल (शीर्ष झींगा आहार)",
                "title_tel": "Avanti Feeds Official Website",
                "badge_en": "Avanti Feeds",
                "badge_te": "టాప్ బ్రాండ్",
                "badge_hi": "प्रमुख ब्रांड",
                "badge_tel": "Top Brand",
                "url": "https://avantifeeds.com/",
                "type": "external",
                "icon": "fa-solid fa-wheat-awn"
            },
            {
                "title_en": "CP Aquaculture India (Shrimp Feed & Hatchery Products)",
                "title_te": "సీపీ ఆక్వాకల్చర్ ఇండియా (రొయ్యల మేత సరఫరా)",
                "title_hi": "सीपी एक्वाकल्चर इंडिया (झींगा आहार)",
                "title_tel": "CP Aquaculture India Website",
                "badge_en": "CP Aqua",
                "badge_te": "గ్లోబల్ లీడర్",
                "badge_hi": "वैश्विक लीडर",
                "badge_tel": "Global Leader",
                "url": "https://www.cp-india.com/",
                "type": "external",
                "icon": "fa-solid fa-globe"
            },
            {
                "title_en": "IndiaMART: Vannamei Shrimp Feed Suppliers",
                "title_te": "ఇండియామార్ట్: వనామి రొయ్యల మేత డీలర్లు",
                "title_hi": "इंडियामार्ट: वेनामेई झींगा आहार विक्रेता",
                "title_tel": "IndiaMART: Vannamei Feed Dealers",
                "badge_en": "Verified",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "Verified",
                "url": "https://dir.indiamart.com/search.mp?ss=vannamei+shrimp+feed",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            }
        ],
        "market_links": [
            {
                "title_en": "e-SANTA Electronic Aquaculture Trading Portal (Govt of India)",
                "title_te": "ఈ-శాంత (e-SANTA): కేంద్ర ప్రభుత్వ రొయ్యల ట్రేడింగ్ పోర్టల్",
                "title_hi": "ई-सांता (e-SANTA): भारत सरकार झींगा व्यापार मंच",
                "title_tel": "e-SANTA Govt Shrimp Trading Portal",
                "badge_en": "Govt Direct Trade",
                "badge_te": "కేంద్ర ప్రభుత్వ పోర్టల్",
                "badge_hi": "सरकारी मंच",
                "badge_tel": "Govt Portal",
                "url": "https://esanta.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-building-columns"
            },
            {
                "title_en": "MPEDA Marine Products Export Development Authority",
                "title_te": "MPEDA సముద్ర ఉత్పత్తుల ఎగుమతి అభివృద్ధి సంస్థ",
                "title_hi": "एमपीईडीए (MPEDA) समुद्री उत्पाद निर्यात प्राधिकरण",
                "title_tel": "MPEDA Official Marine Export Portal",
                "badge_en": "Export Govt",
                "badge_te": "ఎగుమతి మండలి",
                "badge_hi": "निर्यात बोर्ड",
                "badge_tel": "Export Govt",
                "url": "https://mpeda.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-landmark"
            },
            {
                "title_en": "NFDB Aqua Bazar National Portal",
                "title_te": "NFDB ఆక్వా బజార్ మార్కెట్ హబ్",
                "title_hi": "एनएफडीबी एक्वा बाज़ार राष्ट्रीय पोर्टल",
                "title_tel": "NFDB Aqua Bazar Market Hub",
                "badge_en": "Aqua Bazar",
                "badge_te": "మత్స్య మార్కెట్",
                "badge_hi": "मत्स्य बाज़ार",
                "badge_tel": "Aqua Bazar",
                "url": "https://aquabazar.nfdb.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-store"
            }
        ],
        "voice_script_te": "రొయ్యలు చాలా ఆరోగ్యంగా ఉన్నాయి. ఎగుమతి నాణ్యతతో మంచి లాభాలు వస్తాయి.",
        "voice_script_hi": "झींगा पूरी तरह से स्वस्थ है और निर्यात ग्रेड गुणवत्ता का है।",
        "voice_script_en": "Prawns are healthy and of export grade quality."
    },
    "WSSV": {
        "is_healthy": False,
        "species": "prawn",
        "title_en": "White Spot Syndrome Virus (WSSV)",
        "title_te": "తెల్ల మచ్చ వైరస్ (వైట్ స్పాట్ సిండ్రోమ్ - WSSV)",
        "title_hi": "सफेद धब्बा वायरस (WSSV - व्हाइट स्पॉट सिंड्रोम)",
        "title_tel": "Thella Macha Virus (WSSV)",
        "severity": "Critical",
        "severity_color": "#dc2626",
        "symptoms_en": "White circular spots (0.5 - 2 mm) inside carapace, reddish body color, rapid mortality.",
        "symptoms_te": "తల చిప్ప లోపల తెల్లటి గుండ్రని మచ్చలు, రొయ్య శరీరం ఎర్రబారడం, భారీగా చనిపోవడం.",
        "symptoms_hi": "सिर के खोल के अंदर सफेद गोल धब्बे, शरीर का लाल होना, अचानक भारी मृत्यु दर।",
        "symptoms_tel": "Carapace meeda thellati round spots, body red color avvadam, heavy mortality.",
        "medicine_en": "💊 MANDHULU (CRITICAL VIRAL EMERGENCY ACTION):\n1. HARVEST IMMEDIATELY: If prawns are harvestable size (> 14g), HARVEST TODAY! Do not delay a single session!\n2. NO CHEMICAL CURE: WSSV is an incurable virus. Do NOT waste money on unscientific antibiotics or fake chemicals.\n3. BIOSECURITY LOCKDOWN: Close pond sluice gates tightly; DO NOT discharge infected water into public canals.\n4. Pond Sterilization: Apply Bleaching powder @ 100 kg/acre (30 ppm chlorine) post-harvest to decontaminate pond.",
        "medicine_te": "💊 మందులు & అత్యవసర చర్యలు (MANDHULU):\n1. వెంటనే హార్వెస్ట్ చేయండి: రొయ్యలు 14 గ్రాములు దాటి ఉంటే వెంటనే పట్టేయండి! ఒక్క పూట కూడా ఆలస్యం చేయవద్దు!\n2. ఎలాంటి మందులు పనిచేయవు: ఈ వైరస్ కు రసాయన మందులు లేదా యాంటీబయాటిక్స్ లేవు. మోసపూరిత మందులు వాడవద్దు.\n3. క్వారంటైన్: తూములు గట్టిగా మూసివేయండి, పంట కాలువల్లోకి నీటిని అస్సలు వదలకండి.\n4. బ్లీచింగ్: పంట తీసిన తర్వాత ఎకరాకు 100 కేజీల బ్లీచింగ్ పౌడర్ వేసి చెరువును శుద్ధి చేయండి.",
        "medicine_hi": "💊 दवाइयां और आपातकालीन कार्रवाई (MANDHULU):\n1. तुरंत कटाई (हार्वेस्ट) करें: यदि झींगा बिक्री योग्य (> 14 ग्राम) है, तो तुरंत हार्वेस्ट करें! एक सत्र भी देरी न करें!\n2. कोई रासायनिक इलाज नहीं: इस वायरस का कोई इलाज नहीं है; फर्जी एंटीबायोटिक्स पर पैसे बर्बाद न करें।\n3. क्वारंटाइन: पानी के गेट बंद रखें; नहरों में पानी बिल्कुल न छोड़ें।\n4. ब्लीचिंग शुद्धिकरण: कटाई के बाद प्रति एकड़ 100 किलो ब्लीचिंग पाउडर डालकर तालाब को कीटाणुरहित करें।",
        "medicine_tel": "💊 MANDHULU (CRITICAL EMERGENCY):\n1. Royyalu 14 grams paina unte VENTANE HARVEST cheyandi! Alasyam cheyoddu!\n2. Ee virus ki chemicals levu, fake mandulu vadoddu.\n3. Thumulu moosi quarantine cheyandi.\n4. Bleaching powder 100 kg/acre vesi sanitize cheyandi.",
        "medicine_links": [
            {
                "title_en": "IndiaMART: Bleaching Powder 33% Stable Chlorine",
                "title_te": "ఇండియామార్ట్: బ్లీచింగ్ పౌడర్ 33% (చెరువు శానిటైజేషన్)",
                "title_hi": "इंडियामार्ट: ब्लीचिंग पाउडर 33% क्लोरीन (तालाब विसंक्रमण)",
                "title_tel": "IndiaMART: Bleaching Powder 33% Chlorine",
                "badge_en": "Sterilization",
                "badge_te": "చెరువు శుద్ధి",
                "badge_hi": "विसंक्रमण",
                "badge_tel": "Sterilization",
                "url": "https://dir.indiamart.com/search.mp?ss=bleaching+powder+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-spray-can-sparkles"
            },
            {
                "title_en": "AquaCare Store: BKC 80% High Potency Sanitizer",
                "title_te": "AquaCare స్టోర్: BKC 80% శానిటైజర్",
                "title_hi": "एक्वाकेयर स्टोर: BKC 80% सैनिटाइज़र",
                "title_tel": "AquaCare Store: BKC 80% Sanitizer",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_bkc",
                "type": "store",
                "icon": "fa-solid fa-shield-virus"
            },
            {
                "title_en": "MPEDA Aqua Disease Diagnostic Labs & PCR Centers",
                "title_te": "MPEDA ఆక్వా వ్యాధి నిర్ధారణ ల్యాబ్స్ & PCR కేంద్రాలు",
                "title_hi": "एमपीईडीए (MPEDA) पीसीआर टेस्टिंग व रोग निदान केंद्र",
                "title_tel": "MPEDA PCR Testing & Disease Labs",
                "badge_en": "Govt Diagnostic",
                "badge_te": "ప్రభుత్వ ల్యాబ్",
                "badge_hi": "सरकारी लैब",
                "badge_tel": "Govt Lab",
                "url": "https://mpeda.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-microscope"
            },
            {
                "title_en": "e-SANTA: Emergency Shrimp Harvest Trade Portal",
                "title_te": "ఈ-శాంత (e-SANTA): అత్యవసర రొయ్యల మార్కెట్ అమ్మకాలు",
                "title_hi": "ई-सांता: आपातकालीन झींगा बिक्री मंच",
                "title_tel": "e-SANTA Emergency Shrimp Harvest Sale",
                "badge_en": "Instant Harvest Sale",
                "badge_te": "తక్షణ అమ్మకం",
                "badge_hi": "त्वरित बिक्री",
                "badge_tel": "Emergency Sale",
                "url": "https://esanta.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-truck-fast"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. CUT FEED BY 50-70% IMMEDIATELY: Infected prawns completely cease feeding. Excess rotting pellets cause lethal ammonia and hydrogen sulfide spikes.\n2. Continuous Aeration: Run aerators 24/7 at maximum capacity to sustain oxygen.\n3. Cease Feeding Before Harvest: Stop all feed 24 hours prior to emergency drag netting.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. మేతను 50-70% తగ్గించండి: జబ్బు పడ్డ రొయ్యలు మేత తినవు; అడుగున కుళ్ళిన మేత వల్ల అమ్మోనియా పెరిగి నిమిషాల్లో చనిపోతాయి.\n2. 24 గంటల ఎరేషన్: ఆక్సిజన్ పడిపోకుండా ఎరేటర్లను 24 గంటలు నిరంతరం నడపండి.\n3. హార్వెస్ట్ ముందు మేత ఆపండి: వల లాగడానికి 24 గంటల ముందు మేత పూర్తిగా ఆపివేయండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. चारा 50-70% तुरंत कम करें: संक्रमित झींगे चारा नहीं खाते; सड़ने वाला चारा जहरीला अमोनिया पैदा करता है।\n2. 24 घंटे एरेटर: ऑक्सीजन बनाए रखने के लिए एरेटर लगातार चलाएं।\n3. कटाई से पहले चारा रोकें: हार्वेस्टिंग से 24 घंटे पहले चारा पूरी तरह बंद कर दें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meetha ventane 50% nundi 70% thagginchandi. Royyalu meetha thinalevu.\n2. Aerators 24 hours continuous ga on cheyandi.\n3. Harvest mundu meetha poorthiga aapeyandi.",
        "food_links": [
            {
                "title_en": "Avanti Feeds Emergency Health Advisory",
                "title_te": "అవంతి ఫీడ్స్ అత్యవసర ఆరోగ్య సలహాలు (మేత తగ్గింపు)",
                "title_hi": "अवंती फीड्स आपातकालीन आहार सलाह",
                "title_tel": "Avanti Feeds Emergency Feeding Advisory",
                "badge_en": "Feed Advisory",
                "badge_te": "మేత సలహాలు",
                "badge_hi": "आहार सलाह",
                "badge_tel": "Feed Advice",
                "url": "https://avantifeeds.com/",
                "type": "external",
                "icon": "fa-solid fa-triangle-exclamation"
            },
            {
                "title_en": "AquaCare Store: Vannamei Shrimp Feed",
                "title_te": "AquaCare స్టోర్: వనామి రొయ్యల మేత",
                "title_hi": "एक्वाकेयर स्टोर: वेनामेई झींगा चारा",
                "title_tel": "AquaCare Store: Vannamei Shrimp Feed",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:feed_prawn",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            }
        ],
        "voice_script_te": "అత్యవసర హెచ్చరిక: రొయ్యలకు తెల్ల మచ్చ వైరస్ సోకింది. రొయ్యలు మార్కెట్ సైజు ఉంటే వెంటనే పట్టేయండి. మేతను సగానికి పైగా తగ్గించి ఎరేటర్లు నడపండి.",
        "voice_script_hi": "आपातकालीन चेतावनी: झींगा में सफेद धब्बा वायरस पाया गया है। यदि आकार उपयुक्त है तो तुरंत हार्वेस्ट करें और चारा 50% से अधिक कम करें।",
        "voice_script_en": "Critical emergency: White spot virus detected. Immediate harvest is recommended if shrimp are marketable. Cut feed by 60%."
    },
    "Black_Gill": {
        "is_healthy": False,
        "species": "prawn",
        "title_en": "Black Gill Disease (Sludge Fouling / Melanization)",
        "title_te": "నల్ల మొప్ప వ్యాధి (బ్లాక్ గిల్ - అడుగు వ్యర్థాల సంక్రమణ)",
        "title_hi": "काली गलफड़ बीमारी (ब्लैक गिल - तल की गंदगी)",
        "title_tel": "Nalla Moppa Rogam (Black Gill)",
        "severity": "High",
        "severity_color": "#ea580c",
        "symptoms_en": "Gill filaments turned brown or pitch black, prawns gasping near aerators, reduced feed intake.",
        "symptoms_te": "రొయ్య మొప్పలు నల్లగా మారడం, శ్వాస ఆడక ఎరేటర్ల వద్దకు రావడం, ఫీడ్ ట్రేలలో మేత మిగిలిపోవడం.",
        "symptoms_hi": "गलफड़े भूरे या काले पड़ जाना, एरेटर के पास सांस के लिए आना, चारा कम खाना।",
        "symptoms_tel": "Royya moppalu nallaga maradam, shwasa aagaka aerators daggara theladam.",
        "medicine_en": "💊 MANDHULU (MEDICINES & TREATMENT):\n1. Pond Bottom Sludge Detox: Broadcast Zeolite Granules @ 25-30 kg per acre AND Yucca Extract @ 500 ml per acre to absorb toxic hydrogen sulfide (H2S) and ammonia.\n2. Soil Probiotics: Apply high-count Bacillus & Nitrosomonas pond bottom probiotics @ 1 kg per acre to digest black sludge.\n3. Water Exchange: If creek conditions are safe, perform 20-25% clean water exchange.",
        "medicine_te": "💊 మందులు & రసాయన చికిత్స (MANDHULU):\n1. చెరువు అడుగు శుద్ధి: ఎకరాకు 25-30 కేజీల జియోలైట్ గుళికలు మరియు 500 ml యుక్కా లిక్విడ్ వేసి విషవాయువులను తొలగించండి.\n2. మట్టి ప్రోబయోటిక్స్: అడుగున పేరుకుపోయిన మురికిని కరిగించడానికి బాసిల్లస్ సాయిల్ ప్రోబయోటిక్స్ 1 kg/ఎకరా వేయండి.\n3. నీటి మార్పిడి: కాలువ నీరు బాగుంటే 20-25% మంచి నీటి మార్పిడి చేయండి.",
        "medicine_hi": "💊 दवाइयां और उपचार (MANDHULU):\n1. तल की गंदगी डिटॉक्स: प्रति एकड़ 25-30 किलो जियोलाइट और 500 मिली युक्का अर्क डालें।\n2. प्रोबायोटिक्स: तल के अपशिष्ट को पचाने के लिए 1 किग्रा/एकड़ बैसिलस सॉइल प्रोबायोटिक्स डालें।\n3. जल विनिमय: यदि संभव हो तो 20% स्वच्छ जल का विनिमय करें।",
        "medicine_tel": "💊 MANDHULU (MEDICINES):\n1. Zeolite 30 kg/acre mariyu Yucca extract 500 ml veyandi.\n2. Soil probiotics 1 kg/acre vesi adugu chetta clean cheyandi.\n3. Veelaithe 20% fresh neellu change cheyandi.",
        "medicine_links": [
            {
                "title_en": "AquaCare Store: Zeolite Granules + Yucca Extract",
                "title_te": "AquaCare స్టోర్: జియోలైట్ గుళికలు + యుక్కా లిక్విడ్",
                "title_hi": "एक्वाकेयर स्टोर: जियोलाइट दाने + युक्का लिक्विड",
                "title_tel": "AquaCare Store: Zeolite Granules + Yucca Extract",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_zeolite",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "AquaCare Store: Soil & Water Probiotics (Sludge Digester)",
                "title_te": "AquaCare స్టోర్: సాయిల్ & వాటర్ ప్రోబయోటిక్స్",
                "title_hi": "एक्वाकेयर स्टोर: सॉइल प्रोबायोटिक्स (कचरा घोलक)",
                "title_tel": "AquaCare Store: Soil Probiotics",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_probiotic",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "IndiaMART: Yucca Schidigera Extract 100% Aqua",
                "title_te": "ఇండియామార్ట్: యుక్కా ఎక్స్‌ట్రాక్ట్ (అమ్మోనియా తొలగింపు)",
                "title_hi": "इंडियामार्ट: युक्का अर्क 100% (अमोनिया हटाने वाला)",
                "title_tel": "IndiaMART: Yucca Extract Aqua Suppliers",
                "badge_en": "Ammonia Absorber",
                "badge_te": "అమ్మోనియా నివారిణి",
                "badge_hi": "अमोनिया नियंत्रक",
                "badge_tel": "Ammonia Cure",
                "url": "https://dir.indiamart.com/search.mp?ss=yucca+extract+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            },
            {
                "title_en": "IndiaMART: Aquaculture Zeolite Powder & Granules",
                "title_te": "ఇండియామార్ట్: ఆక్వా జియోలైట్ గుళికల డీలర్లు",
                "title_hi": "इंडियामार्ट: एक्वाकल्चर जियोलाइट विक्रेता",
                "title_tel": "IndiaMART: Zeolite Powder & Granules",
                "badge_en": "IndiaMART",
                "badge_te": "ధృవీకరించబడింది",
                "badge_hi": "सत्यापित",
                "badge_tel": "IndiaMART",
                "url": "https://dir.indiamart.com/search.mp?ss=zeolite+powder+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-arrow-up-right-from-square"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT (FOOD EM CHEYALO):\n1. CUT FEED BY 35% FOR 4 CONSECUTIVE DAYS: Excessive feed accumulation on the pond bottom is the primary cause of black gill.\n2. Medicated Feed for Gill Repair: Mix Vitamin C (5 g/kg feed) + Liver Tonic (10 ml/kg feed) with binder gel for 5 days to facilitate gill regeneration.\n3. Check Feed Trays: Inspect trays 2 hours post broadcasting; do not overfeed.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO):\n1. మేతను 35% తగ్గించండి: అడుగున పేరుకుపోయిన మేత వల్లే మొప్పలు నల్లబడతాయి; కనుక 4 రోజుల పాటు మేతను 35% తగ్గించండి.\n2. మేతలో మందు: మొప్పలు త్వరగా మానడానికి మేతలో విటమిన్-సి (5 g/kg) మరియు లివర్ టానిక్ (10 ml/kg) కలిపి 5 రోజులు ఇవ్వండి.\n3. ట్రేల పరిశీలన: 2 గంటల తర్వాత ఫీడ్ ట్రేలు తనిఖీ చేయండి.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन (FOOD EM CHEYALO):\n1. चारा 35% कम करें: तालाब के तल में अधिक चारा सड़ने से ही ब्लैक गिल होता है; अतः 4 दिन चारा 35% कम रखें।\n2. गलफड़ों का उपचार: चारे में विटामिन-सी (5 ग्राम/किग्रा) और लिवर टॉनिक (10 मिली/किग्रा) बाइंडर के साथ मिलाकर 5 दिन दें।\n3. ट्रे जाँच: 2 घंटे बाद फीड ट्रे अवश्य देखें।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meetha 4 rojulu 35% ventane thagginchandi.\n2. Vitamin C 5g mariyu Liver tonic 10ml meethalo kalipi 5 rojulu ivvandi.\n3. Feed trays check cheyandi.",
        "food_links": [
            {
                "title_en": "AquaCare Store: Vannamei Shrimp Grower Feed (35% Protein)",
                "title_te": "AquaCare స్టోర్: వనామి రొయ్యల మేత (35% ప్రోటీన్)",
                "title_hi": "एक्वाकेयर स्टोर: वेनामेई झींगा चारा (35% प्रोटीन)",
                "title_tel": "AquaCare Store: Vannamei Shrimp Feed",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:feed_prawn",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "IndiaMART: Coated Vitamin C (Gill Tissue Repair)",
                "title_te": "ఇండియామార్ట్: కోటెడ్ విటమిన్-సి (మొప్పల రికవరీ)",
                "title_hi": "इंडियामार्ट: कोटेड विटामिन-सी (गलफड़ उपचार)",
                "title_tel": "IndiaMART: Coated Vitamin C for Gills",
                "badge_en": "Gill Repair",
                "badge_te": "మొప్పల రికవరీ",
                "badge_hi": "गलफड़ रिकवरी",
                "badge_tel": "Gill Healing",
                "url": "https://dir.indiamart.com/search.mp?ss=coated+vitamin+c+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-capsules"
            },
            {
                "title_en": "IndiaMART: Aquaculture Feed Binder Gel",
                "title_te": "ఇండియామార్ట్: ఆక్వాఫీడ్ బైండర్ జెల్ డీలర్లు",
                "title_hi": "इंडियामार्ट: एक्वाफीड बाइंडर जेल विक्रेता",
                "title_tel": "IndiaMART: Aqua Feed Binder Gel",
                "badge_en": "Feed Binder",
                "badge_te": "బైండర్ జెల్",
                "badge_hi": "बाइंडर जेल",
                "badge_tel": "Binder Gel",
                "url": "https://dir.indiamart.com/search.mp?ss=aquaculture+feed+binder+gel",
                "type": "external",
                "icon": "fa-solid fa-link"
            },
            {
                "title_en": "Avanti Feeds: Water-Stable Shrimp Nutrition",
                "title_te": "అవంతి ఫీడ్స్: రొయ్యల పోషకాహారం",
                "title_hi": "अवंती फीड्स: झींगा पोषण आहार",
                "title_tel": "Avanti Feeds Official Portal",
                "badge_en": "Avanti",
                "badge_te": "అధికారికం",
                "badge_hi": "आधिकारिक",
                "badge_tel": "Official",
                "url": "https://avantifeeds.com/",
                "type": "external",
                "icon": "fa-solid fa-wheat-awn"
            }
        ],
        "voice_script_te": "రొయ్యలకు నల్ల మొప్ప వ్యాధి సోకింది. చెరువు అడుగున జియోలైట్ మరియు ప్రోబయోటిక్స్ వేయండి. మేతను 35 శాతం తగ్గించి విటమిన్ సి ఇవ్వండి.",
        "voice_script_hi": "झींगा में ब्लैक गिल बीमारी है। तालाब के तल पर जियोलाइट और प्रोबायोटिक्स डालें, चारा 35% कम करें और विटामिन सी दें।",
        "voice_script_en": "Black gill disease detected. Apply zeolite and soil probiotics to pond bottom, reduce feed by 35%, and supplement Vitamin C."
    },
    "WSSV_Black_Gill": {
        "is_healthy": False,
        "species": "prawn",
        "title_en": "Co-infection: WSSV & Black Gill Disease",
        "title_te": "ద్వంద్వ సంక్రమణ: తెల్ల మచ్చ & నల్ల మొప్ప వ్యాధి",
        "title_hi": "दोहरा संक्रमण: सफेद धब्बा और काली गलफड़",
        "title_tel": "Double Infection: Thella Macha & Nalla Moppa",
        "severity": "Critical",
        "severity_color": "#b91c1c",
        "symptoms_en": "Simultaneous presence of white spots on carapace and pitch-black gills. Extreme mortality risk.",
        "symptoms_te": "తలపై తెల్ల మచ్చలు మరియు మొప్పలు నల్లగా మారడం రెండూ కలిసి ఉన్నాయి. 24 గంటల్లో భారీగా చనిపోయే ప్రమాదం ఉంది.",
        "symptoms_hi": "सिर पर सफेद धब्बे और काले गलफड़े दोनों एक साथ। 24 घंटे में अत्यधिक मृत्यु का खतरा।",
        "symptoms_tel": "Thella machalu mariyu nalla moppa rendoo unnayi. Heavy mortality ravachu.",
        "medicine_en": "💊 MANDHULU (EMERGENCY HARVEST):\n1. HARVEST IMMEDIATELY: Harvest the entire pond stock immediately without delay.\n2. BIOSECURITY: Seal pond drainage completely; do not pollute natural water bodies.\n3. AERATION: Run aerators 100% until netting is finalized.",
        "medicine_te": "💊 మందులు & అత్యవసర చర్యలు (MANDHULU):\n1. తక్షణ అత్యవసర హార్వెస్ట్: రొయ్యలను వెంటనే పట్టేసి మార్కెట్ కు తరలించండి.\n2. క్వారంటైన్: చెరువు నీటిని కాలువలోకి వదలకండి.\n3. ఎరేషన్: హార్వెస్ట్ పూర్తయ్యే వరకు ఎరేటర్లను 100% నడపండి.",
        "medicine_hi": "💊 दवाइयां और आपातकालीन कार्रवाई (MANDHULU):\n1. तत्काल कटाई: तुरंत सभी झींगे निकालें और बेचें।\n2. क्वारंटाइन: तालाब का पानी बाहर न जाने दें।\n3. एरेटर: कटाई पूरी होने तक एरेटर 100% चलाएं।",
        "medicine_tel": "💊 MANDHULU:\n1. Royyalanu ventane harvest cheyandi.\n2. Kaluva neellalo neellu vadaloddu.\n3. Aerators full speed lo run cheyandi.",
        "medicine_links": [
            {
                "title_en": "IndiaMART: Bleaching Powder 33% Chlorine (Pond Disinfection)",
                "title_te": "ఇండియామార్ట్: బ్లీచింగ్ పౌడర్ 33% క్లోరిన్",
                "title_hi": "इंडियामार्ट: ब्लीचिंग पाउडर 33% क्लोरीन",
                "title_tel": "IndiaMART: Bleaching Powder 33% Chlorine",
                "badge_en": "Emergency Disinfection",
                "badge_te": "అత్యవసర శుద్ధి",
                "badge_hi": "आपातकालीन विसंक्रमण",
                "badge_tel": "Emergency Sterilization",
                "url": "https://dir.indiamart.com/search.mp?ss=bleaching+powder+aquaculture",
                "type": "external",
                "icon": "fa-solid fa-spray-can-sparkles"
            },
            {
                "title_en": "AquaCare Store: Zeolite Granules + Yucca Extract",
                "title_te": "AquaCare స్టోర్: జియోలైట్ గుళికలు + యుక్కా",
                "title_hi": "एक्वाकेयर स्टोर: जियोलाइट दाने + युक्का",
                "title_tel": "AquaCare Store: Zeolite + Yucca",
                "badge_en": "In-App Store",
                "badge_te": "డైరెక్ట్ ఆర్డర్",
                "badge_hi": "స్టోర్ ఆర్డర్",
                "badge_tel": "Direct Order",
                "url": "internal:med_zeolite",
                "type": "store",
                "icon": "fa-solid fa-cart-shopping"
            },
            {
                "title_en": "e-SANTA: Emergency Shrimp Harvest Trade Portal",
                "title_te": "ఈ-శాంత (e-SANTA): తక్షణ రొయ్యల విక్రయం",
                "title_hi": "ई-सांता: आपातकालीन झींगा बिक्री मंच",
                "title_tel": "e-SANTA Emergency Harvest Portal",
                "badge_en": "Emergency Trade",
                "badge_te": "తక్షణ అమ్మకం",
                "badge_hi": "आपात बिक्री",
                "badge_tel": "Instant Sale",
                "url": "https://esanta.gov.in/",
                "type": "govt",
                "icon": "fa-solid fa-truck-fast"
            }
        ],
        "food_guide_en": "🌾 FOOD & FEEDING MANAGEMENT:\n1. STOP FEEDING COMPLETELY (0% FEED).\n2. Feeding sick shrimp will only worsen bottom toxic gases.",
        "food_guide_te": "🌾 ఆహారం & మేత యాజమాన్యం:\n1. మేత వేయడం పూర్తిగా ఆపివేయండి (0% మేత).\n2. జబ్బు పడ్డ రొయ్యలకు మేత వేస్తే అడుగున విషవాయువులు పెరిగి మరింత నష్టం జరుగుతుంది.",
        "food_guide_hi": "🌾 आहार और चारा प्रबंधन:\n1. चारा देना पूरी तरह बंद करें (0% आहार)।\n2. बीमार झींगों को चारा डालने से तल में जहरीली गैसें बढ़ेंगी।",
        "food_guide_tel": "🌾 FOOD GUIDELINES:\n1. Meetha veyadam poorthiga aapeyandi (0% feed).\n2. Feed trays clean ga unchandi.",
        "food_links": [
            {
                "title_en": "Avanti Feeds Emergency Feed Guidelines (Halt Feeding)",
                "title_te": "అవంతి ఫీడ్స్ అత్యవసర గైడ్‌లైన్స్ (మేత నిలిపివేత)",
                "title_hi": "अवंती फीड्स आपातकालीन दिशानिर्देश (चारा रोकें)",
                "title_tel": "Avanti Feeds Emergency Guidelines",
                "badge_en": "0% Feed Notice",
                "badge_te": "మేత ఆపండి",
                "badge_hi": "चारा रोकें",
                "badge_tel": "0% Feed Notice",
                "url": "https://avantifeeds.com/",
                "type": "external",
                "icon": "fa-solid fa-circle-exclamation"
            }
        ],
        "voice_script_te": "అత్యవసర ప్రమాదం: తెల్ల మచ్చ మరియు నల్ల మొప్ప రెండూ సోకాయి. మేత పూర్తిగా ఆపి వెంటనే రొయ్యలను పట్టేయండి.",
        "voice_script_hi": "अत्यधिक आपातकाल: सफेद धब्बा और ब्लैक गिल दोनों मौजूद हैं। चारा तुरंत रोकें और कटाई करें।",
        "voice_script_en": "Extreme emergency: Simultaneous White Spot and Black Gill detected. Cease feeding and harvest immediately."
    }
}

# ==============================================================================
# 4. ROBUST OBJECT DETECTION ENGINE & STRICT NEGATIVE REJECTION
# ==============================================================================
NON_AQUATIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear"
}

def check_non_aquatic_object(image_path: str) -> Optional[str]:
    """Check if image clearly depicts a human selfie, room, or everyday non-aquatic object."""
    if COCO_DETECTOR is None:
        return None
    try:
        results = COCO_DETECTOR.predict(source=image_path, conf=0.30, imgsz=480, max_det=5, verbose=False)
        for r in results or []:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(r, "names", {}) or getattr(COCO_DETECTOR, "names", {}) or {}
            orig_shape = getattr(r, "orig_shape", (640, 640))
            total_area = float(orig_shape[0] * orig_shape[1]) if orig_shape else 409600.0
            for box in boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                label = names.get(cls_id, "") if isinstance(names, dict) else ""
                label_lower = str(label).strip().lower()
                
                # Calculate bounding box area percentage
                try:
                    xyxy = box.xyxy.cpu().numpy().flatten().tolist()[:4]
                except Exception:
                    xyxy = box.xyxy[0].tolist()
                box_area = max(0.0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
                area_pct = (box_area / max(1.0, total_area)) * 100.0

                # Strict priority on person (e.g. selfie / front camera) when person occupies significant view
                if label_lower == "person" and score >= 0.55 and area_pct >= 40.0:
                    return "person"
                if label_lower in NON_AQUATIC_CLASSES and score >= 0.65 and area_pct >= 45.0:
                    return label_lower
    except Exception as exc:
        print(f"[VALIDATE] Non-aquatic check error: {exc}")
    return None

def _normalized_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")

def _canonical_class_name(model, class_map: Dict[int, str], cls_id: int) -> Optional[str]:
    """Map ONLY a real model label to our disease classes. Never guess by class ID."""
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        raw_name = names.get(cls_id, "") or names.get(str(cls_id), "")
    elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        raw_name = names[cls_id]
    else:
        raw_name = ""

    raw_norm = _normalized_label(raw_name)
    if not raw_norm:
        return None

    # Remove common numeric prefixes used by exported YOLO names such as "0 healthy".
    raw_norm = re.sub(r"^\d+_?", "", raw_norm)

    aliases = {
        "Healthy Fish": {
            "healthy_fish", "healthy_fish_body", "fish_healthy", "healthyfish", "healthy"
        },
        "Bacterial Diseases": {
            "bacterial_disease", "bacterial_diseases", "bacterial", "bacterial_ulcer",
            "hemorrhagic_ulcer", "ulcer", "fin_rot", "dropsy"
        },
        "Parasitic Diseases": {
            "parasitic_disease", "parasitic_diseases", "parasitic", "parasite",
            "argulus", "fish_lice", "anchor_worm", "lernaea"
        },
        "White Tail Diseases": {
            "white_tail", "white_tail_disease", "white_tail_diseases", "whitetail",
            "muscle_necrosis", "mrnv", "white_tail_necrosis"
        },
        "Fungal Diseases": {
            "fungal_disease", "fungal_diseases", "fungal", "saprolegnia", "cotton_wool"
        },
        "Healthy": {
            "healthy", "healthy_prawn", "healthy_prawn_specimen", "prawn_healthy",
            "healthy_shrimp", "shrimp_healthy"
        },
        "WSSV": {
            "wssv", "white_spot", "white_spot_syndrome", "white_spot_disease", "whitespot"
        },
        "Black_Gill": {
            "black_gill", "blackgill", "black_gill_disease", "black_gill_prawn"
        },
        "WSSV_Black_Gill": {
            "wssv_black_gill", "wssv_blackgill", "white_spot_black_gill"
        },
    }

    for friendly, nameset in aliases.items():
        if raw_norm in nameset:
            if friendly in class_map.values():
                return friendly

    # Controlled substring matching for common YOLO labels, still based on the label text.
    ordered_keywords = [
        ("wssv_black_gill", "WSSV_Black_Gill"),
        ("white_spot", "WSSV"),
        ("black_gill", "Black_Gill"),
        ("white_tail", "White Tail Diseases"),
        ("bacterial", "Bacterial Diseases"),
        ("parasit", "Parasitic Diseases"),
        ("fung", "Fungal Diseases"),
    ]
    for key, friendly in ordered_keywords:
        if key in raw_norm and friendly in class_map.values():
            return friendly

    if "healthy" in raw_norm and "fish" in raw_norm and "Healthy Fish" in class_map.values():
        return "Healthy Fish"
    if "healthy" in raw_norm and "Healthy" in class_map.values():
        return "Healthy"

    # Some exported checkpoints lose their class names and expose numeric labels.
    # When the class count exactly matches our known project order, use that order.
    try:
        mn = getattr(model, "names", {}) or {}
        count = len(mn) if isinstance(mn, (dict, list, tuple)) else 0
        if count == len(class_map) and (raw_norm == "" or raw_norm.isdigit()) and cls_id in class_map:
            return class_map[cls_id]
    except Exception:
        pass

    return None


def _get_raw_max_conf(model, image_path: str, conf: float = 0.10, imgsz: int = 640) -> float:
    """Return the highest raw YOLO box confidence from ANY class — no name filtering.

    Used ONLY for species-identity gating (cross-species guard) where we want
    to know if a model fires on the image at all, regardless of class name.
    Never use this for disease diagnosis — it ignores class semantics.
    """
    if model is None:
        return 0.0
    try:
        results = model.predict(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            max_det=20,
            augment=False,
            verbose=False,
        )
        best = 0.0
        for r in results or []:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    c = float(box.conf.item())
                    if c > best:
                        best = c
                except Exception:
                    continue
        return best
    except Exception:
        return 0.0

def _model_inference_imgsz(model, default: int = 640) -> int:
    """Use the image size saved with a YOLO checkpoint when it is available."""
    overrides = getattr(model, "overrides", {}) or {}
    value = overrides.get("imgsz", default) if isinstance(overrides, dict) else default
    if isinstance(value, (list, tuple)):
        value = max(value, default=default)
    try:
        return max(32, int(value))
    except (TypeError, ValueError):
        return default

def _collect_yolo_boxes(model, image_path: str, class_map: Dict[int, str], conf: float = 0.40, imgsz: int = 1024, allow_retry: bool = False):
    """Collect only the selected model's real class boxes.

    IMPORTANT:
    - The coordinates come directly from the YOLO detector.
    - No whole-fish box is substituted for a disease box.
    - Strictly filters detections by the requested confidence threshold.
    """
    if model is None:
        return []

    try:
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=0.45,
            imgsz=imgsz,
            max_det=50,
            augment=False,
            verbose=False
        )
    except Exception as exc:
        print(f"[DETECT] YOLO inference failed: {type(exc).__name__}: {exc}")
        return []

    try:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 640, 640

    out = []
    min_required_conf = conf  # allow_retry kept for API compatibility but retried flag removed
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}

        for box in boxes:
            try:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())

                if score < min_required_conf:
                    continue

                canonical_name = _canonical_class_name(model, class_map, cls_id)
                if canonical_name is None:
                    continue
                if canonical_name not in class_map.values():
                    continue

                if isinstance(names, dict):
                    raw_label = str(names.get(cls_id, canonical_name)).strip()
                elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                    raw_label = str(names[cls_id]).strip()
                else:
                    raw_label = canonical_name

                try:
                    xyxy_raw = box.xyxy.cpu().numpy().flatten().tolist()
                    xyxy = [float(v) for v in xyxy_raw[:4]]
                except Exception:
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                x1 = max(0.0, min(float(img_w), xyxy[0]))
                y1 = max(0.0, min(float(img_h), xyxy[1]))
                x2 = max(0.0, min(float(img_w), xyxy[2]))
                y2 = max(0.0, min(float(img_h), xyxy[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                bw = x2 - x1
                bh = y2 - y1
                if bw < 6 or bh < 6:
                    continue

                area_pct = (bw * bh) / max(1.0, img_w * img_h) * 100.0

                out.append({
                    "class_id": cls_id,
                    "class_name": canonical_name,
                    "raw_label": raw_label,
                    "confidence": score,
                    "confidence_pct": round(score * 100.0, 1),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "area_percent": round(area_pct, 2),
                })
            except Exception as exc:
                print(f"[BOX] Box parse error: {type(exc).__name__}: {exc}")
                continue

    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out[:30]


def _species_gate_features(image_path: str, fish_boxes: List[Dict[str, Any]], prawn_boxes: List[Dict[str, Any]]):
    """Build the fixed feature vector used by the trained Fish/Prawn gate."""
    try:
        with Image.open(image_path) as im:
            image_area = max(1, im.width * im.height)
    except Exception as exc:
        raise ValueError(f"Cannot read image dimensions for species verification: {exc}") from exc

    features = []
    for boxes, class_map in ((fish_boxes, FISH_CLASSES), (prawn_boxes, PRAWN_CLASSES)):
        for class_id in sorted(class_map):
            class_name = class_map[class_id]
            features.append(max(
                (
                    float(detection.get("confidence", 0.0))
                    for detection in boxes
                    if detection.get("class_name") == class_name
                ),
                default=0.0,
            ))
        strongest = max(
            boxes,
            key=lambda detection: float(detection.get("confidence", 0.0)),
            default=None,
        )
        if strongest is None:
            features.extend((0.0, 0.0))
        else:
            bbox = strongest.get("bbox") or []
            if len(bbox) != 4:
                raise ValueError("Species model returned an invalid bounding box.")
            box_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            features.extend((
                float(strongest.get("confidence", 0.0)),
                box_area / image_area,
            ))

    if len(features) != 13:
        raise ValueError(f"Species gate expected 13 features, received {len(features)}.")
    return np.asarray([features], dtype=np.float32)


def _location_for_box(box, width: int, species: str):
    """Describe where the actual YOLO disease box sits in the image.

    Without a pose/orientation model, this must not claim an anatomical head/tail
    position when the fish/prawn may be rotated. The UI therefore reports image
    position while the bounding box itself shows the exact model ROI coordinates.
    """
    cx = ((float(box[0]) + float(box[2])) / 2.0) / max(1.0, float(width))
    if cx < 0.33:
        side_en, side_te, side_hi, side_tel = "left side", "ఎడమ వైపు", "बाईं ओर", "left side"
    elif cx > 0.67:
        side_en, side_te, side_hi, side_tel = "right side", "కుడి వైపు", "दाईं ओर", "right side"
    else:
        side_en, side_te, side_hi, side_tel = "center", "మధ్య భాగం", "बीच", "center"
    return {
        "en": f"Disease box: {side_en} of image",
        "te": f"రోగం బాక్స్: చిత్రంలో {side_te}",
        "hi": f"रोग बॉक्स: चित्र के {side_hi}",
        "tel": f"Disease box: {side_tel}"
    }

def _draw_label(draw, x, y, text, fill, font, image_w):
    """Draw a readable label without letting it run outside the image."""
    if not text:
        return
    fs = max(13, int(image_w / 45))
    try:
        if font is None:
            tb_w = int(len(text) * fs * 0.62) + 16
            tb_h = fs + 10
        else:
            bbox = draw.textbbox((0, 0), text, font=font)
            tb_w = (bbox[2] - bbox[0]) + 14
            tb_h = (bbox[3] - bbox[1]) + 10
    except Exception:
        tb_w = int(len(text) * fs * 0.62) + 16
        tb_h = fs + 10

    left = max(0, min(int(x), max(0, image_w - tb_w)))
    top = max(0, int(y) - tb_h)
    draw.rounded_rectangle(
        [left, top, left + tb_w, top + tb_h],
        radius=5,
        fill=fill
    )
    try:
        draw.text((left + 7, top + 4), text, fill="white", font=font)
    except Exception:
        draw.text((left + 7, top + 4), text, fill="white")


def _draw_corner_box(draw, bbox, outline, width, corner_color="#ffffff"):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

    corner_len = max(10, int(min(max(1, x2 - x1), max(1, y2 - y1)) * 0.12))
    cw = max(2, width + 1)

    segments = [
        ((x1, y1), (x1 + corner_len, y1)),
        ((x1, y1), (x1, y1 + corner_len)),
        ((x2, y1), (x2 - corner_len, y1)),
        ((x2, y1), (x2, y1 + corner_len)),
        ((x1, y2), (x1 + corner_len, y2)),
        ((x1, y2), (x1, y2 - corner_len)),
        ((x2, y2), (x2 - corner_len, y2)),
        ((x2, y2), (x2, y2 - corner_len)),
    ]
    for p1, p2 in segments:
        draw.line([p1, p2], fill=corner_color, width=cw)


def draw_detection_result(
    image_path: str,
    species_boxes: Optional[List[Dict[str, Any]]] = None,
    disease_boxes: Optional[List[Dict[str, Any]]] = None,
    healthy_box: Optional[Dict[str, Any]] = None,
) -> str:
    """Draw organism boxes and disease-region boxes separately.

    species_boxes = organism localization (cyan box).
    disease_boxes = native disease-model ROI (coloured box, disease label).
    healthy_box   = a single healthy detection; draws a GREEN box with ✅ HEALTHY label.
    """
    species_boxes = species_boxes or []
    disease_boxes = disease_boxes or []

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font = None
    try:
        font_size = max(13, int(min(w, h) / 34))
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        pass

    # Organism box(es): one visual layer for the hard fish/prawn gate.
    for idx, b in enumerate(species_boxes, 1):
        bbox = b.get("bbox", [])
        if len(bbox) != 4:
            continue
        species_label = str(b.get("label", "AQUATIC SPECIES")).upper()
        conf_pct = float(b.get("confidence", 0.0)) * 100.0
        _draw_corner_box(
            draw,
            bbox,
            outline="#00d9ff",
            width=max(3, int(min(w, h) / 160)),
            corner_color="#ffffff"
        )
        _draw_label(
            draw,
            bbox[0],
            bbox[1],
            f"{species_label} ({conf_pct:.1f}%)",
            "#007f99",
            font,
            w
        )

    # ---------------------------------------------------------------
    # HEALTHY box: green border around the fish/prawn with ✅ HEALTHY
    # ---------------------------------------------------------------
    if healthy_box is not None:
        hbbox = healthy_box.get("bbox", [])
        hconf = float(healthy_box.get("confidence", 0.0)) * 100.0
        c_name = str(healthy_box.get("class_name", "Healthy"))
        GREEN = "#10b981"
        if len(hbbox) == 4:
            # Draw a bright green corner box
            _draw_corner_box(
                draw,
                hbbox,
                outline=GREEN,
                width=max(4, int(min(w, h) / 120)),
                corner_color="#ffffff",
            )
            _draw_label(
                draw,
                hbbox[0],
                hbbox[1],
                f"\u2705 HEALTHY ({hconf:.1f}%)",
                GREEN,
                font,
                w,
            )
        else:
            # No bbox from model — stamp a centred ✅ HEALTHY banner at top
            banner_text = f"\u2705 HEALTHY  {c_name}  ({hconf:.1f}%)"
            fs = max(16, int(min(w, h) / 22))
            try:
                big_font = ImageFont.truetype("arial.ttf", fs)
            except Exception:
                big_font = font
            try:
                tb = draw.textbbox((0, 0), banner_text, font=big_font)
                tb_w = tb[2] - tb[0] + 24
                tb_h = tb[3] - tb[1] + 14
            except Exception:
                tb_w = int(len(banner_text) * fs * 0.60) + 24
                tb_h = fs + 14
            bx = max(0, (w - tb_w) // 2)
            by = max(0, int(h * 0.04))
            draw.rounded_rectangle([bx, by, bx + tb_w, by + tb_h], radius=8, fill=GREEN)
            try:
                draw.text((bx + 12, by + 7), banner_text, fill="white", font=big_font)
            except Exception:
                draw.text((bx + 12, by + 7), banner_text, fill="white")

    # ---------------------------------------------------------------
    # Disease boxes: preserve exact native model ROI.
    # Only drawn when the class is a REAL disease (not healthy).
    # ---------------------------------------------------------------
    color_map = {
        "Bacterial Diseases": "#ef4444",
        "Parasitic Diseases": "#f59e0b",
        "White Tail Diseases": "#ea580c",
        "Fungal Diseases": "#8b5cf6",
        "WSSV": "#dc2626",
        "Black_Gill": "#d97706",
        "WSSV_Black_Gill": "#b91c1c",
    }

    disease_idx = 0
    for b in disease_boxes:
        bbox = b.get("bbox", [])
        c_name = str(b.get("class_name", "Disease"))

        # Skip healthy detections that were accidentally passed as disease boxes
        if c_name in {"Healthy Fish", "Healthy"}:
            continue

        if len(bbox) != 4:
            continue

        disease_idx += 1
        color = color_map.get(c_name, "#ff1744")
        conf_pct = float(b.get("confidence", 0.0)) * 100.0

        _draw_corner_box(
            draw,
            bbox,
            outline=color,
            width=max(4, int(min(w, h) / 120)),
            corner_color="#ffffff"
        )
        _draw_label(
            draw,
            bbox[0],
            bbox[1],
            f"DISEASE {disease_idx}: {c_name} ({conf_pct:.1f}%)",
            color,
            font,
            w
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=94)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Backward-compatible wrapper used elsewhere in the existing UI.
def draw_annotated_box(image_path: str, boxes: List[Dict[str, Any]]) -> str:
    healthy = [b for b in (boxes or []) if b.get("class_name") in {"Healthy Fish", "Healthy"}]
    disease = [b for b in (boxes or []) if b.get("class_name") not in {"Healthy Fish", "Healthy"}]
    hbox = healthy[0] if healthy else None
    return draw_detection_result(image_path, species_boxes=None, disease_boxes=disease, healthy_box=hbox)


def _model_class_profile(model, class_map: Dict[int, str]) -> Dict[str, Any]:
    """Inspect a checkpoint's real class names without guessing by class id."""
    profile = {
        "model": model,
        "path": str(getattr(model, "ckpt_path", "") or ""),
        "raw_names": [],
        "recognized": [],
        "recognized_count": 0,
        "coverage": 0.0,
    }
    try:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            items = []
            for k, v in names.items():
                try:
                    items.append((int(k), str(v)))
                except Exception:
                    continue
            items.sort(key=lambda x: x[0])
        elif isinstance(names, (list, tuple)):
            items = list(enumerate([str(v) for v in names]))
        else:
            items = []

        recognized = []
        raw_names = []
        seen = set()
        for cls_id, raw in items:
            raw = str(raw).strip()
            raw_names.append(raw)
            canonical = _canonical_class_name(model, class_map, cls_id)
            if canonical and canonical in class_map.values() and canonical not in seen:
                seen.add(canonical)
                recognized.append(canonical)

        profile["raw_names"] = raw_names
        profile["recognized"] = recognized
        profile["recognized_count"] = len(recognized)
        profile["coverage"] = len(recognized) / max(1, len(class_map))
    except Exception as exc:
        profile["error"] = str(exc)
    return profile


def _select_primary_disease_model(models, class_map: Dict[int, str], species: str):
    """Choose one genuinely multi-class disease checkpoint.

    The previous app pooled every checkpoint. That is dangerous when one old
    checkpoint contains only a bacterial class: it can overpower the correct model.
    We now select the checkpoint with the broadest real class coverage and use that
    checkpoint as the source of truth.
    """
    profiles = [_model_class_profile(m, class_map) for m in (models or []) if m is not None]
    profiles = [p for p in profiles if p["recognized_count"] > 0]

    print(f"[MODEL PROFILE] {species.upper()} disease checkpoints:")
    for p in profiles:
        print(
            f"  - {p.get('path') or '<unknown>'} | "
            f"classes={p['raw_names']} | recognized={p['recognized']} | "
            f"coverage={p['coverage']:.0%}"
        )

    if not profiles:
        return None, [], "No checkpoint exposes recognized disease classes."

    # Prefer the broadest class coverage. Among equal coverage, prefer a 'best.pt'
    # checkpoint over a test/fast checkpoint.
    profiles.sort(
        key=lambda p: (
            p["recognized_count"],
            "best.pt" in p.get("path", "").lower(),
            "final" in p.get("path", "").lower(),
            -len(p.get("path", "")),
        ),
        reverse=True,
    )
    best = profiles[0]

    return best["model"], profiles, ""


def _model_supports_species_classes(model, class_map: Dict[int, str]) -> bool:
    """Return True when the checkpoint exposes at least one compatible class label."""
    try:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            items = []
            for k, v in names.items():
                try:
                    k_int = int(k)
                except Exception:
                    continue
                items.append((k_int, str(v)))
        elif isinstance(names, (list, tuple)):
            items = list(enumerate([str(v) for v in names]))
        else:
            items = []

        for cls_id, raw in items:
            norm = _normalized_label(raw)
            if any(_normalized_label(v) == norm for v in class_map.values()):
                return True
            if _canonical_class_name(model, class_map, cls_id) is not None:
                return True
        return False
    except Exception:
        return False


def _box_visual_consistency(image_path: str, det: Dict[str, Any], species: str, allow_closeup: bool = False) -> bool:
    """Model-first validation of a native YOLO disease ROI.

    Conservative geometric sanity checks. In close-up / skin crop mode,
    allow lesions occupying larger areas of the frame and moderate confidence.
    """
    conf = float(det.get("confidence", 0.0))
    cls = str(det.get("class_name", "")).strip()
    if cls in {"Healthy Fish", "Healthy"}:
        return conf >= (0.35 if allow_closeup else HEALTHY_MIN_CONF)

    bbox = det.get("bbox") or []
    if len(bbox) != 4:
        return False

    try:
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 640, 640

    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area_pct = (bw * bh) / max(1.0, float(w*h))

    if bw < 5 or bh < 5:
        return False
    max_area = 0.98 if allow_closeup else 0.85
    if area_pct > max_area:
        return False
    min_conf = 0.18 if allow_closeup else DISEASE_MIN_CONF
    return conf >= min_conf


def _filter_disease_false_positives(
    image_path: str,
    detections: List[Dict[str, Any]],
    species: str,
    allow_closeup: bool = False,
) -> List[Dict[str, Any]]:
    """Accept only sufficiently confident native detections from the selected model.

    This deliberately does NOT reinterpret the disease from pixel colour.
    The model's trained class name stays unchanged.
    """
    out = []
    for det in detections or []:
        cls = str(det.get("class_name", "")).strip()
        conf = float(det.get("confidence", 0.0))

        if cls in {"Healthy Fish", "Healthy"}:
            if conf >= (0.35 if allow_closeup else HEALTHY_MIN_CONF):
                out.append(det)
            continue

        min_conf = 0.20 if allow_closeup else DISEASE_MIN_CONF
        if not allow_closeup:
            if cls in {"Black_Gill", "WSSV_Black_Gill"}:
                min_conf = 0.55
            elif cls == "WSSV":
                min_conf = 0.50
            elif cls in {"Bacterial Diseases", "Fungal Diseases", "Parasitic Diseases", "White Tail Diseases"}:
                min_conf = 0.50

        if conf < min_conf:
            continue
        if _box_visual_consistency(image_path, det, species, allow_closeup=allow_closeup):
            out.append(det)
    return out


def _dedupe_detection_boxes(boxes: List[Dict[str, Any]], max_items: int = 8) -> List[Dict[str, Any]]:
    """Keep the strongest non-overlapping boxes of the same class."""
    kept: List[Dict[str, Any]] = []
    for det in sorted(boxes or [], key=lambda d: float(d.get("confidence", 0.0)), reverse=True):
        b1 = det.get("bbox") or []
        if len(b1) != 4:
            continue
        duplicate = False
        for old in kept:
            if det.get("class_name") != old.get("class_name"):
                continue
            b2 = old.get("bbox") or []
            if len(b2) != 4:
                continue
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            iou = inter / max(1.0, a1 + a2 - inter)
            if iou >= 0.45:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
        if len(kept) >= max_items:
            break
    return kept


def _save_inference_view(image: Image.Image, tag: str) -> str:
    """Save a temporary inference view and return its path."""
    view_dir = os.path.join(UPLOAD_FOLDER, "_inference_views")
    os.makedirs(view_dir, exist_ok=True)
    path = os.path.join(view_dir, f"{uuid.uuid4().hex}_{tag}.jpg")
    image.save(path, format="JPEG", quality=95)
    return path


def _make_inference_views(image_path: str) -> List[Tuple[str, Tuple[int, int, int, int], bool]]:
    """Create full-image + overlapping zoom views.

    Returns tuples of (temporary_path, crop_xyxy_on_original, is_full_view).
    The crop views help when the fish/prawn is large in the frame or only a
    diseased body region is visible. All predictions still come from the same
    trained disease checkpoint; no disease is inferred from handcrafted color rules.
    """
    views: List[Tuple[str, Tuple[int, int, int, int], bool]] = []
    try:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
            w, h = image.size

            full_path = _save_inference_view(image, "full")
            views.append((full_path, (0, 0, w, h), True))

            # Only create zoom views for reasonably large images. Four overlapping
            # quadrants preserve context while making small lesions more visible.
            if w >= 320 and h >= 320:
                overlap = 0.15
                cw = int(w * 0.60)
                ch = int(h * 0.60)
                positions = [
                    (0, 0),
                    (max(0, w - cw), 0),
                    (0, max(0, h - ch)),
                    (max(0, w - cw), max(0, h - ch)),
                ]
                # Add center crop for central lesions when the image is large.
                if w >= 640 and h >= 640:
                    positions.append((max(0, (w - cw)//2), max(0, (h - ch)//2)))

                seen = set()
                for idx, (x1, y1) in enumerate(positions, 1):
                    x1 = int(max(0, min(x1, w - 1)))
                    y1 = int(max(0, min(y1, h - 1)))
                    x2 = int(min(w, x1 + cw))
                    y2 = int(min(h, y1 + ch))
                    key = (x1, y1, x2, y2)
                    if key in seen or x2 - x1 < 120 or y2 - y1 < 120:
                        continue
                    seen.add(key)
                    crop = image.crop(key)
                    crop_path = _save_inference_view(crop, f"zoom{idx}")
                    views.append((crop_path, key, False))
    except Exception as exc:
        print(f"[MULTI-VIEW] Could not create views: {type(exc).__name__}: {exc}")

    return views


def _map_view_box_to_original(bbox: List[float], crop_xyxy: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> List[float]:
    """Translate a detection box from a crop back into original-image coordinates."""
    ox1, oy1, _, _ = crop_xyxy
    iw, ih = image_size
    x1 = max(0.0, min(float(iw), float(bbox[0]) + ox1))
    y1 = max(0.0, min(float(ih), float(bbox[1]) + oy1))
    x2 = max(0.0, min(float(iw), float(bbox[2]) + ox1))
    y2 = max(0.0, min(float(ih), float(bbox[3]) + oy1))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]



def _collect_yolo_classification(model, image_path: str, class_map: Dict[int, str], imgsz: int = 1280) -> Optional[Dict[str, Any]]:
    """Read a YOLO classification checkpoint when it returns `result.probs`.

    Classification checkpoints do not emit disease boxes. We therefore keep the
    disease class from the REAL model output and pair it with a visual lesion
    localization fallback so close-up disease photographs still show a box.
    """
    if model is None:
        return None
    try:
        results = model.predict(
            source=image_path,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as exc:
        print(f"[CLASSIFY] YOLO classification inference failed: {type(exc).__name__}: {exc}")
        return None

    best = None
    for result in results or []:
        probs = getattr(result, "probs", None)
        if probs is None:
            continue
        try:
            top1 = int(probs.top1)
            conf = float(probs.top1conf.item())
            canonical = _canonical_class_name(model, class_map, top1)
            if not canonical:
                continue
            item = {
                "class_id": top1,
                "class_name": canonical,
                "raw_label": str((getattr(result, "names", {}) or {}).get(top1, canonical)),
                "confidence": conf,
                "confidence_pct": round(conf * 100.0, 1),
            }
            if best is None or conf > best["confidence"]:
                best = item
        except Exception:
            continue
    return best


def _localize_visible_abnormal_region(image_path: str) -> Optional[List[float]]:
    """Best-effort localization for close-up/cropped disease photos.

    This does NOT decide the disease name. It only finds a visually abnormal
    region using colour/texture contrast against the surrounding tissue. This is
    intentionally used only when a classification checkpoint or vision model
    lacks a native disease bounding box.
    """
    if cv2 is None:
        return None
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if h < 40 or w < 40:
            return None

        # Lab residual highlights lesions that differ from nearby normal tissue.
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        smooth = cv2.GaussianBlur(lab, (0, 0), sigmaX=max(10.0, min(w, h) / 20.0))
        colour_residual = np.sqrt(np.sum((lab - smooth) ** 2, axis=2))

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx * gx + gy * gy)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        dark = np.maximum(0.0, 120.0 - hsv[:, :, 2].astype(np.float32))

        score = (
            colour_residual / (float(colour_residual.mean()) + 1e-6)
            + 0.35 * edges / (float(edges.mean()) + 1e-6)
            + 0.18 * sat / 255.0
            + 0.12 * dark / 120.0
        )

        # Ignore a thin border; backgrounds and frames are common false positives.
        border = max(6, int(min(w, h) * 0.05))
        score[:border, :] = 0
        score[-border:, :] = 0
        score[:, :border] = 0
        score[:, -border:] = 0

        positive = score[score > 0]
        if positive.size < 100:
            return None
        threshold = float(np.percentile(positive, 97.0))
        mask = (score >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        image_area = float(w * h)
        for i in range(1, n):
            x, y, bw, bh, area = [int(v) for v in stats[i]]
            if area < max(40, int(image_area * 0.00015)):
                continue
            if bw < 8 or bh < 8:
                continue
            area_pct = area / image_area
            if area_pct > 0.55:
                continue
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            # Prefer substantial interior regions over tiny edge artefacts.
            edge_dist = min(cx, cy, w - cx, h - cy) / max(1.0, min(w, h) / 2.0)
            component_score = float(score[labels == i].mean())
            rank = component_score * (0.6 + 0.8 * min(1.0, area / (image_area * 0.03))) * (0.75 + 0.5 * max(0.0, edge_dist))
            candidates.append((rank, area, [x, y, x + bw, y + bh]))

        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda t: (t[0], t[1]))
        return [float(v) for v in candidates[0][2]]
    except Exception as exc:
        print(f"[LOCALIZER] {type(exc).__name__}: {exc}")
        return None


def _classification_as_detection(
    classification: Dict[str, Any],
    image_path: str,
    class_map: Dict[int, str],
    source: str = "classification",
) -> Optional[Dict[str, Any]]:
    """Convert classification only when no disease localization is required.

    A classification model has no spatial output. Therefore a diseased
    classification is NOT converted into a guessed bounding box.
    """
    if not classification:
        return None
    class_name = classification.get("class_name")
    if not class_name:
        return None
    if class_name in {"Healthy Fish", "Healthy"}:
        return {
            **classification,
            "bbox": [],
            "area_percent": 0.0,
            "view": source,
        }
    # Disease classification without a real detection box is not sufficient
    # for this application's exact-localization requirement.
    return None

def detect_disease_spots(models, image_path: str, class_map: Dict[int, str], species: str) -> Dict[str, Any]:
    """Detect the exact trained disease class and localize its ROI.

    Pipeline:
      1. Select ONE primary multi-class checkpoint for the selected species.
      2. Run the original image first.
      3. When needed, run overlapping zoom views so cropped/close-up disease
         photos are not missed.
      4. Aggregate only the class labels emitted by that same checkpoint.
      5. Return the winning trained class and its real YOLO boxes mapped back to
         the original image.

    No fallback disease label, symptom dictionary, color heuristic or opposite
    species model is used to create the final disease answer.
    """
    model, profiles, select_error = _select_primary_disease_model(models, class_map, species)
    if model is None:
        return {
            "loaded": bool(models),
            "detections": [],
            "spots": [],
            "annotated_b64": None,
            "message": select_error or f"No reliable {species} disease model is available.",
            "model_profile": profiles,
        }

    try:
        with Image.open(image_path) as im:
            image_size = im.convert("RGB").size
    except Exception:
        image_size = (640, 640)

    views: List[Tuple[str, Tuple[int, int, int, int], bool]] = []
    try:
        # Always run the original view first.
        with Image.open(image_path) as src:
            image = src.convert("RGB")
            full_tmp = _save_inference_view(image, "primary")
            views.append((full_tmp, (0, 0, image_size[0], image_size[1]), True))

        # If the primary view is weak, create zoom views. This avoids unnecessary
        # extra inference for straightforward full-fish/full-prawn photographs.
        imgsz = _model_inference_imgsz(model)
        primary_boxes = _collect_yolo_boxes(model, image_path, class_map, conf=0.35, imgsz=imgsz)
        if not primary_boxes:
            views = _make_inference_views(image_path)
        else:
            # Also inspect zooms when the strongest hit is weak enough that a crop
            # may contain a clearer symptom.
            if float(primary_boxes[0].get("confidence", 0.0)) < 0.60:
                extra_views = _make_inference_views(image_path)[1:]
                views.extend(extra_views)

        candidates: List[Dict[str, Any]] = []
        temp_paths: List[str] = []

        for view_path, crop_xyxy, is_full in views:
            temp_paths.append(view_path)
            # Keep the original view at a slightly stricter threshold; zoom views
            # can use a lower threshold because the symptom occupies more pixels.
            conf = 0.35 if is_full else 0.30
            view_boxes = _collect_yolo_boxes(model, view_path, class_map, conf=conf, imgsz=imgsz)

            for d in view_boxes:
                mapped = dict(d)
                mapped["bbox"] = _map_view_box_to_original(d["bbox"], crop_xyxy, image_size)
                mapped["view"] = "full" if is_full else "zoom"
                # Recalculate original-image box area.
                bx = mapped["bbox"]
                mapped["area_percent"] = round(
                    max(0.0, bx[2]-bx[0]) * max(0.0, bx[3]-bx[1]) /
                    max(1.0, image_size[0] * image_size[1]) * 100.0,
                    2
                )
                candidates.append(mapped)

        # Clean up temporary views.
        for tp in temp_paths:
            try:
                os.remove(tp)
            except Exception:
                pass

        # IMPORTANT: some YOLO checkpoints are classification models, not
        # detection models. In that case result.boxes is empty even though the
        # disease class is correctly predicted. Use the real classifier class
        # and add a best-effort visible-lesion box for the affected region.
        if not candidates:
            cls_pred = _collect_yolo_classification(model, image_path, class_map, imgsz=imgsz)
            cls_det = _classification_as_detection(cls_pred, image_path, class_map, source="classification")
            if cls_det:
                candidates.append(cls_det)

        # Remove weak single-view disease candidates before choosing a final class.
        # A disease that appears only once at medium confidence is a common false
        # positive on healthy fish/prawn photos. A very strong single-view result
        # is still allowed, which preserves genuine close-up disease detection.
        disease_candidates = [
            d for d in candidates
            if d.get("class_name") not in {"Healthy Fish", "Healthy"}
        ]
        healthy_candidates = [
            d for d in candidates
            if d.get("class_name") in {"Healthy Fish", "Healthy"}
        ]

        if disease_candidates:
            grouped = {}
            for d in disease_candidates:
                k = d.get("class_name")
                grouped.setdefault(k, []).append(d)

            stable = []
            for cls_name, group in grouped.items():
                max_conf = max(float(d.get("confidence", 0.0)) for d in group)
                views_seen = {str(d.get("view", "full")) for d in group}
                if max_conf >= 0.25 or (len(views_seen) >= MIN_DISEASE_VIEW_SUPPORT and max_conf >= 0.18):
                    stable.extend(group)

            disease_candidates = stable if stable else disease_candidates
            candidates = disease_candidates + healthy_candidates

        if not candidates:
            return {
                "loaded": True,
                "detections": [],
                "spots": [],
                "annotated_b64": None,
                "message": f"No confident {species} disease/healthy class was detected.",
                "model_profile": _model_class_profile(model, class_map),
                "source_model": str(getattr(model, "ckpt_path", "") or ""),
            }

        # CRITICAL FIX:
        # Healthy and disease YOLO detections are NOT mutually exclusive boxes.
        # A whole-animal "Healthy" box can have high confidence even when a
        # localized disease box is also present. Therefore a healthy confidence
        # score must NEVER override an accepted disease detection.
        #
        # Rule:
        #   - If an accepted disease candidate exists, disease ALWAYS wins.
        #   - Healthy is returned only when there is NO accepted disease evidence.
        top_healthy = max(
            healthy_candidates,
            key=lambda d: float(d.get("confidence", 0.0)),
            default=None
        )
        top_disease = max(
            disease_candidates,
            key=lambda d: float(d.get("confidence", 0.0)),
            default=None
        )

        if top_disease is None and top_healthy is not None:
            hc = float(top_healthy.get("confidence", 0.0))
            if hc >= HEALTHY_MIN_CONF:
                return {
                    "loaded": True,
                    "detections": [top_healthy],
                    "spots": [],
                    "annotated_b64": None,
                    "message": f"{species.capitalize()} appears healthy according to the trained model; no accepted disease evidence was found.",
                    "model_profile": _model_class_profile(model, class_map),
                    "source_model": str(getattr(model, "ckpt_path", "") or ""),
                    "model_vote": {"Healthy": {"max_conf": hc, "count": len(healthy_candidates)}},
                    "analysis_engine": "local_yolo_healthy_no_disease",
                }

        # Never mix disease classes. The winning class is selected from evidence
        # produced by ONE checkpoint only. Prefer high confidence and repeated
        # observations across views, while keeping the strongest native score.
        class_stats: Dict[str, Dict[str, Any]] = {}
        for d in candidates:
            name = d["class_name"]
            st = class_stats.setdefault(name, {"max_conf": 0.0, "sum_conf": 0.0, "count": 0})
            c = float(d.get("confidence", 0.0))
            st["max_conf"] = max(st["max_conf"], c)
            st["sum_conf"] += c
            st["count"] += 1

        def _class_rank(item):
            name, st = item
            # Strongest direct model confidence dominates; repeated detections are
            # a tie-breaker rather than a source of a new label.
            return (st["max_conf"], st["count"], st["sum_conf"])

        # DISEASE-FIRST RULE: once a disease candidate has survived the stability
        # check, never let a high-confidence whole-animal Healthy box replace it.
        # This was the exact bug causing a clearly ulcerated fish to be shown as
        # "Safe / Healthy Fish" in the UI.
        accepted_disease = [
            d for d in candidates
            if d.get("class_name") not in {"Healthy Fish", "Healthy"}
        ]
        if accepted_disease:
            disease_class_stats = {
                name: st for name, st in class_stats.items()
                if name not in {"Healthy Fish", "Healthy"}
            }
            winning_class = max(disease_class_stats.items(), key=_class_rank)[0]
        else:
            healthy_stats = {
                name: st for name, st in class_stats.items()
                if name in {"Healthy Fish", "Healthy"}
            }
            if not healthy_stats:
                return {
                    "loaded": True,
                    "detections": [],
                    "spots": [],
                    "annotated_b64": None,
                    "message": f"No reliable {species} disease/healthy class was detected.",
                    "model_profile": _model_class_profile(model, class_map),
                    "source_model": str(getattr(model, "ckpt_path", "") or ""),
                }
            winning_class = max(healthy_stats.items(), key=_class_rank)[0]

        selected_boxes = [d for d in candidates if d.get("class_name") == winning_class]
        selected_boxes = _dedupe_detection_boxes(selected_boxes, max_items=8)

        is_healthy = winning_class in {"Healthy Fish", "Healthy"}
        if is_healthy:
            top = max(selected_boxes, key=lambda d: float(d.get("confidence", 0.0)))
            return {
                "loaded": True,
                "detections": [top],
                "spots": [],
                "annotated_b64": None,
                "message": f"{species.capitalize()} appears healthy according to the trained model.",
                "model_profile": _model_class_profile(model, class_map),
                "source_model": str(getattr(model, "ckpt_path", "") or ""),
                "model_vote": class_stats,
                "analysis_engine": "local_yolo_healthy",
            }

        # Disease box is a native YOLO ROI. If missing, extract visible lesion box.
        if not selected_boxes:
            abnormal_box = _localize_visible_abnormal_region(image_path)
            if abnormal_box:
                selected_boxes = [{
                    "class_name": winning_class,
                    "confidence": class_stats.get(winning_class, {}).get("max_conf", 0.65),
                    "bbox": abnormal_box,
                    "area_percent": 15.0
                }]
            else:
                return {
                    "loaded": True,
                    "detections": [],
                    "spots": [],
                    "annotated_b64": None,
                    "message": f"{winning_class} was predicted, but no localized disease region was returned.",
                    "model_profile": _model_class_profile(model, class_map),
                    "source_model": str(getattr(model, "ckpt_path", "") or ""),
                }

        width = image_size[0]
        spots = []
        for idx, d in enumerate(selected_boxes, 1):
            loc = _location_for_box(d["bbox"], width, species)
            spots.append({
                "spot_num": idx,
                "class_name": d["class_name"],
                "raw_label": d.get("raw_label", d["class_name"]),
                "confidence_pct": f"{float(d['confidence']) * 100.0:.1f}%",
                "location_en": loc["en"],
                "location_te": loc["te"],
                "location_hi": loc["hi"],
                "location_tel": loc["tel"],
                "box_coords": {
                    "x1": d["bbox"][0], "y1": d["bbox"][1],
                    "x2": d["bbox"][2], "y2": d["bbox"][3]
                }
            })

        return {
            "loaded": True,
            "detections": selected_boxes,
            "spots": spots,
            "annotated_b64": None,
            "message": f"{winning_class} disease spot(s) found.",
            "model_profile": _model_class_profile(model, class_map),
            "source_model": str(getattr(model, "ckpt_path", "") or ""),
            "model_vote": class_stats,
            "multi_view": len(views) > 1,
            "analysis_engine": "local_yolo_verified",
        }

    except Exception as exc:
        print(f"[MULTI-VIEW DETECTION ERROR] {type(exc).__name__}: {exc}")
        # Safe fallback: run the exact same selected checkpoint on the original
        # image only. We still never fabricate a disease label.
        boxes = _collect_yolo_boxes(
            model,
            image_path,
            class_map,
            conf=0.35,
            imgsz=_model_inference_imgsz(model),
        )
        if not boxes:
            return {
                "loaded": True,
                "detections": [],
                "spots": [],
                "annotated_b64": None,
                "message": f"No confident {species} disease/healthy class was detected.",
                "model_profile": _model_class_profile(model, class_map),
                "source_model": str(getattr(model, "ckpt_path", "") or ""),
            }
        winning_class = boxes[0]["class_name"]
        selected = _dedupe_detection_boxes([b for b in boxes if b["class_name"] == winning_class], max_items=8)
        return {
            "loaded": True,
            "detections": selected,
            "spots": [] if winning_class in {"Healthy Fish", "Healthy"} else selected,
            "annotated_b64": None,
            "message": f"{winning_class} disease spot(s) found." if winning_class not in {"Healthy Fish", "Healthy"} else f"{species.capitalize()} appears healthy according to the trained model.",
            "model_profile": _model_class_profile(model, class_map),
            "source_model": str(getattr(model, "ckpt_path", "") or ""),
        }


def fallback_aquatic_inspection(image_path: str, target_species: str = "fish") -> Optional[Dict[str, Any]]:
    """Local symptom-based and close-up YOLO recovery for difficult/skin-crop disease photos.

    This fallback is used when the full-body object detector does not find a whole fish/prawn.
    It runs zoom-crop YOLO inference, classification fallback, and visible lesion symptom localization.
    """
    if cv2 is None:
        return None

    species = str(target_species).strip().lower()
    if species not in {"fish", "prawn"}:
        return None

    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        if h < 80 or w < 80:
            return None

        # ===============================================================
        # FIRST: selected-species YOLO close-up recovery
        # ===============================================================
        # This is the important path for images where only the affected
        # portion of the animal is visible. We deliberately use ONLY the
        # requested species' trained checkpoint. No opposite-species model
        # can create a disease label here.
        try:
            models = ALL_FISH_MODELS if species == "fish" else ALL_PRAWN_MODELS
            class_map = FISH_CLASSES if species == "fish" else PRAWN_CLASSES
            model, _, _ = _select_primary_disease_model(models, class_map, species)

            if model is not None:
                candidates = []
                temp_paths = []
                with Image.open(image_path) as src_img:
                    original_size = src_img.convert("RGB").size

                views = _make_inference_views(image_path)
                for view_path, crop_xyxy, is_full in views:
                    temp_paths.append(view_path)
                    # Low threshold is intentional only for close-up recovery.
                    # Final acceptance is checked again below.
                    for conf in (0.20, 0.12):
                        boxes = _collect_yolo_boxes(
                            model, view_path, class_map, conf=conf, imgsz=1536
                        )
                        if boxes:
                            for det in boxes:
                                mapped = dict(det)
                                mapped["bbox"] = _map_view_box_to_original(
                                    det.get("bbox", []),
                                    crop_xyxy,
                                    original_size,
                                )
                                mapped["view"] = "full" if is_full else "zoom"
                                candidates.append(mapped)
                            break

                for tp in temp_paths:
                    try:
                        os.remove(tp)
                    except Exception:
                        pass

                disease = [
                    d for d in candidates
                    if d.get("class_name") not in {"Healthy Fish", "Healthy"}
                    and len(d.get("bbox", [])) == 4
                ]

                if disease:
                    grouped = {}
                    for d in disease:
                        grouped.setdefault(d.get("class_name"), []).append(d)

                    ranked = []
                    for class_name, group in grouped.items():
                        max_conf = max(
                            float(x.get("confidence", 0.0)) for x in group
                        )
                        zoom_count = len({
                            str(x.get("view", "full")) for x in group
                        })
                        # Accept a strong direct close-up detection, or a
                        # lower-confidence class supported by multiple views.
                        if max_conf >= 0.34 or (zoom_count >= 2 and max_conf >= 0.25):
                            ranked.append((max_conf, zoom_count, class_name, group))

                    if ranked:
                        _, _, winning_class, group = max(
                            ranked, key=lambda x: (x[0], x[1])
                        )
                        selected = _dedupe_detection_boxes(group, max_items=8)

                        # IMPORTANT: close-up YOLO boxes can land on a nearby
                        # texture/spot instead of the actual visible lesion.
                        # Before returning from this early close-up path, run the
                        # class-aware lesion localizer so the disease box is moved
                        # onto the real affected region. The disease CLASS remains
                        # the exact YOLO class; only bbox coordinates are corrected.
                        selected = _force_class_aware_disease_localization(
                            image_path, selected, species
                        )
                        selected = _apply_disease_box_refinement(
                            image_path, selected, species
                        )

                        best = max(
                            selected,
                            key=lambda x: float(x.get("confidence", 0.0)),
                        )
                        confidence = float(best.get("confidence", 0.0))
                        spots = []
                        for idx, d in enumerate(selected, 1):
                            loc = _location_for_box(d["bbox"], original_size[0], species)
                            spots.append({
                                "spot_num": idx,
                                "class_name": d["class_name"],
                                "raw_label": d.get("raw_label", d["class_name"]),
                                "confidence_pct": f"{float(d.get('confidence', 0.0))*100.0:.1f}%",
                                "location_en": loc["en"],
                                "location_te": loc["te"],
                                "location_hi": loc["hi"],
                                "location_tel": loc["tel"],
                                "box_coords": {
                                    "x1": d["bbox"][0], "y1": d["bbox"][1],
                                    "x2": d["bbox"][2], "y2": d["bbox"][3],
                                },
                            })

                        annotated_b64 = draw_detection_result(image_path, species_boxes=[], disease_boxes=selected)

                        return {
                            "success": True,
                            "is_valid_aquatic": True,
                            "species": species,
                            "requested_species": species,
                            "species_locked": True,
                            "predicted_class": winning_class,
                            "confidence": confidence,
                            "confidence_pct": f"{confidence*100.0:.1f}%",
                            "detections": selected,
                            "object_detections": [],
                            "spots": spots,
                            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
                            "clinical_data": CLINICAL_DATABASE.get(winning_class, {}),
                            "analysis_engine": "local_yolo_closeup_recovery",
                            "closeup_mode": True,
                            "closeup_yolo_recovery": True,
                            "message": f"{species.capitalize()} close-up disease spot(s) found: {winning_class}.",
                            "message_en": f"{species.capitalize()} close-up disease spot(s) found: {winning_class}.",
                            "message_te": f"{('చేప' if species == 'fish' else 'రొయ్య')} close-up disease spot(s) found: {winning_class}.",
                            "message_hi": f"{species.capitalize()} close-up disease spot(s) found: {winning_class}.",
                            "message_tel": f"{species.capitalize()} close-up disease spot(s) found: {winning_class}.",
                        }

                # Classification-only checkpoints have no boxes. Use the real
                # class output and localize the visible abnormal region.
                try:
                    cls_pred = _collect_yolo_classification(
                        model, image_path, class_map, imgsz=1536
                    )
                    if cls_pred:
                        class_name = str(cls_pred.get("class_name", ""))
                        cls_conf = float(cls_pred.get("confidence", 0.0))
                        if (
                            class_name
                            and class_name not in {"Healthy Fish", "Healthy"}
                            and cls_conf >= 0.50
                        ):
                            bbox = _localize_visible_abnormal_region(image_path)
                            if bbox:
                                det = dict(cls_pred)
                                det["bbox"] = bbox
                                det["source"] = "classification_closeup_recovery"

                                # Classification checkpoints have no native bbox.
                                # Localize the visible lesion using the same
                                # class-aware pass used by detection checkpoints.
                                det = _force_class_aware_disease_localization(
                                    image_path, [det], species
                                )[0]
                                det = _apply_disease_box_refinement(
                                    image_path, [det], species
                                )[0]

                                annotated_b64 = draw_detection_result(image_path, species_boxes=[], disease_boxes=[det])
                                spots = [{
                                    "spot_num": 1,
                                    "class_name": class_name,
                                    "raw_label": class_name,
                                    "confidence_pct": f"{cls_conf*100.0:.1f}%",
                                    "location_en": "Visible lesion on skin",
                                    "location_te": "చర్మంపై కనిపిస్తున్న వ్యాధి ప్రాంతం",
                                    "location_hi": "त्वचा पर दिखाई दे रहा रोग क्षेत्र",
                                    "location_tel": "Visible lesion on skin",
                                    "box_coords": {
                                        "x1": det["bbox"][0], "y1": det["bbox"][1],
                                        "x2": det["bbox"][2], "y2": det["bbox"][3]
                                    }
                                }]

                                return {
                                    "success": True,
                                    "is_valid_aquatic": True,
                                    "species": species,
                                    "requested_species": species,
                                    "species_locked": True,
                                    "predicted_class": class_name,
                                    "confidence": cls_conf,
                                    "confidence_pct": f"{cls_conf*100.0:.1f}%",
                                    "detections": [det],
                                    "object_detections": [],
                                    "spots": spots,
                                    "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
                                    "clinical_data": CLINICAL_DATABASE.get(class_name, {}),
                                    "analysis_engine": "local_yolo_classification_closeup",
                                    "closeup_mode": True,
                                    "closeup_yolo_recovery": True,
                                    "message": f"{species.capitalize()} close-up screening: {class_name}.",
                                    "message_en": f"{species.capitalize()} close-up screening: {class_name}.",
                                    "message_te": f"{('చేప' if species == 'fish' else 'రొయ్య')} close-up screening: {class_name}.",
                                    "message_hi": f"{species.capitalize()} close-up screening: {class_name}.",
                                    "message_tel": f"{species.capitalize()} close-up screening: {class_name}.",
                                }
                except Exception as exc:
                    print(f"[CLOSE-UP CLASSIFICATION] {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"[CLOSE-UP YOLO RECOVERY] {type(exc).__name__}: {exc}")

        # Work on HSV/Lab for robust colour/texture measurements.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        H, S, V = cv2.split(hsv)
        L, A, B = cv2.split(lab)

        # Re-use the abnormal-region localizer to obtain a stable ROI.
        abnormal_box = _localize_visible_abnormal_region(image_path)
        if abnormal_box:
            ax1, ay1, ax2, ay2 = [int(v) for v in abnormal_box]
            ax1 = max(0, min(w - 1, ax1)); ay1 = max(0, min(h - 1, ay1))
            ax2 = max(ax1 + 1, min(w, ax2)); ay2 = max(ay1 + 1, min(h, ay2))
            roi = bgr[ay1:ay2, ax1:ax2]
            roi_hsv = hsv[ay1:ay2, ax1:ax2]
            roi_mask_box = [ax1, ay1, ax2, ay2]
        else:
            roi = bgr
            roi_hsv = hsv
            roi_mask_box = [0, 0, w, h]

        if roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        rH, rS, rV = cv2.split(roi_hsv)
        mean_s = float(np.mean(rS))
        mean_v = float(np.mean(rV))

        # --------------------------------------------------------------
        # Fish symptom signals
        # --------------------------------------------------------------
        # Red/orange hemorrhagic-ulcer pixels.
        red1 = cv2.inRange(roi_hsv, np.array([0, 80, 45]), np.array([15, 255, 255]))
        red2 = cv2.inRange(roi_hsv, np.array([160, 70, 45]), np.array([179, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)
        red_ratio = float(np.count_nonzero(red_mask)) / max(1, rw * rh)

        # Very dark necrotic tissue.
        dark_mask = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([179, 255, 65]))
        dark_ratio = float(np.count_nonzero(dark_mask)) / max(1, rw * rh)

        # Bright/low-saturation patchiness (useful for fungal-like cottony areas
        # and prawn white-spot patterns, but intentionally scored conservatively).
        white_mask = cv2.inRange(roi_hsv, np.array([0, 0, 165]), np.array([179, 85, 255]))
        white_ratio = float(np.count_nonzero(white_mask)) / max(1, rw * rh)

        # High-frequency texture.
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())

        # Find a tighter lesion box inside the existing abnormal ROI using the
        # strongest symptom mask. This is only localization; the class is chosen
        # from the symptom scores below.
        candidate_mask = red_mask.copy()
        if species == "prawn" and dark_ratio > 0.10:
            candidate_mask = cv2.bitwise_or(candidate_mask, dark_mask)
        if species == "fish" and white_ratio > 0.18 and lap_var > 300:
            candidate_mask = cv2.bitwise_or(candidate_mask, white_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
        best_component = None
        best_area = 0
        for i in range(1, n):
            x, y, bw, bh, area = [int(v) for v in stats[i]]
            if area < max(20, int(rw * rh * 0.001)):
                continue
            if area > best_area:
                best_area = area
                best_component = (x, y, bw, bh)

        if best_component:
            cx, cy, cbw, cbh = best_component
            bx1 = roi_mask_box[0] + cx
            by1 = roi_mask_box[1] + cy
            bx2 = roi_mask_box[0] + cx + cbw
            by2 = roi_mask_box[1] + cy + cbh
            bbox = [float(max(0, bx1)), float(max(0, by1)),
                    float(min(w, bx2)), float(min(h, by2))]
        else:
            bbox = [float(v) for v in roi_mask_box]

        # --------------------------------------------------------------
        # Conservative class scores.
        # --------------------------------------------------------------
        if species == "fish":
            # Ulcer/hemorrhage is the strongest visible signal in the supplied
            # type of close-up image and maps to the project's Bacterial Diseases
            # class. Fungal gets priority only when a broad cottony/light lesion is
            # visible with lower red dominance. White-tail requires a tail-like
            # whitening pattern and cannot be safely inferred from every close-up.
            bacterial_score = min(0.94, 0.52 + 1.55 * min(red_ratio, 0.18) + 0.35 * min(dark_ratio, 0.18))
            fungal_score = min(0.88, 0.42 + 1.05 * min(white_ratio, 0.28) + 0.00008 * min(lap_var, 2500))
            parasitic_score = min(0.72, 0.26 + 0.55 * min(lap_var / 1800.0, 1.0) + 0.22 * min(white_ratio, 0.25))
            white_tail_score = 0.20 + 0.65 * min(white_ratio, 0.30)

            scores = {
                "Bacterial Diseases": bacterial_score if red_ratio >= 0.015 or dark_ratio >= 0.12 else 0.22,
                "Fungal Diseases": fungal_score if white_ratio >= 0.20 else 0.20,
                "Parasitic Diseases": parasitic_score if red_ratio < 0.05 else 0.18,
                "White Tail Diseases": white_tail_score if white_ratio >= 0.26 else 0.15,
            }
        else:
            # Prawn: dark gill/necrotic areas favour Black Gill; multiple bright
            # spots favour WSSV. Both together favour WSSV + Black Gill.
            black_score = 0.34 + 1.45 * min(dark_ratio, 0.30)
            wssv_score = 0.30 + 1.30 * min(white_ratio, 0.35)
            combo_score = min(0.94, (black_score + wssv_score) / 1.55)
            scores = {
                "Black_Gill": min(0.90, black_score),
                "WSSV": min(0.88, wssv_score),
                "WSSV_Black_Gill": combo_score if black_score > 0.48 and wssv_score > 0.48 else 0.18,
            }

        disease, score = max(scores.items(), key=lambda kv: kv[1])

        # Require enough visual evidence. Lowered threshold so cropped/close-up fish
        # skin images with lesions are safely diagnosed instead of falsely rejected.
        if float(score) < 0.28:
            return None

        det_obj = {
            "class_name": disease,
            "raw_label": disease,
            "confidence": float(min(0.85, max(0.40, score))),
            "confidence_pct": float(min(85.0, max(40.0, score * 100.0))),
            "bbox": bbox,
            "area_percent": round((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) / max(1, w*h) * 100.0, 2),
            "source": "local_symptom_fallback"
        }

        annotated_b64 = draw_detection_result(image_path, species_boxes=[], disease_boxes=[det_obj])

        return {
            "success": True,
            "is_valid_aquatic": True,
            "species": species,
            "requested_species": species,
            "species_locked": True,
            "predicted_class": disease,
            "confidence": float(min(0.85, max(0.40, score))),
            "confidence_pct": f"{min(85.0, max(40.0, score * 100.0)):.1f}%",
            "detections": [det_obj],
            "object_detections": [det_obj],
            "spots": [{
                "spot_num": 1,
                "class_name": disease,
                "confidence_pct": f"{min(85.0, max(40.0, score * 100.0)):.1f}%",
                "box_coords": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "location_en": "Visible abnormal/diseased region",
                "location_te": "కనిపిస్తున్న వ్యాధి/అసాధారణ ప్రాంతం",
                "location_hi": "दिखाई दे रहा असामान्य/रोग क्षेत्र",
                "location_tel": "Visible disease area"
            }],
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
            "clinical_data": CLINICAL_DATABASE.get(disease, {}),
            "analysis_engine": "local_symptom_fallback",
            "closeup_mode": True,
            "symptom_metrics": {
                "red_ratio": round(red_ratio, 4),
                "dark_ratio": round(dark_ratio, 4),
                "white_ratio": round(white_ratio, 4),
                "texture": round(lap_var, 1),
            },
            "message": f"{species.capitalize()} symptom screening suggests {disease}.",
            "message_en": f"{species.capitalize()} symptom screening suggests {disease}.",
            "message_te": f"{('చేప' if species == 'fish' else 'రొయ్య')} లక్షణాల పరిశీలనలో {disease} గా గుర్తించబడింది.",
            "message_hi": f"{species.capitalize()} लक्षण जांच से {disease} का संकेत मिलता है।",
            "message_tel": f"{species.capitalize()} symptom screening suggests {disease}.",
        }
    except Exception as exc:
        print(f"[SYMPTOM FALLBACK] {type(exc).__name__}: {exc}")
        return None

def _analyze_species_morphology(image_path: str) -> Dict[str, Any]:
    """Analyze image structure, sharpness, edge gradients, and colors using OpenCV.
    Provides morphological verification to complement YOLO neural models.
    """
    if cv2 is None:
        return {"valid": True, "blur_score": 100.0, "is_prawn_like": False, "is_fish_like": False}

    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return {"valid": False, "reason": "Unreadable image"}

        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 1. Blur / detail metric
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Color distributions
        b_c, g_c, r_c = cv2.split(bgr)
        mean_r = float(np.mean(r_c))
        mean_g = float(np.mean(g_c))
        mean_b = float(np.mean(b_c))

        # Red / orange ratio (common in Koi / hemorrhagic ulcers / diseased fish)
        red_mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
        red_ratio = float(np.count_nonzero(cv2.bitwise_or(red_mask1, red_mask2))) / max(1, w * h)

        # Translucent / grey-green ratio (common in vannamei prawn / shrimp carapace)
        prawn_mask = cv2.inRange(hsv, np.array([15, 10, 40]), np.array([110, 140, 220]))
        prawn_color_ratio = float(np.count_nonzero(prawn_mask)) / max(1, w * h)

        # 3. Horizontal vs Vertical edge gradients (prawn abdominal segments produce distinct oriented ridges)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        mean_grad = float(np.mean(grad_mag))

        return {
            "valid": True,
            "blur_score": round(lap_var, 2),
            "is_too_blurry": lap_var < 10.0,
            "red_ratio": round(red_ratio, 4),
            "prawn_color_ratio": round(prawn_color_ratio, 4),
            "edge_density": round(mean_grad, 2),
            "aspect_ratio": round(w / max(1.0, h), 2)
        }
    except Exception as exc:
        print(f"[MORPHOLOGY] Analysis warning: {exc}")
        return {"valid": True, "blur_score": 50.0}


def _evaluate_cross_species(image_path: str, target_species: str) -> Dict[str, Any]:
    """Verify the selected species before using either disease model."""
    target = str(target_species).strip().lower()
    target_models = ALL_FISH_MODELS if target == "fish" else ALL_PRAWN_MODELS
    target_classes = FISH_CLASSES if target == "fish" else PRAWN_CLASSES
    model, profiles, select_error = _select_primary_disease_model(
        target_models,
        target_classes,
        target,
    )
    if model is None:
        return {
            "rejected": False,
            "target_species": target,
            "fish_conf": 0.0,
            "prawn_conf": 0.0,
            "fish_boxes": [],
            "prawn_boxes": [],
            "model_error": select_error,
            "model_profile": profiles,
        }

    other_species = "prawn" if target == "fish" else "fish"
    other_models = ALL_PRAWN_MODELS if target == "fish" else ALL_FISH_MODELS
    other_classes = PRAWN_CLASSES if target == "fish" else FISH_CLASSES
    other_model, _, other_error = _select_primary_disease_model(
        other_models,
        other_classes,
        other_species,
    )
    if other_model is None:
        print(f"[SPECIES GATE] {other_species} model unavailable: {other_error}")
        return {
            "rejected": True,
            "rejection_type": "species_uncertain",
            "target_species": target,
        }

    target_boxes = _collect_yolo_boxes(
        model,
        image_path,
        target_classes,
        conf=0.05,
        imgsz=_model_inference_imgsz(model),
    )
    other_boxes = _collect_yolo_boxes(
        other_model,
        image_path,
        other_classes,
        conf=0.05,
        imgsz=_model_inference_imgsz(other_model),
    )
    fish_boxes = target_boxes if target == "fish" else other_boxes
    prawn_boxes = target_boxes if target == "prawn" else other_boxes
    target_detection_conf = max(
        (float(detection.get("confidence", 0.0)) for detection in target_boxes),
        default=0.0,
    )

    if SPECIES_GATE_MODEL is None:
        print("[SPECIES GATE] Species classifier is unavailable; rejecting unverified image.")
        return {
            "rejected": True,
            "rejection_type": "species_uncertain",
            "target_species": target,
        }

    try:
        gate_features = _species_gate_features(image_path, fish_boxes, prawn_boxes)
        prawn_probability = float(SPECIES_GATE_MODEL.predict_proba(gate_features)[0][1])
    except Exception as exc:
        print(f"[SPECIES GATE] Verification failed: {type(exc).__name__}: {exc}")
        return {
            "rejected": True,
            "rejection_type": "species_uncertain",
            "target_species": target,
        }

    target_probability = prawn_probability if target == "prawn" else 1.0 - prawn_probability
    other_probability = 1.0 - target_probability
    target_healthy_class = "Healthy Fish" if target == "fish" else "Healthy"
    other_healthy_class = "Healthy" if other_species == "prawn" else "Healthy Fish"
    strongest_target = max(
        target_boxes,
        key=lambda detection: float(detection.get("confidence", 0.0)),
        default=None,
    )
    strongest_other = max(
        other_boxes,
        key=lambda detection: float(detection.get("confidence", 0.0)),
        default=None,
    )
    accepted_as_target = target_probability >= SPECIES_GATE_ACCEPT_CONF

    healthy_species_conflict = (
        accepted_as_target
        and strongest_target is not None
        and strongest_target.get("class_name") == target_healthy_class
        and float(strongest_target.get("confidence", 0.0))
        >= SPECIES_GATE_TARGET_HEALTHY_CONFLICT_CONF
        and strongest_other is not None
        and strongest_other.get("class_name") == other_healthy_class
        and float(strongest_other.get("confidence", 0.0)) >= SPECIES_GATE_HEALTHY_CONFLICT_CONF
    )
    if healthy_species_conflict:
        accepted_as_target = False

    if not accepted_as_target and target == "fish":
        accepted_as_target = (
            target_probability >= SPECIES_GATE_FISH_BORDERLINE_CONF
            and strongest_target is not None
            and strongest_target.get("class_name") != target_healthy_class
            and float(strongest_target.get("confidence", 0.0))
            >= SPECIES_GATE_FISH_BORDERLINE_DETECTION_CONF
            and strongest_other is not None
            and strongest_other.get("class_name") != other_healthy_class
            and float(strongest_other.get("confidence", 0.0))
            >= SPECIES_GATE_OTHER_CONFIRM_CONF
        )
    elif not accepted_as_target and target == "prawn":
        accepted_as_target = (
            not healthy_species_conflict
            and target_probability >= SPECIES_GATE_PRAWN_BORDERLINE_CONF
            and strongest_target is not None
            and float(strongest_target.get("confidence", 0.0))
            >= SPECIES_GATE_PRAWN_BORDERLINE_DETECTION_CONF
        )

    if not accepted_as_target:
        rejection_type = (
            "opposite_species"
            if other_probability >= SPECIES_GATE_ACCEPT_CONF
            else "species_uncertain"
        )
        print(
            f"[SPECIES GATE] rejected target={target} "
            f"target_confidence={target_probability:.3f} "
            f"other={other_species} other_confidence={other_probability:.3f}"
        )
        return {
            "rejected": True,
            "rejection_type": rejection_type,
            "detected_species": other_species,
            "detected_conf": other_probability,
            "target_species": target,
        }

    # Generic COCO classes often mistake prawns for people or birds. A strong,
    # class-labeled result from the selected fish/prawn disease model takes
    # precedence over that generic detector.
    non_aquatic = check_non_aquatic_object(image_path)
    species_model_confirms_target = (
        target_probability >= 0.90
        and strongest_target is not None
        and target_detection_conf >= 0.50
    )
    if non_aquatic and target_detection_conf < 0.55 and not species_model_confirms_target:
        return {
            "rejected": True,
            "rejection_type": "non_aquatic",
            "detected_object": non_aquatic,
            "target_species": target
        }

    fish_conf = target_detection_conf if target == "fish" else 0.0
    prawn_conf = target_detection_conf if target == "prawn" else 0.0
    print(
        f"[DISEASE SCORES] target={target} | selected={getattr(model, 'ckpt_path', '')} "
        f"| detection_confidence={target_detection_conf:.3f} "
        f"| species_confidence={target_probability:.3f}"
    )

    return {
        "rejected": False,
        "target_species": target,
        "target_model": model,
        "target_species_confidence": target_probability,
        "fish_conf": fish_conf,
        "prawn_conf": prawn_conf,
        "best_fish_det": max(fish_boxes, key=lambda d: float(d.get("confidence", 0.0))) if fish_boxes else None,
        "best_prawn_det": max(prawn_boxes, key=lambda d: float(d.get("confidence", 0.0))) if prawn_boxes else None,
        "fish_boxes": fish_boxes,
        "prawn_boxes": prawn_boxes
    }


def _make_species_rejection(species: str, reason_en: str, reason_te: str = None, reason_hi: str = None, reason_tel: str = None) -> Dict[str, Any]:
    te = reason_te or reason_en
    hi = reason_hi or reason_en
    tel = reason_tel or reason_en
    return {
        "success": False,
        "is_valid_aquatic": False,
        "species": species,
        "predicted_class": None,
        "confidence": 0.0,
        "confidence_pct": "0.0%",
        "detections": [],
        "object_detections": [],
        "spots": [],
        "annotated_image": None,
        "clinical_data": {},
        "message": reason_en,
        "message_en": reason_en,
        "message_te": te,
        "message_hi": hi,
        "message_tel": tel,
        "voice_script_te": te,
        "voice_script_hi": hi,
        "voice_script_en": reason_en,
    }



# ==============================================================================
# 4A. OPTIONAL GENERAL VISUAL DISEASE ANALYSIS
#
# YOLO can only recognize classes it was trained for. To handle a NEW disease
# image that is outside the local dataset classes, AquaCare can optionally use
# a multimodal vision model. The vision result is never silently converted into
# one of the local YOLO classes. It is returned as a separate visual diagnosis.
#
# Set OPENAI_API_KEY in the environment to enable this feature.
# Example (PowerShell):
#   $env:OPENAI_API_KEY = "YOUR_KEY"
#
# The API is optional. Without a key, the normal local YOLO pipeline continues.
# ==============================================================================
VISION_MODEL = os.getenv("AQUACARE_VISION_MODEL", "gpt-6-astra")
VISION_TIMEOUT_SECONDS = int(os.getenv("AQUACARE_VISION_TIMEOUT", "45"))


def _vision_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        raw = f.read()
    encoded = base64.b64encode(raw).decode("ascii")
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg"
    if ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    return f"data:{mime};base64,{encoded}"


def _find_json_object(value):
    if isinstance(value, dict):
        # Prefer an object that looks like the requested diagnosis schema.
        if "disease" in value or "diagnosis" in value or "bbox" in value:
            return value
        for v in value.values():
            found = _find_json_object(v)
            if found is not None:
                return found
    elif isinstance(value, list):
        for v in value:
            found = _find_json_object(v)
            if found is not None:
                return found
    return None


def _extract_response_text(payload: Dict[str, Any]) -> str:
    """Extract plain text from the LLM payload.

    The payload may be one of several shapes:
    1. {"output_text": "..."}
    2. {"output": [{"content": [{"text": "..."}, ...]}, ...]}
    3. {"output": {"text": "..."}}
    4. A bare list of such dicts.

    This helper normalises all cases into a single string. If nothing
    textual is found, it returns an empty string.
    """
    # Case 1 – direct output_text
    if isinstance(payload, dict):
        out_text = payload.get("output_text")
        if isinstance(out_text, str) and out_text.strip():
            return out_text.strip()
        # Case 2 – output list of rich response objects
        output = payload.get("output")
        if isinstance(output, list):
            pieces: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []) or []:
                    if isinstance(content, dict):
                        txt = content.get("text")
                        if isinstance(txt, str):
                            pieces.append(txt)
            if pieces:
                return "\n".join(pieces).strip()
        # Case 3 – output dict with a direct text field
        if isinstance(output, dict):
            txt = output.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    # Case 4 – payload itself is a list of rich blocks
    if isinstance(payload, list):
        pieces: List[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    txt = content.get("text")
                    if isinstance(txt, str):
                        pieces.append(txt)
        if pieces:
            return "\n".join(pieces).strip()
    return ""



def _parse_json_text(text_value: str) -> Optional[Dict[str, Any]]:
    if not text_value:
        return None
    raw = text_value.strip()
    # Strip accidental markdown fences.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Recover a JSON object embedded in a short explanation.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalise_vision_bbox(bbox: Any, width: int, height: int) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(x) for x in bbox]
    except Exception:
        return None
    # Requested schema uses 0..1000 coordinates. Also tolerate normalized 0..1.
    if max(abs(v) for v in vals) <= 1.5:
        x1, y1, x2, y2 = [v * 1000.0 for v in vals]
    else:
        x1, y1, x2, y2 = vals
    x1 = max(0.0, min(1000.0, x1))
    y1 = max(0.0, min(1000.0, y1))
    x2 = max(0.0, min(1000.0, x2))
    y2 = max(0.0, min(1000.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    px = [
        int(round(x1 / 1000.0 * (width - 1))),
        int(round(y1 / 1000.0 * (height - 1))),
        int(round(x2 / 1000.0 * (width - 1))),
        int(round(y2 / 1000.0 * (height - 1))),
    ]
    if px[2] <= px[0] or px[3] <= px[1]:
        return None
    return px


def _annotate_vision_bbox(image_path: str, bbox: Optional[List[int]], label: str, confidence: float) -> Optional[str]:
    try:
        im = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        w, h = im.size
        if bbox:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=(255, 20, 50), width=max(4, min(w, h) // 160))
            text = f"{label} {confidence * 100:.1f}%"
            try:
                font = ImageFont.truetype("arial.ttf", max(18, min(w, h) // 28))
            except Exception:
                font = ImageFont.load_default()
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 10)
            draw.rectangle([x1, ty, min(w, x1 + tw + 14), min(h, ty + th + 10)], fill=(255, 20, 50))
            draw.text((x1 + 7, ty + 4), text, fill=(255, 255, 255), font=font)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=92)
        return base64.b64encode(out.getvalue()).decode("ascii")
    except Exception as exc:
        print(f"[VISION ANNOTATION] {exc}")
        return None


def general_vision_diagnosis(image_path: str, species: str) -> Optional[Dict[str, Any]]:
    """Ask a multimodal model for a symptom-based second layer.

    This is for NEW / out-of-dataset images. It does not pretend that a
    non-supported image is one of the local YOLO classes.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    species_label = "fish" if species == "fish" else "prawn/shrimp"
    known = list((FISH_CLASSES if species == "fish" else PRAWN_CLASSES).values())
    prompt = f"""
You are the visual-diagnosis layer of an aquaculture screening application.
Analyze this image of a {species_label}. The image may be a complete animal,
a close-up, or only the diseased body part.

IMPORTANT:
1. Identify the visible species as fish or prawn/shrimp.
2. Diagnose from visible symptoms, lesion appearance, distribution, color,
ulceration, gill changes, tail changes, body spots, fungal growth, etc.
3. You are NOT restricted to the local YOLO classes. The local classes are:
   {known}
   A NEW image may show another disease not present in those classes.
4. Never force a match to a local class just because one exists.
5. If the visual evidence is insufficient, return disease="Uncertain".
6. Give ONE most likely disease name, not a list.
7. Give one bounding box around the main visible abnormal/diseased region.
   Coordinates MUST be integers from 0 to 1000 in [x1,y1,x2,y2].
8. For a healthy animal, disease must be "Healthy" and bbox can be null.
9. Return ONLY valid JSON with this exact shape:
{{
  "species": "fish|prawn|unknown",
  "disease": "one disease name or Healthy or Uncertain",
  "confidence": 0.0,
  "symptoms": ["short visible symptom", "short visible symptom"],
  "reason": "one short sentence explaining the visual evidence",
  "bbox": [x1,y1,x2,y2] or null
}}
""".strip()

    try:
        image_url = _vision_data_url(image_path)
        body = {
            "model": VISION_MODEL,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url}
                ]
            }]
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=VISION_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text_value = _extract_response_text(payload)
        obj = _parse_json_text(text_value) or _find_json_object(payload)
        if not isinstance(obj, dict):
            return None

        pred_species = str(obj.get("species", "unknown")).strip().lower()
        disease = str(obj.get("disease") or obj.get("diagnosis") or "Uncertain").strip()
        try:
            confidence = float(obj.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.0
        symptoms = obj.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = [str(symptoms)] if symptoms else []
        symptoms = [str(x).strip() for x in symptoms if str(x).strip()][:6]
        reason = str(obj.get("reason", "")).strip()

        with Image.open(image_path) as im:
            width, height = im.convert("RGB").size
        bbox = _normalise_vision_bbox(obj.get("bbox"), width, height)

        return {
            "species": pred_species,
            "disease": disease,
            "confidence": confidence,
            "symptoms": symptoms,
            "reason": reason,
            "bbox": bbox,
            "model": VISION_MODEL,
        }
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="ignore")[:500]
        except Exception:
            detail = str(exc)
        print(f"[VISION API] HTTP {exc.code}: {detail}")
    except Exception as exc:
        print(f"[VISION API] {type(exc).__name__}: {exc}")
    return None


def _vision_result_payload(image_path: str, target_species: str, vision: Dict[str, Any]) -> Dict[str, Any]:
    species_label = "Fish" if target_species == "fish" else "Prawn"
    disease = vision.get("disease", "Uncertain")
    confidence = float(vision.get("confidence", 0.0))
    bbox = vision.get("bbox")
    annotated_b64 = _annotate_vision_bbox(image_path, bbox, disease, confidence)
    healthy = disease.strip().lower() == "healthy"
    if healthy:
        title = f"Healthy {species_label}"
    else:
        title = disease

    clinical_data = {
        "title_en": title,
        "title_te": title,
        "title_hi": title,
        "title_tel": title,
        "is_healthy": healthy,
        "severity": "Visual AI screening",
        "medicine_en": "Image-based screening only. Confirm the disease with an aquatic-animal veterinarian or qualified fish-health professional before treatment.",
        "medicine_te": "ఇది image-based screening మాత్రమే. Treatment ముందు aquatic-animal veterinarian లేదా qualified fish-health professional తో confirm చేయండి.",
        "medicine_hi": "यह केवल image-based screening है। उपचार से पहले aquatic-animal veterinarian या qualified fish-health professional से पुष्टि करें।",
        "medicine_tel": "Idi image-based screening matrame. Treatment mundu aquatic-animal veterinarian tho confirm cheyandi.",
        "food_guide_en": "Maintain clean, species-appropriate feed and avoid overfeeding while the stock is under stress.",
        "food_guide_te": "సరైన species feed ఇవ్వండి; stock stress లో ఉన్నప్పుడు overfeeding చేయకండి.",
        "food_guide_hi": "उचित species feed दें और stress में overfeeding से बचें।",
        "food_guide_tel": "Proper species feed ivvandi; stress unnapudu overfeeding cheyakandi.",
        "voice_script_en": f"{species_label} visual screening result: {title}, confidence {confidence * 100:.1f} percent.",
        "voice_script_te": f"{species_label} visual screening result: {title}, confidence {confidence * 100:.1f} percent.",
        "voice_script_hi": f"{species_label} visual screening result: {title}, confidence {confidence * 100:.1f} percent.",
        "voice_script_tel": f"{species_label} visual screening result: {title}, confidence {confidence * 100:.1f} percent.",
    }

    try:
        with Image.open(image_path) as _im:
            _w, _h = _im.size
    except Exception:
        _w, _h = 640, 640

    return {
        "success": True,
        "is_valid_aquatic": True,
        "species": target_species,
        "predicted_class": disease,
        "confidence": confidence,
        "confidence_pct": f"{confidence * 100:.1f}%",
        "detections": ([{
            "class_name": disease,
            "raw_label": disease,
            "confidence": confidence,
            "bbox": bbox,
            "area_percent": round(max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1]) / max(1, _w * _h) * 100, 2) if bbox else 0.0,
            "source": "vision_ai"
        }] if bbox else []),
        "object_detections": [],
        "spots": [],
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
        "clinical_data": clinical_data,
        "analysis_engine": "general_vision_ai",
        "vision_symptoms": vision.get("symptoms", []),
        "vision_reason": vision.get("reason", ""),
        "vision_model": vision.get("model", VISION_MODEL),
        "message": f"{species_label} visual AI screening: {title} ({confidence * 100:.1f}%).",
        "message_en": f"{species_label} visual AI screening: {title} ({confidence * 100:.1f}%).",
        "message_te": f"{species_label} visual AI screening: {title} ({confidence * 100:.1f}%).",
        "message_hi": f"{species_label} visual AI screening: {title} ({confidence * 100:.1f}%).",
        "message_tel": f"{species_label} visual AI screening: {title} ({confidence * 100:.1f}%).",
    }


def analyze_species_disease(image_path: str, target_species: str = "fish") -> Dict[str, Any]:
    """Rigorous Fish/Prawn inspection & disease detection pipeline.

    Enforces strict species isolation:
      - Fish mode accepts only real fish images (rejects prawn or non-aquatic uploads).
      - Prawn mode accepts only real prawn images (rejects fish or non-aquatic uploads).
      - Rejections provide clear trilingual (English, Telugu, Hindi) explanations.
      - When verified, returns real YOLO disease bounding boxes, severity, and treatments.
    """
    target = (target_species or "").strip().lower()
    if target not in {"fish", "prawn"}:
        return _make_species_rejection(
            "unknown",
            "❌ Please select Fish or Prawn first.",
            "❌ ముందుగా చేప (Fish) లేదా రొయ్య (Prawn) ఎంచుకోండి.",
            "❌ कृपया पहले मछली या झींगा चुनें।"
        )

    species = target
    disease_models = ALL_FISH_MODELS if species == "fish" else ALL_PRAWN_MODELS
    disease_classes = FISH_CLASSES if species == "fish" else PRAWN_CLASSES

    if not disease_models:
        return _make_species_rejection(
            species,
            f"❌ The trained {species} disease model is not loaded. Check the {species}/models folder.",
            f"❌ {('చేప' if species == 'fish' else 'రొయ్య')} మోడల్ లోడ్ కాలేదు. {species}/models ఫోల్డర్ పరిశీలించండి.",
            f"❌ {species} मॉडल लोड नहीं हुआ है। कृपया {species}/models फ़ोल्डर की जांच करें।"
        )

    # -------------------------------------------------------------------------
    # STEP 1: Cross-species verification and negative rejection gate
    # -------------------------------------------------------------------------
    cross_eval = _evaluate_cross_species(image_path, target_species=species)

    if cross_eval.get("rejected"):
        rejection_type = cross_eval.get("rejection_type")
        if rejection_type == "non_aquatic":
            obj_name = str(cross_eval.get("detected_object", "Non-aquatic object")).capitalize()
            return _make_species_rejection(
                species,
                f"❌ {obj_name} detected. This scanner accepts only Fish or Prawn images.",
                f"❌ {obj_name} గుర్తించబడింది. ఈ స్కానర్‌లో కేవలం చేప లేదా రొయ్య ఫోటోలు మాత్రమే అనుమతించబడతాయి.",
                f"❌ {obj_name} पाया गया। कृपया केवल मछली या झींगा की फोटो अपलोड करें。"
            )
        elif rejection_type == "opposite_species":
            det_sp = cross_eval.get("detected_species", "prawn" if species == "fish" else "fish")
            det_conf = float(cross_eval.get("detected_conf", 0.0))
            if species == "fish" and det_sp == "prawn":
                return _make_species_rejection(
                    "fish",
                    f"❌ Wrong species selected! This image contains a Prawn ({det_conf*100:.1f}% match), but you selected Fish mode. Please switch to the Prawn tab.",
                    f"❌ తప్పుడు విభాగం! ఇది రొయ్య (Prawn) ఫోటో ({det_conf*100:.1f}% మ్యాచ్), కానీ మీరు చేపల (Fish) విభాగంలో ఉన్నారు. దయచేసి రొయ్యల (Prawn) విభాగాన్ని ఎంచుకోండి.",
                    f"❌ गलत प्रजाति! यह झींगा (Prawn) की फोटो है ({det_conf*100:.1f}%), लेकिन आपने मछली (Fish) मोड चुना है। कृपया झींगा टैब चुनें।"
                )
            elif species == "prawn" and det_sp == "fish":
                return _make_species_rejection(
                    "prawn",
                    f"❌ Wrong species selected! This image contains a Fish ({det_conf*100:.1f}% match), but you selected Prawn mode. Please switch to the Fish tab.",
                    f"❌ తప్పుడు విభాగం! ఇది చేప (Fish) ఫోటో ({det_conf*100:.1f}% మ్యాచ్), కానీ మీరు రొయ్యల (Prawn) విభాగంలో ఉన్నారు. దయచేసి చేపల (Fish) విభాగాన్ని ఎంచుకోండి.",
                    f"❌ गलत प्रजाति! यह मछली (Fish) की फोटो है ({det_conf*100:.1f}%), लेकिन आपने झींगा (Prawn) मोड चुना है। कृपया मछली टैब चुनें।"
                )
        elif rejection_type == "species_uncertain":
            return _make_species_rejection(
                species,
                f"❌ This image could not be confidently verified as a {species.capitalize()}. Please upload one clear, well-lit image of only a {species}.",
                f"❌ ఈ చిత్రంలో {('చేప' if species == 'fish' else 'రొయ్య')}ని నమ్మకంగా నిర్ధారించలేకపోయాం. ఒక స్పష్టమైన, మంచి వెలుతురు ఉన్న {('చేప' if species == 'fish' else 'రొయ్య')} ఫోటోను మాత్రమే అప్‌లోడ్ చేయండి.",
                f"❌ इस तस्वीर में {('मछली' if species == 'fish' else 'झींगा')} की पुष्टि भरोसे से नहीं हो सकी। कृपया केवल {('मछली' if species == 'fish' else 'झींगा')} की एक साफ़ और अच्छी रोशनी वाली तस्वीर अपलोड करें।"
            )

    # -------------------------------------------------------------------------
    # STEP 2: Collect detections from target species models
    # -------------------------------------------------------------------------
    raw_detections = cross_eval.get("fish_boxes" if species == "fish" else "prawn_boxes", [])
    inference_model = cross_eval.get("target_model")
    if not raw_detections and inference_model is not None:
        raw_detections = _collect_yolo_boxes(
            inference_model,
            image_path,
            disease_classes,
            conf=0.30,
            imgsz=_model_inference_imgsz(inference_model),
        )

    # Deduplicate and sort detections
    detections = _dedupe_detection_boxes(raw_detections, max_items=8)

    disease_boxes = [
        d for d in detections
        if d.get("class_name") not in {"Healthy Fish", "Healthy"}
        and float(d.get("confidence", 0.0)) >= 0.25
    ]
    healthy_boxes = [
        d for d in detections
        if d.get("class_name") in {"Healthy Fish", "Healthy"}
        and float(d.get("confidence", 0.0)) >= HEALTHY_MIN_CONF
    ]

    strongest_healthy = max(
        healthy_boxes,
        key=lambda d: float(d.get("confidence", 0.0)),
        default=None,
    )
    strongest_disease = max(
        disease_boxes,
        key=lambda d: float(d.get("confidence", 0.0)),
        default=None,
    )
    if strongest_healthy is not None and (
        strongest_disease is None
        or float(strongest_healthy.get("confidence", 0.0))
        >= float(strongest_disease.get("confidence", 0.0)) + DISEASE_HEALTHY_MARGIN
    ):
        disease_boxes = []
        healthy_boxes = [strongest_healthy]

    # -------------------------------------------------------------------------
    # STEP 3: Diagnose disease or healthy condition
    # -------------------------------------------------------------------------
    if disease_boxes:
        if species == "fish":
            try:
                with Image.open(image_path) as im:
                    image_area = max(1, im.width * im.height)
            except Exception:
                image_area = 640 * 640

            def fish_detection_score(d):
                bbox = d.get("bbox") or []
                if len(bbox) != 4:
                    return float(d.get("confidence", 0.0))
                box_area = max(1.0, bbox[2] - bbox[0]) * max(1.0, bbox[3] - bbox[1])
                # Down-rank tiny, high-confidence false positives without
                # letting box size outweigh the model's confidence entirely.
                return float(d.get("confidence", 0.0)) * (box_area / image_area) ** 0.25

            best_det = max(disease_boxes, key=fish_detection_score)
        else:
            best_det = max(disease_boxes, key=lambda d: float(d.get("confidence", 0.0)))
        predicted_class = best_det["class_name"]
        confidence = float(best_det.get("confidence", 0.0))
        final_boxes = [
            d for d in disease_boxes
            if d.get("class_name") == predicted_class
        ][:6]

        annotated_b64 = draw_detection_result(
            image_path,
            species_boxes=[],
            disease_boxes=final_boxes,
        )

        try:
            with Image.open(image_path) as im:
                image_width = im.size[0]
        except Exception:
            image_width = 640

        spots = []
        for idx, db in enumerate(final_boxes, 1):
            loc = _location_for_box(db.get("bbox", []), image_width, species)
            spots.append({
                "spot_num": idx,
                "class_name": db.get("class_name"),
                "raw_label": db.get("raw_label", db.get("class_name")),
                "confidence_pct": f"{float(db.get('confidence', 0.0))*100:.1f}%",
                "location_en": loc["en"],
                "location_te": loc["te"],
                "location_hi": loc["hi"],
                "location_tel": loc["tel"],
                "box_coords": {
                    "x1": db["bbox"][0], "y1": db["bbox"][1],
                    "x2": db["bbox"][2], "y2": db["bbox"][3],
                },
            })

    elif healthy_boxes:
        best_det = max(healthy_boxes, key=lambda d: float(d.get("confidence", 0.0)))
        predicted_class = best_det["class_name"]
        confidence = float(best_det.get("confidence", 0.0))
        final_boxes = [best_det]

        annotated_b64 = draw_detection_result(
            image_path,
            species_boxes=[],
            disease_boxes=[],
            healthy_box=best_det,
        )
        spots = []

    else:
        # -------------------------------------------------------------------
        # STEP 3B: Multi-view YOLO close-up pipeline (detect_disease_spots)
        # Handles close-up / affected area photos where full-body fails.
        # -------------------------------------------------------------------
        print(f"[PIPELINE] Main detection empty → trying multi-view close-up YOLO for {species}...")
        try:
            spots_result = detect_disease_spots(
                [inference_model] if inference_model is not None else disease_models,
                image_path,
                disease_classes,
                species,
            )
            dets = spots_result.get("detections", [])
            if dets:
                best_det2 = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
                predicted_class2 = best_det2.get("class_name", "")
                confidence2 = float(best_det2.get("confidence", 0.0))
                final_boxes2 = [b for b in dets if b.get("class_name") == predicted_class2][:6]
                spots2 = spots_result.get("spots", [])
                is_healthy2 = predicted_class2 in {"Healthy Fish", "Healthy"}
                annotated_b64_2 = (
                    draw_detection_result(image_path, species_boxes=[], disease_boxes=[], healthy_box=best_det2)
                    if is_healthy2 else
                    draw_detection_result(image_path, species_boxes=[], disease_boxes=final_boxes2)
                )
                clinical_data2 = dict(CLINICAL_DATABASE.get(predicted_class2, {}))
                if clinical_data2:
                    clinical_data2["is_healthy"] = is_healthy2
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute(
                            "INSERT INTO inspection_history (inspection_type, species, predicted_class, confidence, status, notes) VALUES (?,?,?,?,?,?)",
                            ("disease_detection", species, predicted_class2, confidence2,
                             clinical_data2.get("severity", "Normal" if is_healthy2 else "High"),
                             f"Spots: {len(spots2)} | engine: closeup_multiview"),
                        )
                        conn.commit()
                        conn.close()
                    except Exception:
                        pass
                    sp_cap = species.capitalize()
                    sp_te = "చేప" if species == "fish" else "రొయ్య"
                    sp_hi = "मछली" if species == "fish" else "झींगा"
                    t_en = clinical_data2.get("title_en", predicted_class2)
                    t_te = clinical_data2.get("title_te", predicted_class2)
                    t_hi = clinical_data2.get("title_hi", predicted_class2)
                    t_tel = clinical_data2.get("title_tel", predicted_class2)
                    return {
                        "success": True, "is_valid_aquatic": True, "species": species,
                        "predicted_class": predicted_class2, "confidence": confidence2,
                        "confidence_pct": f"{confidence2*100.0:.1f}%",
                        "detections": final_boxes2, "object_detections": [], "spots": spots2,
                        "annotated_image": f"data:image/jpeg;base64,{annotated_b64_2}" if annotated_b64_2 else None,
                        "clinical_data": clinical_data2, "analysis_engine": "yolo_closeup_multiview",
                        "message": f"{sp_cap} close-up: {t_en} ({confidence2*100.0:.1f}%).",
                        "message_en": f"{sp_cap} close-up: {t_en} ({confidence2*100.0:.1f}%).",
                        "message_te": f"{sp_te} close-up విశ్లేషణ: {t_te} ({confidence2*100.0:.1f}%).",
                        "message_hi": f"{sp_hi} close-up: {t_hi} ({confidence2*100.0:.1f}%).",
                        "message_tel": f"{sp_cap} close-up: {t_tel} ({confidence2*100.0:.1f}%).",
                        "voice_script_te": clinical_data2.get("voice_script_te", f"{sp_te} పై {t_te} గుర్తించబడింది."),
                        "voice_script_hi": clinical_data2.get("voice_script_hi", f"{sp_hi} में {t_hi} पाया गया।"),
                        "voice_script_en": clinical_data2.get("voice_script_en", f"{sp_cap} diagnosed with {t_en}."),
                    }
        except Exception as exc:
            print(f"[PIPELINE] Multi-view close-up failed: {type(exc).__name__}: {exc}")

        # -------------------------------------------------------------------
        # STEP 3C: Symptom/colour-based fallback (fallback_aquatic_inspection)
        # Last resort for very close-up / skin-crop disease photos.
        # -------------------------------------------------------------------
        print(f"[PIPELINE] Trying symptom-based fallback for {species}...")
        try:
            fallback_result = fallback_aquatic_inspection(image_path, target_species=species)
            if fallback_result and fallback_result.get("success"):
                return fallback_result
        except Exception as exc:
            print(f"[PIPELINE] Symptom fallback failed: {type(exc).__name__}: {exc}")

        # -------------------------------------------------------------------
        # STEP 3D: Nothing found — return clear rejection
        # -------------------------------------------------------------------
        morph = _analyze_species_morphology(image_path)
        if morph.get("is_too_blurry"):
            return _make_species_rejection(
                species,
                "❌ Uploaded image is too blurry. Please take a clear, well-lit photo of the aquatic animal.",
                "❌ ఫోటో చాలా మసకగా (Blur) ఉంది. దయచేసి స్పష్టమైన వెలుతురులో ఫోటో తీసి అప్‌లోడ్ చేయండి.",
                "❌ फोटो बहुत धुंधली है। कृपया स्पष्ट और अच्छी रोशनी में फोटो अपलोड करें।"
            )
        sp_label_en = "Fish" if species == "fish" else "Prawn"
        sp_label_te = "చేప" if species == "fish" else "రొయ్య"
        sp_label_hi = "मछली" if species == "fish" else "झींगा"
        return _make_species_rejection(
            species,
            f"❌ No recognizable {sp_label_en} disease or healthy symptoms detected. Please upload a clear photo of the {sp_label_en} or affected lesion area.",
            f"❌ స్పష్టమైన {sp_label_te} వ్యాధి లేదా ఆరోగ్య లక్షణాలు గుర్తించబడలేదు. దయచేసి {sp_label_te} లేదా వ్యాధి సోకిన భాగాన్ని స్పష్టంగా ఫోటో తీసి అప్‌లోడ్ చేయండి.",
            f"❌ కోई स्पष्ट {sp_label_hi} रोग या लक्षण नहीं मिले। कृपया स्पष्ट फोटो अपलोड करें।"
        )

    clinical_data = dict(CLINICAL_DATABASE.get(predicted_class, {}))
    if not clinical_data:
        return _make_species_rejection(
            species,
            f"❌ Predicted '{predicted_class}', but clinical database record is missing.",
            f"❌ మోడల్ '{predicted_class}' గుర్తించింది, కానీ క్లినికల్ రికార్డ్ అందుబాటులో లేదు."
        )
    clinical_data["is_healthy"] = predicted_class in {"Healthy Fish", "Healthy"}

    # Record in SQLite history
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO inspection_history (inspection_type, species, predicted_class, confidence, status, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "disease_detection",
                species,
                predicted_class,
                confidence,
                clinical_data.get("severity", "Normal" if clinical_data["is_healthy"] else "High"),
                f"Spots: {len(spots)}",
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    sp_cap = species.capitalize()
    sp_te = "చేప" if species == "fish" else "రొయ్య"
    sp_hi = "मछली" if species == "fish" else "झींगा"

    title_en = clinical_data.get("title_en", predicted_class)
    title_te = clinical_data.get("title_te", predicted_class)
    title_hi = clinical_data.get("title_hi", predicted_class)
    title_tel = clinical_data.get("title_tel", predicted_class)

    return {
        "success": True,
        "is_valid_aquatic": True,
        "species": species,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_pct": f"{confidence*100.0:.1f}%",
        "detections": final_boxes,
        "object_detections": [],
        "spots": spots,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None,
        "clinical_data": clinical_data,
        "analysis_engine": "yolo_deep_learning",
        "message": f"{sp_cap} analyzed: {title_en} ({confidence*100.0:.1f}%).",
        "message_en": f"{sp_cap} analyzed: {title_en} ({confidence*100.0:.1f}%).",
        "message_te": f"{sp_te} విశ్లేషణ: {title_te} ({confidence*100.0:.1f}%).",
        "message_hi": f"{sp_hi} विश्लेषण: {title_hi} ({confidence*100.0:.1f}%).",
        "message_tel": f"{sp_cap} analysis: {title_tel} ({confidence*100.0:.1f}%).",
        "voice_script_te": clinical_data.get("voice_script_te", f"{sp_te} పై {title_te} గుర్తించబడింది."),
        "voice_script_hi": clinical_data.get("voice_script_hi", f"{sp_hi} में {title_hi} पाया गया।"),
        "voice_script_en": clinical_data.get("voice_script_en", f"{sp_cap} diagnosed with {title_en}."),
    }


# ==============================================================================
# 5. DISSOLVED OXYGEN (DO) PREDICTION ENGINE
# ==============================================================================
def empirical_do_estimation(temp: float, ph: float, salinity: float, ammonia: float,
                            nitrite: float, turbidity: float, feed: float,
                            rainfall: float, aerator: int, species: str) -> float:
    base_do = 8.24 - (0.13 * (temp - 25.0))
    base_do -= (0.042 * salinity)
    base_do += 0.28 * (ph - 7.5)
    if aerator:
        base_do += 2.1
    base_do -= (ammonia * 3.2)
    base_do -= (nitrite * 1.8)
    base_do -= (feed * 0.04)
    base_do -= (turbidity * 0.012)
    if rainfall > 0:
        base_do -= (rainfall * 0.018)
    if species == "prawn":
        base_do -= 0.15
    return max(1.1, min(11.5, round(base_do, 2)))

def predict_do_level(payload: Dict[str, Any]) -> Dict[str, Any]:
    species = str(payload.get("species") or "fish").strip().lower()
    if species not in {"fish", "prawn"}:
        return {"success": False, "message": "Select Fish or Prawn only."}
    try:
        temp = float(payload.get("water_temp_C", ""))
        ph = float(payload.get("pH", ""))
        salinity = float(payload.get("salinity_ppt", ""))
        ammonia = float(payload.get("ammonia_mg_L", ""))
        nitrite = float(payload.get("nitrite_mg_L", ""))
        turbidity = float(payload.get("turbidity_NTU", ""))
        feed = float(payload.get("feed_kg", ""))
        rainfall = float(payload.get("rainfall_mm", ""))
        aerator = int(str(payload.get("aerator_on", "1")).lower() in {"1", "true", "on", "yes"})
    except (TypeError, ValueError):
        return {"success": False, "message": "Please enter valid values for all water-quality fields."}

    model = FISH_DO_MODEL if species == "fish" else PRAWN_DO_MODEL
    meta = FISH_DO_META if species == "fish" else PRAWN_DO_META

    if model is None:
        predicted_do = empirical_do_estimation(temp, ph, salinity, ammonia, nitrite, turbidity, feed, rainfall, aerator, species)
        engine_used = "empirical_science"
    else:
        features = meta.get("features") if isinstance(meta, dict) else None
        if not features:
            features = getattr(model, "feature_names_in_", None)
        features = [str(x) for x in features] if features is not None else [
            "water_temp_C", "pH", "salinity_ppt", "ammonia_mg_L", "nitrite_mg_L", "turbidity_NTU", "feed_kg", "rainfall_mm", "aerator_on", "species"
        ]
        default_species = "Rohu (L. rohita)" if species == "fish" else "Vannamei (L. vannamei)"
        values = {
            "water_temp_C": temp, "pH": ph, "salinity_ppt": salinity, "ammonia_mg_L": ammonia, "nitrite_mg_L": nitrite,
            "turbidity_NTU": turbidity, "feed_kg": feed, "rainfall_mm": rainfall, "aerator_on": aerator,
            "species": str(payload.get("species_name") or default_species)
        }
        aliases = {
            "water_temp": temp, "water_temperature": temp, "temperature": temp, "temperature_C": temp,
            "ph": ph, "salinity": salinity, "ammonia": ammonia, "nitrite": nitrite, "turbidity": turbidity,
            "feed": feed, "rainfall": rainfall, "aerator": aerator
        }
        row = {c: values.get(c, aliases.get(c, str(payload.get(c, "")) if c == "species" else np.nan)) for c in features}
        try:
            predicted_do = round(float(model.predict(pd.DataFrame([row], columns=features))[0]), 2)
            engine_used = "machine_learning"
        except Exception:
            predicted_do = empirical_do_estimation(temp, ph, salinity, ammonia, nitrite, turbidity, feed, rainfall, aerator, species)
            engine_used = "empirical_science_fallback"

    predicted_do = max(0.0, predicted_do)
    if predicted_do >= 6.0:
        status, color = "Optimal", "#10b981"
        ae = "Dissolved Oxygen is optimal. Continue routine monitoring."
        at = "చెరువులో ఆక్సిజన్ సరైన స్థాయిలో ఉంది. సాధారణ మానిటరింగ్ కొనసాగించండి."
        ahi = "घुलित ऑक्सीजन का स्तर उत्तम है। नियमित निगरानी जारी रखें।"
        alt = "Oxygen optimal ga undi. Regular monitoring continue cheyandi."
        aer = "Continue normal aeration and check DO before dawn."
        aert = "సాధారణ ఎరేషన్ కొనసాగించి తెల్లవారుజామున DO చెక్ చేయండి."
        aerhi = "सामान्य एरेटर चलाएं और सुबह के समय ऑक्सीजन की जाँच करें।"
        aerlt = "Normal aeration continue chesi dawn mundu DO check cheyandi."
        fe = "Normal feeding based on biomass and feed response."
        ft = "చేప/రొయ్యల బరువు మరియు మేత స్పందనను బట్టి సాధారణ మేత ఇవ్వండి."
        fhi = "बायोमास के अनुसार सामान्य आहार दें।"
        felt = "Biomass mariyu feeding response batti normal feed ivvandi."
    elif predicted_do >= 4.0:
        status, color = "Moderate / Caution", "#f59e0b"
        ae = "DO is moderate. Monitor closely and review aeration, feeding and water quality."
        at = "ఆక్సిజన్ మధ్యస్థంగా ఉంది. ఎరేటర్, మేత, నీటి నాణ్యతను పరిశీలించండి."
        ahi = "ऑक्सीजन मध्यम है। एरेटर, चारा और पानी की गुणवत्ता पर ध्यान दें।"
        alt = "Oxygen moderate ga undi. Aeration, feed mariyu water quality check cheyandi."
        aer = "Increase night/pre-dawn aeration if measured DO is falling."
        aert = "DO తగ్గుతున్నట్లయితే రాత్రి/తెల్లవారుజామున ఎరేషన్ పెంచండి."
        aerhi = "रात और सुबह के समय एरेटर का समय बढ़ाएं।"
        aerlt = "DO tagguthunte night/pre-dawn aeration penchandi."
        fe = "Reduce feeding if feed response is poor or water quality worsens."
        ft = "మేత స్పందన తగ్గితే లేదా నీటి నాణ్యత దిగజారితే మేత తగ్గించండి."
        fhi = "चारा थोड़ा कम करें।"
        felt = "Feed response poor unte leda water quality worsen aithe feed thagginchandi."
    else:
        status, color = "Critical (Hypoxia Risk)", "#ef4444"
        ae = "Critical low DO screening result. Run aeration promptly, reduce feeding temporarily and re-measure DO."
        at = "ఆక్సిజన్ చాలా తక్కువగా ఉంది. వెంటనే ఎరేటర్లు నడిపి మేతను తాత్కాలికంగా తగ్గించి DO మళ్లీ కొలవండి."
        ahi = "ऑक्सीजन बहुत कम है! तुरंत सभी एरेटर चलाएं और चारा अस्थायी रूप से रोकें।"
        alt = "Oxygen chala thakkuva undi. Ventane aerators on chesi feed temporarily thagginchi DO malli measure cheyandi."
        aer = "Run available aerators promptly and re-measure DO."
        aert = "అందుబాటులో ఉన్న ఎరేటర్లను వెంటనే నడిపి DO మళ్లీ కొలవండి."
        aerhi = "उपलब्ध सभी एरेटर तुरंत पूरी गति से चलाएं।"
        aerlt = "Available aerators ventane run chesi DO malli measure cheyandi."
        fe = "Pause or sharply reduce feeding until DO improves."
        ft = "DO మెరుగుపడే వరకు మేతను ఆపండి లేదా గణనీయంగా తగ్గించండి."
        fhi = "ऑक्सीजन सुधरने तक चारा पूरी तरह बंद रखें।"
        felt = "DO improve ayye varaku feed pause leda strong ga reduce cheyandi."

    return {
        "success": True,
        "species": species,
        "predicted_do_mg_l": predicted_do,
        "status": status,
        "status_color": color,
        "engine": engine_used,
        "advice_en": ae,
        "advice_te": at,
        "advice_hi": ahi,
        "advice_tel": alt,
        "aerator_en": aer,
        "aerator_te": aert,
        "aerator_hi": aerhi,
        "aerator_tel": aerlt,
        "feed_en": fe,
        "feed_te": ft,
        "feed_hi": fhi,
        "feed_tel": felt,
        "voice_script_te": f"{species} కోసం అంచనా ఆక్సిజన్ స్థాయి {predicted_do} మిల్లీగ్రాములు లీటరుకు ఉంది. {at}",
        "voice_script_hi": f"अनुमानित ऑक्सीजन स्तर {predicted_do} मिलीग्राम प्रति लीटर है। {ahi}"
    }

# ==============================================================================
# 6. FARMER ASSISTANT INTENTS
# ==============================================================================
FARMER_INTENTS = [
    {
        "keywords": ["oxygen", "dissolved oxygen", "do level", "do low", "do problem", "do ppm", "aerator", "hypoxia", "ఆక్సిజన్", "ఎరేటర్", "శ్వాస", "taggindi", "padipoindi", "ऑक्सीजन", "एरेटर"],
        "reply_english": "💧 **Oxygen (DO) Problem Guidance:**\n1. Turn on all available aerators immediately and monitor continuously.\n2. Avoid adding feed while the stock is under oxygen stress.\n3. Check dissolved oxygen, ammonia, nitrite and pond water condition.\n4. Seek local aquaculture/veterinary advice before using any chemical treatment.\n\nThis is farm screening guidance, not a veterinary prescription.",
        "reply_telenglish": "💧 **Oxygen (DO) Problem Solution:**\n1. **Ventane Action:** Anni aerators full speed tho on cheyandi (continuous 24 hours).\n2. **Emergency Mandu:** Oxygen tablets (O2 Max leda Hydrogen peroxide) ekaraaniki 1-2 kg cheruvulo veyandi.\n3. **Meetha:** Meetha veyadam thakkaligamga poorthiga aapeyandi.\n4. **Neellu:** Veelaithe fresh neellu 20% pump cheyandi.",
        "reply_telugu": "💧 **ఆక్సిజన్ (DO) సమస్య పరిష్కారం:**\n1. **తక్షణ చర్య:** అన్ని ఎరేటర్లను వెంటనే ఆన్ చేసి నిరంతరం నడపండి.\n2. **అత్యవసర మందు:** ఎకరాకు 1-2 కేజీల ఆక్సిజన్ మాత్రలు (O2 Max) చెరువులో వేయండి.\n3. **మేత:** మేత వేయడం తాత్కాలికంగా పూర్తిగా ఆపివేయండి.\n4. **నీరు:** 20% మంచి నీటి మార్పిడి చేయండి.",
        "reply_hindi": "💧 **ऑक्सीजन (DO) समस्या समाधान:**\n1. **त्वरित कार्रवाई:** सभी एरेटर तुरंत 24 घंटे लगातार पूरी गति से चलाएं।\n2. **आपातकालीन दवा:** प्रति एकड़ 1-2 किलो ऑक्सीजन गोलियां (O2 Max) तालाब में डालें।\n3. **चारा:** चारा देना तुरंत पूरी तरह बंद करें।\n4. **जल विनिमय:** संभव हो तो 20% स्वच्छ पानी बदलें।",
        "voice_telugu": "చెరువులో ఆక్సిజన్ తక్కువగా ఉంటే వెంటనే అన్ని ఎరేటర్లను ఆన్ చేయండి. మేత వేయడం తాత్కాలికంగా ఆపి, ఎకరాకు ఒకటి నుండి రెండు కేజీల ఆక్సిజన్ మాత్రలు వేయండి.",
        "voice_hindi": "तालाब में ऑक्सीजन कम होने पर तुरंत सभी एरेटर चलाएं, चारा रोकें और प्रति एकड़ एक से दो किलो ऑक्सीजन गोलियां डालें।"
    },
    {
        "keywords": ["white spot", "wssv", "spots", "తెల్ల మచ్చ", "వైట్ స్పాట్", "thella macha", "सफेद धब्बा"],
        "reply_english": "🦐 **White Spot Syndrome (WSSV) Guidance:**\n1. Isolate suspected affected stock and strengthen biosecurity.\n2. Check water quality and dissolved oxygen immediately.\n3. Avoid relying on unverified chemical or antibiotic claims for a viral disease.\n4. Contact a qualified aquaculture specialist or diagnostic laboratory for confirmation before treatment or harvest decisions.",
        "reply_telenglish": "🦐 **Thella Macha (WSSV Virus) Emergency Guide:**\n1. **Royyalu 14 grams paina unte:** VENTANE HARVEST (pattadam) cheyandi! Alasyam cheyoddu!\n2. **Virus ki Mandu Ledu:** Ee virus ki chemicals levu, fake antibiotics vadi dabbu vrudha chesukovaddu.\n3. **Quarantine:** Cheruvu nundi bayataki neellu vadaloddu.\n4. **Sanitize:** Panta poyina cheruvulo 100 kg bleaching powder vesi disinfect cheyandi.",
        "reply_telugu": "🦐 **తెల్ల మచ్చ వ్యాధి (WSSV) అత్యవసర సలహా:**\n1. **హార్వెస్ట్:** రొయ్యలు 14 గ్రాములు దాటితే వెంటనే పట్టేయండి.\n2. **మందులు లేవు:** ఈ వైరస్ కు ఎలాంటి మందులు లేవు, అనవసర ఖర్చులు చేయకండి.\n3. **క్వారంటైన్:** చెరువు నీటిని కాలువలోకి వదలకండి.\n4. **బ్లీచింగ్:** ఎకరాకు 100 కేజీల బ్లీచింగ్ పౌడర్ వేసి శుద్ధి చేయండి.",
        "reply_hindi": "🦐 **सफेद धब्बा (WSSV) आपातकालीन मार्गदर्शन:**\n1. **तत्काल कटाई:** झींगा 14 ग्राम से अधिक है तो तुरंत हार्वेस्ट करें।\n2. **कोई दवा नहीं:** इस वायरस का कोई रासायनिक इलाज नहीं है।\n3. **क्वारंटाइन:** पानी नहरों में न छोड़ें।\n4. **ब्लीचिंग:** प्रति एकड़ 100 किलो ब्लीचिंग पाउडर डालें।",
        "voice_telugu": "రొయ్యలకు తెల్ల మచ్చ వైరస్ సోకితే రొయ్యలు మార్కెట్ సైజు ఉంటే వెంటనే పట్టేయండి. చెరువు తూములు మూసివేసి ఎరేటర్లు నడపండి.",
        "voice_hindi": "झींगा में सफेद धब्बा वायरस होने पर यदि वे बिक्री योग्य हैं तो तुरंत हार्वेस्ट करें।"
    },
    {
        "keywords": ["black gill", "gill", "మొప్ప", "నల్ల మొప్ప", "nalla moppa", "काली गलफड़", "गलफड़"],
        "reply_english": "🦐 **Black Gill Guidance:**\n1. Check dissolved oxygen and overall pond water quality.\n2. Reduce stress and closely observe feeding and shrimp behaviour.\n3. Check ammonia, nitrite and organic load where possible.\n4. Get professional confirmation before using disease-specific chemicals or medicines.",
        "reply_telenglish": "🦐 **Nalla Moppa (Black Gill) Treatment:**\n1. **Mandu (Adugu Shuddhi):** Zeolite 30 kg/acre mariyu Yucca extract 500 ml veyandi.\n2. **Probiotics:** Bacillus soil probiotics 1 kg/acre vesi adugu chetta kariginchandi.\n3. **Meetha:** Meethanu 35% thagginchandi. Meethalo Vitamin C 5g kalipi ivvandi.\n4. **Check Trays:** 2 hours tharwatha trays check chesi migilipothe meetha thagginchandi.",
        "reply_telugu": "🦐 **నల్ల మొప్ప వ్యాధి (బ్లాక్ గిల్) చికిత్స:**\n1. **అడుగు శుద్ధి:** ఎకరాకు 30 కేజీల జియోలైట్ మరియు 500 ml యుక్కా వేయండి.\n2. **ప్రోబయోటిక్స్:** మట్టి ప్రోబయోటిక్స్ 1 కేజీ వేసి అడుగున వ్యర్థాలను కరిగించండి.\n3. **మేత:** మేతను 35% తగ్గించి విటమిన్-సి కలిపి ఇవ్వండి.\n4. **ట్రేలు:** 2 గంటల తర్వాత ఫీడ్ ట్రేలు తనిఖీ చేయండి.",
        "reply_hindi": "🦐 **काली गलफड़ (ब्लैक गिल) उपचार:**\n1. **तल सफाई:** प्रति एकड़ 30 किलो जियोलाइट और 500 मिली युक्का अर्क डालें।\n2. **प्रोबायोटिक्स:** 1 किग्रा सॉइल प्रोबायोटिक्स डालें।\n3. **चारा:** चारा 35% कम करें और विटामिन-सी मिलाएं।",
        "voice_telugu": "నల్ల మొప్ప వ్యాధికి చెరువు అడుగున జియోలైట్ మరియు ప్రోబయోటిక్స్ వేయండి. మేతను ముప్పై ఐదు శాతం తగ్గించి విటమిన్ సి ఇవ్వండి.",
        "voice_hindi": "काली गलफड़ के लिए तालाब के तल पर जियोलाइट और प्रोबायोटिक्स डालें, और चारा 35% कम करें।"
    },
    {
        "keywords": ["toka kullu", "fin rot", "bacterial", "dropsy", "pundlu", "తోక కుళ్ళు", "పుండ్లు", "घाव", "सड़न"],
        "reply_english": "🐟 **Fish Fin/Tail Rot and Ulcer Guidance:**\n1. Check water quality and reduce environmental stress.\n2. Remove or isolate clearly affected fish where appropriate.\n3. Review feeding and pond hygiene.\n4. Confirm the cause with an aquaculture professional before using antibiotics or other disease-specific treatments.",
        "reply_telenglish": "🐟 **Chepalaki Toka Kullu & Pundlu Treatment:**\n1. **Water Sanitizer:** Neellalo Potassium Permanganate 2.5 ppm leda BKC 1 liter/acre veyandi.\n2. **Meethalo Mandu:** Oxytetracycline (OTC) powder 65mg per kg fish weight meethalo kalipi 7 rojulu ivvandi.\n3. **Meetha Padhati:** Meethanu 40% thagginchandi. Egg white tho mandu kalipi needalo aarabetti ivvandi.",
        "reply_telugu": "🐟 **చేపలకు తోక కుళ్ళు & పుండ్ల చికిత్స:**\n1. **నీటి మందు:** నీటిలో పొటాషియం పర్మాంగనేట్ 2.5 ppm లేదా BKC 1 లీటరు వేయండి.\n2. **మేతలో మందు:** ఆక్సిటెట్రాసైక్లిన్ 65 mg కేజీ చేపల బరువుకు 7 రోజులు ఇవ్వండి.\n3. **మేత పద్ధతి:** మేతను 40% తగ్గించండి. గుడ్డు తెల్లసొనతో మందు కలిపి ఇవ్వండి.",
        "reply_hindi": "🐟 **मछलियों में पूंछ सड़न और घाव उपचार:**\n1. **जल सैनिटाइज़र:** पानी में 2.5 ppm पोटैशियम परमैंगनेट या 1 लीटर BKC डालें।\n2. **चारा एंटीबायोटिक:** 65 मिलीग्राम प्रति किलो मछली वजन से 7 दिन ऑक्सीटेट्रासाइक्लिन दें।\n3. **चारा विधि:** चारा 40% कम करें।",
        "voice_telugu": "చేపలకు తోక కుళ్ళు వ్యాధికి నీటిలో పొటాషియం పర్మాంగనేట్ వేయండి. మేతలో ఆక్సిటెట్రాసైక్లిన్ అరవై అయిదు మిల్లీగ్రాములు కలిపి ఏడు రోజులు ఇవ్వండి.",
        "voice_hindi": "मछली में घाव होने पर पानी में पोटैशियम परमैंगनेट डालें और 7 दिन चारे में ऑक्सीटेट्रासाइक्लिन दें।"
    }
]

def detect_question_language(text: str, preferred: str = "te") -> str:
    """Detect script when possible, otherwise respect the selected UI language.

    Telugu Unicode range: 0C00-0C7F
    Devanagari (Hindi) range: 0900-097F
    Latin-script input follows the user's selected interface language.
    """
    value = str(text or "")
    if any("\u0c00" <= ch <= "\u0c7f" for ch in value):
        return "te"
    if any("\u0900" <= ch <= "\u097f" for ch in value):
        return "hi"
    preferred_language = str(preferred or "").strip().lower()
    return preferred_language if preferred_language in {"en", "te", "hi"} else "en"


def process_farmer_query(
    query: str,
    lang: str = "te",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    clean_query = str(query or "").lower().strip()
    reply_language = detect_question_language(clean_query, lang)
    result_context = context if isinstance(context, dict) else {}
    section = str(result_context.get("application_section", "")).lower()
    is_disease_section = section in {"fish-dis", "prawn-dis"}
    is_do_section = section in {"fish-do", "prawn-do"}
    app_terms = (
        "aquacare", "aqua care", "application", "website", "dashboard", "fish tab",
        "prawn tab", "fish disease scan", "prawn disease scan", "disease detection",
        "healthy result", "disease result", "upload photo", "image scan", "bounding box",
        "market rate", "selling rate", "do prediction", "oxygen prediction", "camera",
        "app result", "app feature", "application feature", "app help", "app error",
        "app language", "app model", "precautions in", "medicine in the app",
        "food guidance in", "how to use the app", "how does the app", "what does the app",
        "what can this app", "open the app", "use this app", "how to use aquacare",
        "how does this work", "how do i use this", "how to use this", "what are the features",
        "what can i ask", "help me use", "app about",
        "చేపల ట్యాబ్", "రొయ్యల ట్యాబ్", "చేపల వ్యాధి స్కాన్", "రొయ్యల వ్యాధి స్కాన్",
        "వ్యాధి ఫలితం", "ఆరోగ్య ఫలితం", "ఫోటో అప్లోడ్", "ఫోటో అప్‌లోడ్", "బౌండింగ్ బాక్స్",
        "మార్కెట్ ధర", "అమ్మకపు ధర", "డీఓ అంచనా", "కెమెరా", "యాప్", "అప్లికేషన్",
        "ఈ యాప్", "ఈ అప్లికేషన్", "యాప్ ఎలా", "యాప్ గురించి", "యాప్‌లో",
        "मछली टैब", "झींगा टैब", "रोग परिणाम", "स्वस्थ परिणाम", "फोटो अपलोड",
        "बाउंडिंग बॉक्स", "बाजार दर", "DO अनुमान", "ऐप", "एप्लिकेशन", "इस ऐप",
        "ऐप के बारे", "ऐप में", "ऐप कैसे",
    )
    is_app_question = (
        result_context.get("type") in {"Disease detection", "DO prediction"}
        or bool(re.search(r"\b(app|website)\b", clean_query))
        or any(term in clean_query for term in app_terms)
        or (
            is_disease_section
            and any(term in clean_query for term in (
                "disease", "medicine", "food", "feed", "healthy", "result", "precaution",
                "chepa", "royya", "vyadhi", "mandu", "mandhulu", "meetha", "result",
                "cheppandi", "em cheyyali", "చేప", "రొయ్య", "వ్యాధి", "మందు", "మేత",
                "ఫలితం", "फिश", "झींगा", "रोग", "दवा",
            ))
        )
        or (
            is_do_section
            and any(term in clean_query for term in (
                "oxygen", "do", "water", "prediction", "result", "neellu", "neeti",
                "cheppandi", "em cheyyali", "ఆక్సిజన్", "నీరు", "ऑक्सीजन", "पानी",
            ))
        )
    )

    if reply_language == "te":
        if is_app_question:
            telugu = (
                "AquaCare యాప్‌లో Fish/Prawn‌ను వేర్వేరు tabs‌లో ఎంచుకుని ఫోటో upload లేదా cameraతో "
                "scan చేయండి. వ్యాధి result‌లో గుర్తించిన ప్రాంతం, సూచనలు, మందులు మరియు ఆహార సమాచారం "
                "చూడవచ్చు; Healthy result‌లో సూచిక market-rate range కనిపిస్తుంది. DO tab నీటి వివరాల "
                "ఆధారంగా అంచనా ఇస్తుంది. ఇవి screening/reference మాత్రమే; మందులు, ధరలను స్థానిక "
                "మత్స్య నిపుణుడు/మార్కెట్‌తో నిర్ధారించండి."
            )
            if result_context.get("type") == "Disease detection":
                telugu += (
                    f"\nప్రస్తుత ఫలితం: {result_context.get('title') or result_context.get('disease', 'తెలియదు')} "
                    f"({result_context.get('confidence', 'N/A')})."
                )
                for key, label in (
                    ("precautions", "ముందుజాగ్రత్తలు"),
                    ("medicine", "మందుల సూచన"),
                    ("food", "ఆహారం/మేత"),
                ):
                    if result_context.get(key):
                        telugu += f"\n{label}: {result_context[key]}"
                if result_context.get("market_rates"):
                    telugu += f"\nధరలు: {result_context['market_rates']}"
            elif result_context.get("type") == "DO prediction":
                telugu += (
                    f"\nప్రస్తుత DO అంచనా: {result_context.get('predicted_do_mg_l', 'N/A')} mg/L."
                )
                for key, label in (
                    ("status", "స్థితి"),
                    ("mortality_risk", "మరణ ప్రమాదం"),
                    ("advice", "సలహా"),
                    ("aerator", "ఎరేటర్"),
                    ("feed", "మేత"),
                ):
                    if result_context.get(key):
                        telugu += f"\n{label}: {result_context[key]}"
        else:
            telugu = (
                "నేను AquaCare అప్లికేషన్ గురించి మాత్రమే సహాయం చేయగలను—దాని Fish/Prawn tabs, "
                "ఫోటో scan, వ్యాధి/ఆరోగ్య ఫలితాలు, market-rate guide, DO prediction లేదా app వాడే విధానం "
                "గురించి అడగండి."
            )
        english = (
            "I can help only with the AquaCare application: its Fish/Prawn tabs, image scanning, "
            "disease/healthy results, market-rate guide, DO prediction, and how to use the app."
        )
        hindi = "मैं केवल AquaCare ऐप के उपयोग और उसके Fish/Prawn, disease, market-rate तथा DO परिणामों में मदद कर सकता हूँ।"
    elif reply_language == "hi":
        if is_app_question:
            hindi = (
                "AquaCare में अलग Fish/Prawn टैब चुनकर फोटो अपलोड करें या कैमरा से स्कैन करें। रोग "
                "परिणाम में पहचाना गया क्षेत्र, सावधानियां, दवा और भोजन की जानकारी मिलती है; Healthy "
                "परिणाम में संकेतात्मक बाजार-दर दिखाई जाती है। DO टैब पानी के दर्ज मानों से अनुमान देता है। "
                "ये screening/reference जानकारी है—दवा और कीमत स्थानीय मत्स्य विशेषज्ञ/बाजार से पुष्टि करें।"
            )
            if result_context.get("type") == "Disease detection":
                hindi += (
                    f"\nवर्तमान परिणाम: {result_context.get('title') or result_context.get('disease', 'अज्ञात')} "
                    f"({result_context.get('confidence', 'N/A')})."
                )
                for key, label in (
                    ("precautions", "सावधानियां"),
                    ("medicine", "दवा संबंधी जानकारी"),
                    ("food", "भोजन/चारा"),
                ):
                    if result_context.get(key):
                        hindi += f"\n{label}: {result_context[key]}"
                if result_context.get("market_rates"):
                    hindi += f"\nदरें: {result_context['market_rates']}"
            elif result_context.get("type") == "DO prediction":
                hindi += f"\nवर्तमान DO अनुमान: {result_context.get('predicted_do_mg_l', 'N/A')} mg/L."
                for key, label in (
                    ("status", "स्थिति"),
                    ("mortality_risk", "मृत्यु जोखिम"),
                    ("advice", "सलाह"),
                    ("aerator", "एरेटर"),
                    ("feed", "चारा"),
                ):
                    if result_context.get(key):
                        hindi += f"\n{label}: {result_context[key]}"
        else:
            hindi = (
                "मैं केवल AquaCare ऐप के बारे में मदद कर सकता हूँ—Fish/Prawn टैब, फोटो स्कैन, "
                "रोग/स्वस्थ परिणाम, बाजार-दर, DO अनुमान या ऐप उपयोग के बारे में पूछें।"
            )
        english = "I can help only with the AquaCare application and its Fish/Prawn, disease, market-rate, and DO features."
        telugu = "నేను AquaCare అప్లికేషన్‌కి సంబంధించిన విషయాల్లో మాత్రమే సహాయం చేయగలను."
    else:
        if is_app_question:
            english = (
                "In AquaCare, choose the separate Fish or Prawn tab and upload/capture a photo to "
                "screen it. Disease results show the detected region, precautions, medicine and food "
                "guidance; healthy results show indicative selling-rate ranges. The DO tab estimates "
                "oxygen from entered water values. These are screening/reference outputs, not a "
                "confirmed diagnosis or guaranteed live price."
            )
            if result_context.get("type") == "Disease detection":
                english += (
                    f"\nCurrent result: {result_context.get('title') or result_context.get('disease', 'Unknown')} "
                    f"({result_context.get('confidence', 'N/A')})."
                )
                for key, label in (
                    ("precautions", "Precautions"),
                    ("medicine", "Medicine guidance"),
                    ("food", "Food/feed"),
                ):
                    if result_context.get(key):
                        english += f"\n{label}: {result_context[key]}"
                if result_context.get("market_rates"):
                    english += f"\nIndicative rates: {result_context['market_rates']}"
            elif result_context.get("type") == "DO prediction":
                english += (
                    f"\nCurrent DO estimate: {result_context.get('predicted_do_mg_l', 'N/A')} mg/L."
                )
                for key, label in (
                    ("status", "Status"),
                    ("mortality_risk", "Mortality risk"),
                    ("advice", "Advice"),
                    ("aerator", "Aerator guidance"),
                    ("feed", "Feed guidance"),
                ):
                    if result_context.get(key):
                        english += f"\n{label}: {result_context[key]}"
        else:
            english = (
                "I can help only with the AquaCare application: its Fish/Prawn tabs, image scanning, "
                "disease/healthy results, market-rate guide, DO prediction, or how to use the app."
            )
        telugu = "AquaCare యాప్‌లో Fish/Prawn tabs, ఫోటో scan, disease/healthy results, market rates, DO prediction లేదా app వాడకం గురించి అడగండి."
        hindi = "AquaCare ऐप के Fish/Prawn टैब, फोटो स्कैन, रोग/स्वस्थ परिणाम, बाजार-दर, DO अनुमान या ऐप उपयोग के बारे में पूछें।"

    if is_app_question and result_context.get("type") not in {"Disease detection", "DO prediction"}:
        asks_for_rates = any(term in clean_query for term in (
            "rate", "price", "market", "selling", "ధర", "మార్కెట్", "कीमत", "दर",
        ))
        if asks_for_rates:
            if any(term in clean_query for term in ("prawn", "royya", "రొయ్య", "झींगा")):
                rate_species = ("prawn",)
            elif any(term in clean_query for term in ("fish", "chepa", "చేప", "मछली")):
                rate_species = ("fish",)
            else:
                rate_species = ("fish", "prawn")
            rate_details = "\n".join(
                f"{species.capitalize()} — {item}: {rate['price_range']}"
                for species in rate_species
                for item, rate in MARKET_SELLING_RATES[species].items()
            )
            if reply_language == "te":
                telugu += (
                    f"\nసూచిక అమ్మకపు ధరలు:\n{rate_details}\n"
                    "ఇవి live quotes కావు; స్థానిక కొనుగోలుదారులతో ఈరోజు ధరను నిర్ధారించండి."
                )
            elif reply_language == "hi":
                hindi += (
                    f"\nसंकेतात्मक बिक्री दरें:\n{rate_details}\n"
                    "ये live quotes नहीं हैं; आज की कीमत स्थानीय खरीदार से पुष्टि करें।"
                )
            else:
                english += (
                    f"\nIndicative selling rates:\n{rate_details}\n"
                    "These are not live quotes; confirm today's price with local buyers."
                )
        elif any(term in clean_query for term in ("disease detection", "disease scan", "వ్యాధి", "रोग पहचान")):
            if reply_language == "te":
                telugu += (
                    "\nFish categories: Healthy Fish, Bacterial, Parasitic, White Tail, Fungal. "
                    "Prawn categories: Healthy, WSSV, Black Gill, and WSSV + Black Gill."
                )
            elif reply_language == "hi":
                hindi += (
                    "\nFish categories: Healthy Fish, Bacterial, Parasitic, White Tail, Fungal. "
                    "Prawn categories: Healthy, WSSV, Black Gill, and WSSV + Black Gill."
                )
            else:
                english += (
                    "\nFish categories: Healthy Fish, Bacterial, Parasitic, White Tail, and Fungal. "
                    "Prawn categories: Healthy, WSSV, Black Gill, and WSSV + Black Gill."
                )
        elif any(term in clean_query for term in ("do prediction", "oxygen prediction", "DO estimate", "ఆక్సిజన్", "DO అంచనా")):
            if reply_language == "te":
                telugu += "\nDO tab‌లో నీటి విలువలను నమోదు చేసి estimate పొందండి; ఇది కొలిచిన meter reading కాదు."
            elif reply_language == "hi":
                hindi += "\nDO टैब में पानी के मान भरकर अनुमान लें; यह meter की वास्तविक reading नहीं है।"
            else:
                english += "\nEnter pond-water values in the DO tab for an estimate; it is not a live meter reading."

    if is_app_question and result_context.get("type") == "DO prediction":
        value = result_context.get("predicted_do_mg_l", "N/A")
        status = result_context.get("status", "N/A")
        risk = result_context.get("mortality_risk", "N/A")
        advice = result_context.get("advice", "")
        aerator = result_context.get("aerator", "")
        feed = result_context.get("feed", "")
        english = (
            f"Your {result_context.get('species', 'pond')} DO estimate is {value} mg/L. "
            f"App status: {status}. Mortality risk: {risk}."
        )
        if advice:
            english += f"\nRecommended action: {advice}"
        if aerator:
            english += f"\nAeration: {aerator}"
        if feed:
            english += f"\nFeeding: {feed}"
        english += "\nThis is an estimate from the values entered in AquaCare, not a live meter reading."

        telugu = (
            f"AquaCare అంచనా DO: {value} mg/L. యాప్ స్థితి: {status}. మరణ ప్రమాదం: {risk}."
        )
        if advice:
            telugu += f"\nసూచించిన చర్య: {advice}"
        if aerator:
            telugu += f"\nఎరేషన్: {aerator}"
        if feed:
            telugu += f"\nమేత: {feed}"
        telugu += "\nఇది నమోదు చేసిన నీటి విలువల ఆధారంగా వచ్చిన అంచనా మాత్రమే; live meter reading కాదు."

        hindi = (
            f"AquaCare DO अनुमान: {value} mg/L. ऐप स्थिति: {status}. मृत्यु जोखिम: {risk}."
        )
        if advice:
            hindi += f"\nसुझाई गई कार्रवाई: {advice}"
        if aerator:
            hindi += f"\nएरेशन: {aerator}"
        if feed:
            hindi += f"\nचारा: {feed}"
        hindi += "\nयह दर्ज किए गए पानी के मानों से निकला अनुमान है, live meter reading नहीं।"

    elif is_app_question and result_context.get("type") == "Disease detection":
        disease = result_context.get("title") or result_context.get("disease", "N/A")
        confidence = result_context.get("confidence", "N/A")
        severity = result_context.get("severity", "N/A")
        precautions = result_context.get("precautions", "")
        medicine = result_context.get("medicine", "")
        food = result_context.get("food", "")
        market_rates = result_context.get("market_rates", "")
        regions = result_context.get("affected_regions") or []

        english = (
            f"AquaCare {result_context.get('species', 'animal')} image-screening result: "
            f"{disease}; confidence {confidence}; severity {severity}."
        )
        if regions:
            english += "\nDetected image region(s): " + "; ".join(
                str(region.get("where", "region")) for region in regions if isinstance(region, dict)
            )
        if precautions:
            english += f"\nPrecautions: {precautions}"
        if medicine:
            english += f"\nMedicine guidance: {medicine}"
        if food:
            english += f"\nFood/feed guidance: {food}"
        if market_rates:
            english += f"\nIndicative selling rates: {market_rates}"
        english += "\nThis is an AI screening result, not a confirmed diagnosis. Confirm treatment with a qualified fisheries professional."

        telugu = (
            f"AquaCare {result_context.get('species', 'జలచరం')} చిత్రం-ఆధారిత ఫలితం: "
            f"{disease}; confidence {confidence}; తీవ్రత {severity}."
        )
        if regions:
            telugu += "\nగుర్తించిన చిత్ర ప్రాంతం: " + "; ".join(
                str(region.get("where", "ప్రాంతం")) for region in regions if isinstance(region, dict)
            )
        if precautions:
            telugu += f"\nముందుజాగ్రత్తలు: {precautions}"
        if medicine:
            telugu += f"\nమందుల సూచన: {medicine}"
        if food:
            telugu += f"\nఆహారం/మేత: {food}"
        if market_rates:
            telugu += f"\nసూచిక అమ్మకపు ధరలు: {market_rates}"
        telugu += "\nఇది AI screening మాత్రమే, నిర్ధారిత diagnosis కాదు. చికిత్సకు ముందు మత్స్య నిపుణుడితో నిర్ధారించండి."

        hindi = (
            f"AquaCare {result_context.get('species', 'जीव')} image-screening परिणाम: "
            f"{disease}; confidence {confidence}; गंभीरता {severity}."
        )
        if regions:
            hindi += "\nपहचाना गया image क्षेत्र: " + "; ".join(
                str(region.get("where", "क्षेत्र")) for region in regions if isinstance(region, dict)
            )
        if precautions:
            hindi += f"\nसावधानियां: {precautions}"
        if medicine:
            hindi += f"\nदवा संबंधी जानकारी: {medicine}"
        if food:
            hindi += f"\nभोजन/चारा: {food}"
        if market_rates:
            hindi += f"\nसंकेतात्मक बिक्री दरें: {market_rates}"
        hindi += "\nयह AI screening है, पुष्टि किया हुआ diagnosis नहीं। उपचार से पहले योग्य मत्स्य विशेषज्ञ से पुष्टि करें।"

    selected = {"en": english, "te": telugu, "hi": hindi}[reply_language]
    return {
        "reply_language": reply_language,
        "reply_english": english,
        "reply_telugu": telugu,
        "reply_hindi": hindi,
        "reply": selected,
        "voice_text": selected,
    }

# ==============================================================================
# 7. FRONTEND DASHBOARD TEMPLATE WITH LIVE CAMERA, MULTILINGUAL ENGINE
# ==============================================================================
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AquaCare AI Pro • Fish & Prawn Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

  <style>
    :root {
      --bg: #030d14;
      --card-bg: rgba(9, 29, 41, 0.85);
      --card-border: rgba(56, 189, 248, 0.25);
      --primary: #00f0ff;
      --primary-glow: rgba(0, 240, 255, 0.4);
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #ef4444;
      --text: #f0fdfa;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(0, 240, 255, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.06) 0px, transparent 50%);
      color: var(--text);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
    }
    h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; }

    header {
      background: rgba(3, 13, 20, 0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 14px 24px;
    }
    .header-content {
      max-width: 1320px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 15px;
    }
    .logo-area { display: flex; align-items: center; gap: 12px; }
    .logo-icon {
      font-size: 2rem;
      background: linear-gradient(135deg, #00f0ff, #10b981);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .logo-text h1 {
      font-size: 1.4rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .logo-text p { font-size: 0.78rem; color: var(--text-muted); }

    .lang-switcher-box {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(0, 240, 255, 0.08);
      border: 1px solid var(--primary);
      border-radius: 12px;
      padding: 6px 14px;
      box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
    }
    .lang-switcher-box select {
      background: transparent;
      border: none;
      color: var(--primary);
      font-weight: 700;
      font-size: 0.94rem;
      cursor: pointer;
      outline: none;
      padding: 2px 4px;
    }
    .lang-switcher-box select option {
      background: #092332;
      color: white;
    }

    .container {
      max-width: 1320px;
      margin: 25px auto;
      padding: 0 20px;
    }

    .tabs-nav {
      display: flex;
      gap: 10px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
      margin-bottom: 24px;
      overflow-x: auto;
    }
    .tab-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 10px 18px;
      border-radius: 12px;
      cursor: pointer;
      font-family: inherit;
      font-size: 0.92rem;
      font-weight: 600;
      transition: all 0.25s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .tab-btn:hover {
      background: rgba(0, 240, 255, 0.1);
      color: var(--text);
      border-color: var(--primary);
    }
    .tab-btn.active {
      background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(16, 185, 129, 0.2));
      border-color: var(--primary);
      color: var(--primary);
      box-shadow: 0 0 15px var(--primary-glow);
    }

    .tab-pane {
      display: none;
      animation: fadeIn 0.3s ease;
    }
    .tab-pane.active { display: block; }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    .card-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--primary);
    }
    .card-subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 20px;
    }

    .form-group { margin-bottom: 16px; }
    .form-group label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      margin-bottom: 6px;
      color: var(--text);
    }
    .form-group input, .form-group select {
      width: 100%;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 0.95rem;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }
    .form-group input:focus, .form-group select:focus {
      border-color: var(--primary);
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .drop-zone {
      border: 2px dashed var(--card-border);
      border-radius: 14px;
      padding: 24px;
      text-align: center;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.02);
      transition: all 0.25s ease;
      margin-bottom: 16px;
    }
    .drop-zone:hover, .drop-zone.dragover {
      border-color: var(--primary);
      background: rgba(0, 240, 255, 0.06);
    }
    .drop-zone i {
      font-size: 2.2rem;
      color: var(--accent);
      margin-bottom: 8px;
    }

    .preview-box {
      margin-bottom: 16px;
      text-align: center;
      display: none;
      position: relative;
    }
    .preview-box img {
      max-height: 260px;
      max-width: 100%;
      border-radius: 12px;
      border: 1px solid var(--card-border);
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 20px;
      border-radius: 12px;
      font-weight: 700;
      font-size: 0.95rem;
      font-family: inherit;
      cursor: pointer;
      border: none;
      transition: all 0.2s ease;
      width: 100%;
    }
    .btn-primary {
      background: linear-gradient(135deg, #00f0ff, #0ea5e9);
      color: #030d14;
      box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
    }
    .btn-primary:hover {
      box-shadow: 0 0 25px rgba(0, 240, 255, 0.6);
      transform: translateY(-1px);
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: var(--text);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.15);
      border-color: var(--primary);
    }

    .result-panel { display: none; }
    .result-badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .result-badge.safe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .result-badge.warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .result-badge.danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

    .annotated-container {
      margin-top: 14px;
      margin-bottom: 14px;
      border-radius: 12px;
      overflow: hidden;
      border: 2px solid var(--card-border);
      background: #000;
      text-align: center;
      display: none;
    }
    .annotated-container img {
      width: 100%;
      max-height: 380px;
      object-fit: contain;
      display: block;
    }

    .spots-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 16px;
      margin-bottom: 14px;
      display: none;
    }
    .spot-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      font-size: 0.88rem;
    }
    .spot-item:last-child { border-bottom: none; }

    .action-card {
      background: rgba(255, 255, 255, 0.03);
      border-radius: 12px;
      padding: 16px;
      margin-top: 14px;
    }
    .action-card h4 {
      font-size: 1rem;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn-speak {
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid var(--primary);
      color: var(--primary);
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-speak:hover {
      background: var(--primary);
      color: #030d14;
    }

    .btn-buy {
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
      border: none;
      padding: 6px 14px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.82rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: transform 0.2s;
    }
    .btn-buy:hover { transform: scale(1.03); }

    .catalog-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }
    .product-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.2s ease;
    }
    .product-card:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 240, 255, 0.15);
    }

    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .modal-card {
      background: #092332;
      border: 1px solid var(--primary);
      border-radius: 18px;
      padding: 24px;
      width: 100%;
      max-width: 480px;
      position: relative;
    }

    /* Medicine & Feed Provider Website Links */
    .provider-links-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .provider-link-pill {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 8px 12px;
      color: #e2e8f0;
      text-decoration: none;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      text-align: left;
      width: 100%;
      box-sizing: border-box;
      font-family: inherit;
    }
    .provider-link-pill:hover {
      background: rgba(0, 240, 255, 0.12);
      border-color: var(--primary);
      color: #ffffff;
      transform: translateY(-2px);
      box-shadow: 0 4px 14px rgba(0, 240, 255, 0.2);
    }
    .provider-link-pill.store-pill {
      background: rgba(16, 185, 129, 0.08);
      border-color: rgba(16, 185, 129, 0.3);
    }
    .provider-link-pill.store-pill:hover {
      background: rgba(16, 185, 129, 0.2);
      border-color: #10b981;
      box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }
    .provider-pill-badge {
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .badge-external {
      background: rgba(0, 240, 255, 0.15);
      color: var(--primary);
      border: 1px solid rgba(0, 240, 255, 0.3);
    }
    .badge-store-order {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-govt {
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    @media (max-width: 768px) {
      .header-content { flex-direction: column; align-items: flex-start; }
      .form-row { grid-template-columns: 1fr; }
      .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

  <!-- Top Header with English / Telugu / Hindi Selector -->
  <header>
    <div class="header-content">
      <div class="logo-area">
        <i class="fa-solid fa-water logo-icon"></i>
        <div class="logo-text">
          <h1>AquaCare AI Pro <span style="font-size:0.75rem; color:#38bdf8; font-weight:600; padding:2px 8px; background:rgba(56,189,248,0.15); border-radius:12px;">v4.2 SELECTED-MODEL-FIRST</span></h1>
          <p id="ui-subheading" class="t-key" data-k="header_sub">Fish & Prawn AI Platform • Live Camera • Disease & Feeding • Multilingual</p>
        </div>
      </div>

      <!-- Main UI language: English / Telugu / Hindi only -->
      <div class="lang-switcher-box">
        <i class="fa-solid fa-language" style="color:var(--primary); font-size:1.15rem;"></i>
        <select id="lang-select" onchange="switchLanguage(this.value)">
          <option value="te" selected>🇮🇳 తెలుగు (Telugu)</option>
          <option value="en">🇬🇧 English</option>
          <option value="hi">🇮🇳 हिन्दी (Hindi)</option>
        </select>
      </div>

      <div>
        <span style="font-size:0.8rem; padding:5px 12px; border-radius:20px; background:rgba(16,185,129,0.15); color:#34d399; border:1px solid #10b981;">
          <i class="fa-solid fa-shield-halved"></i> <span class="t-key" data-k="guard_badge">Fish & Prawn Guard Active</span>
        </span>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <div class="container">

    <!-- Tab Navigation -->
    <div class="tabs-nav">
      <button class="tab-btn active" id="tab-btn-detect" onclick="switchTab('tab-detection')">
        <i class="fa-solid fa-microscope"></i> <span class="t-key" data-k="tab_detection">వ్యాధి గుర్తింపు & కెమెరా</span>
      </button>
      <button class="tab-btn" id="tab-btn-do" onclick="switchTab('tab-do')">
        <i class="fa-solid fa-droplet"></i> <span class="t-key" data-k="tab_do">ఆక్సిజన్ & నీటి నాణ్యత</span>
      </button>
      <button class="tab-btn" id="tab-btn-assistant" onclick="switchTab('tab-assistant')">
        <i class="fa-solid fa-headset"></i> <span class="t-key" data-k="tab_assistant">రైతు వాయిస్ అసిస్టెంట్</span>
      </button>
      <button class="tab-btn" id="tab-btn-selling" onclick="switchTab('tab-selling')">
        <i class="fa-solid fa-hand-holding-dollar"></i> <span class="t-key" data-k="tab_selling">అమ్మకాలు & మార్కెట్ ధరలు</span>
      </button>
      <button class="tab-btn" id="tab-btn-buying" onclick="switchTab('tab-buying')">
        <i class="fa-solid fa-cart-shopping"></i> <span class="t-key" data-k="tab_buying">మందులు కొనడానికి (Buy Hub)</span>
      </button>
    </div>

    <!-- ======================================================================= -->
    <!-- TAB 1: DISEASE DETECTION + LIVE CAMERA                                 -->
    <!-- ======================================================================= -->
    <div id="tab-detection" class="tab-pane active">
      <div class="grid-2">

        <!-- Upload & Live Camera Form Card -->
        <div class="card">
          <h2 class="card-title">
            <i class="fa-solid fa-camera"></i> <span class="t-key" data-k="detect_title">ఫోటో అప్‌లోడ్ చేయండి లేదా లైవ్ కెమెరా వాడండి</span>
          </h2>
          <p class="card-subtitle t-key" data-k="detect_sub">
            కేవలం చేప మరియు రొయ్యల ఫోటోలు మాత్రమే అనుమతించబడతాయి. వ్యాధి ఉన్న ప్రదేశంపై బౌండింగ్ బాక్స్‌లు వస్తాయి.
          </p>

          <!-- FRONT & BACK CAMERA BUTTONS -->
          <div style="display:flex; gap:10px; margin-bottom:14px; flex-wrap:wrap;">
            <button type="button" class="btn btn-secondary" id="btn-front-cam" onclick="startCameraMode('user')" style="width:auto; border-color:#00f0ff; color:#00f0ff;">
              <i class="fa-solid fa-user"></i> <span class="t-key" data-k="open_front_cam">🤳 ఫ్రంట్ కెమెరా (Webcam)</span>
            </button>
            <button type="button" class="btn btn-secondary" id="btn-back-cam" onclick="startCameraMode('environment')" style="width:auto; border-color:#10b981; color:#34d399;">
              <i class="fa-solid fa-camera-rotate"></i> <span class="t-key" data-k="open_back_cam">📷 బ్యాక్ / వెబ్‌క్యామ్ కెమెరా (చేప / రొయ్య)</span>
            </button>
            <button type="button" class="btn btn-secondary" id="flip-camera-btn" onclick="flipCamera()" style="display:none; width:auto;" title="Flip Camera">
              <i class="fa-solid fa-arrows-rotate"></i> <span class="t-key" data-k="flip_camera">కెమెరా మార్చండి</span>
            </button>
          </div>

          <!-- Camera device dropdown if multiple webcams -->
          <div id="camera-select-container" style="display:none; margin-bottom:12px;">
            <label style="font-size:0.8rem; color:var(--text-muted); display:block; margin-bottom:4px;">
              <i class="fa-solid fa-video"></i> Select Video Device / వెబ్‌క్యామ్ ఎంచుకోండి:
            </label>
            <select id="camera-device-select" onchange="switchCameraDevice(this.value)"></select>
          </div>

          <!-- Live Camera Stream Viewport -->
          <div id="camera-container" style="display:none; margin-bottom:15px; border-radius:14px; overflow:hidden; border:2px solid var(--primary); background:#000; text-align:center; position:relative;">
            <!-- Active Camera Badge Indicator -->
            <div id="camera-status-pill" style="position:absolute; top:12px; left:12px; z-index:10; background:rgba(3,13,20,0.85); border:1px solid var(--primary); padding:4px 12px; border-radius:20px; font-size:0.82rem; font-weight:700; color:var(--primary);">
              <i class="fa-solid fa-circle" style="color:#10b981; font-size:0.6rem; margin-right:5px;"></i>
              <span id="camera-status-text">Camera Active</span>
            </div>

            <video id="camera-stream" autoplay playsinline muted style="width:100%; max-height:360px; object-fit:cover;"></video>
            <div id="camera-message" style="padding:8px 12px; color:#c6eef2; font-size:0.84rem; background:rgba(3,13,20,0.9);">
              కెమెరా ఆన్ లో ఉంది. చేప లేదా రొయ్యను స్పష్టంగా చూపించండి — Live AI Scan ఆటోమేటిక్‌గా జరుగుతుంది.
            </div>
            <canvas id="camera-canvas" style="display:none;"></canvas>

            <div style="padding:10px; background:rgba(3,13,20,0.92); display:flex; justify-content:center; gap:12px; flex-wrap:wrap;">
              <button type="button" class="btn btn-primary" onclick="capturePhoto()" style="width:auto; padding:10px 24px;">
                <i class="fa-solid fa-circle-dot"></i> <span class="t-key" data-k="capture_photo">Snapshot</span>
              </button>
              <button type="button" class="btn btn-secondary" onclick="stopCamera()" style="width:auto; padding:10px 18px;">
                <i class="fa-solid fa-xmark"></i> <span class="t-key" data-k="close_camera">కెమెరా ఆపివేయండి</span>
              </button>
            </div>
          </div>

          <form id="disease-form" action="javascript:void(0);" onsubmit="submitDiseaseDetection(event); return false;">
            <div class="form-group">
              <label class="t-key" data-k="species_label">జాతి ఎంపిక</label>
              <select id="species-select" name="species">
                <option value="fish" selected>🐟 చేప (Fish)</option>
                <option value="prawn">🦐 రొయ్య (Prawn / Shrimp)</option>
              </select>
            </div>

            <!-- Quick Test Sample Photos -->
            <div style="margin-bottom:14px; background:rgba(0,240,255,0.04); border:1px dashed var(--card-border); border-radius:12px; padding:10px 14px;">
              <span style="font-size:0.8rem; color:var(--primary); font-weight:700; display:block; margin-bottom:6px;" class="t-key" data-k="quick_samples_title">
                ⚡ శాంపిల్ ఫోటోలు (వెంటనే టెస్ట్ చేయడానికి):
              </span>
              <div style="display:flex; flex-wrap:wrap; gap:8px;">
                <button type="button" class="btn btn-secondary" onclick="loadSample('fish_healthy', 'fish')" style="padding:6px 12px; font-size:0.8rem; width:auto; border-radius:16px;">
                  <span class="t-key" data-k="sample_healthy_fish">🐟 ఆరోగ్యకరమైన చేప</span>
                </button>
                <button type="button" class="btn btn-secondary" onclick="loadSample('fish_bacterial', 'fish')" style="padding:6px 12px; font-size:0.8rem; width:auto; border-radius:16px;">
                  <span class="t-key" data-k="sample_bacterial">🐟 బాక్టీరియల్ పుండ్లు</span>
                </button>
                <button type="button" class="btn btn-secondary" onclick="loadSample('prawn_wssv', 'prawn')" style="padding:6px 12px; font-size:0.8rem; width:auto; border-radius:16px;">
                  <span class="t-key" data-k="sample_wssv">🦐 తెల్ల మచ్చ వైరస్</span>
                </button>
                <button type="button" class="btn btn-secondary" onclick="loadSample('prawn_blackgill', 'prawn')" style="padding:6px 12px; font-size:0.8rem; width:auto; border-radius:16px;">
                  <span class="t-key" data-k="sample_blackgill">🦐 నల్ల మొప్ప రొయ్య</span>
                </button>
              </div>
            </div>

            <!-- Drag and Drop Zone -->
            <div class="drop-zone" id="drop-zone">
              <i class="fa-solid fa-cloud-arrow-up"></i>
              <h3 style="font-size:1.05rem; margin-bottom:4px;" class="t-key" data-k="drag_title">చేప లేదా రొయ్య ఫోటో ఇక్కడ వేయండి</h3>
              <p style="font-size:0.8rem; color:var(--text-muted);" class="t-key" data-k="drag_sub">లేదా ఫైల్ ఎంచుకోవడానికి ఇక్కడ క్లిక్ చేయండి (JPG, PNG, WEBP)</p>
              <input type="file" id="image-input" name="image" accept="image/*" style="display:none;">
            </div>

            <div class="preview-box" id="preview-box">
              <img id="preview-img" src="" alt="Preview">
            </div>

            <button type="button" class="btn btn-primary" id="detect-btn" onclick="submitDiseaseDetection(event)">
              <i class="fa-solid fa-magnifying-glass"></i> <span class="t-key" data-k="detect_btn">వ్యాధి & ప్రదేశాన్ని గుర్తించండి</span>
            </button>
          </form>
        </div>

        <!-- Detection Results Card -->
        <div class="card" id="result-card">
          <h2 class="card-title">
            <i class="fa-solid fa-clipboard-check"></i> <span class="t-key" data-k="report_title">రోగ నిర్ధారణ & చికిత్స నివేదిక</span>
          </h2>
          <p class="card-subtitle t-key" data-k="report_sub">
            శరీరంపై వ్యాధి ఉన్న ఖచ్చితమైన ప్రదేశాలు, సరైన మందులు మరియు ఆహార సూచనలు.
          </p>

          <div id="detect-placeholder" style="text-align:center; padding:50px 20px; color:var(--text-muted);">
            <i class="fa-solid fa-fish-fins" style="font-size:3rem; margin-bottom:12px; opacity:0.3;"></i>
            <p><span class="t-key" data-k="placeholder_text">ఫోటో అప్‌లోడ్ చేయండి లేదా కెమెరాతో తీయండి.</span></p>
          </div>

          <div id="detect-loading" style="display:none; text-align:center; padding:40px 20px;">
            <i class="fa-solid fa-circle-notch fa-spin" style="font-size:2.5rem; color:var(--primary); margin-bottom:12px;"></i>
            <p style="font-weight:600;" class="t-key" data-k="scanning_text">YOLO బౌండింగ్ బాక్స్‌లతో స్కాన్ చేయబడుతోంది...</p>
          </div>

          <!-- SUCCESS RESULT PANEL -->
          <div class="result-panel" id="detect-result-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:10px;">
              <span id="result-status-badge" class="result-badge safe">Healthy</span>
              <button class="btn-speak" id="speak-diagnosis-btn">
                <i class="fa-solid fa-volume-high"></i> <span class="t-key" data-k="listen_btn">వాయిస్ వినండి (Listen)</span>
              </button>
            </div>

            <!-- Annotated Bounding Box Image from the selected YOLO disease-region box -->
            <div class="annotated-container" id="annotated-container">
              <img id="annotated-img" src="" alt="Annotated Image with Disease Bounding Boxes">
            </div>

            <!-- Exact Detected Spots List -->
            <div class="spots-card" id="spots-card">
              <h4 style="font-size:0.88rem; color:var(--primary); margin-bottom:8px;">
                <i class="fa-solid fa-location-crosshairs"></i> <span class="t-key" data-k="spots_title">శరీరంపై వ్యాధి ఉన్న ఖచ్చితమైన ప్రదేశాలు:</span>
              </h4>
              <div id="spots-list"></div>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;">
              <div style="background:rgba(255,255,255,0.04); padding:10px; border-radius:10px;">
                <label style="margin:0; font-size:0.75rem;" class="t-key" data-k="identified_class">గుర్తించిన వ్యాధి / స్థితి</label>
                <div id="res-class-name" style="font-weight:700; font-size:1.05rem; color:var(--primary);">-</div>
              </div>
              <div style="background:rgba(255,255,255,0.04); padding:10px; border-radius:10px;">
                <label style="margin:0; font-size:0.75rem;" class="t-key" data-k="confidence_score">ఖచ్చితత్వం (కాన్ఫిడెన్స్)</label>
                <div id="res-confidence" style="font-weight:700; font-size:1.05rem; color:var(--emerald);">-</div>
              </div>
            </div>

            <!-- DISEASE DETECTED: MANDHULU (MEDICINES) & FOOD (MEETHA) GUIDANCE -->
            <div id="disease-treatment-view" style="display:none;">
              <!-- 1. MANDHULU (MEDICINE) CARD -->
              <div class="action-card" style="border-left:4px solid var(--rose); background:rgba(239,68,68,0.06);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <h4 style="color:#f87171;">
                    <i class="fa-solid fa-pills"></i> <span class="t-key" data-k="med_title">మందులు & రసాయన చికిత్స (MANDHULU)</span>
                  </h4>
                  <button class="btn-buy" onclick="switchTab('tab-buying')" style="padding:4px 10px; font-size:0.75rem;">
                    <i class="fa-solid fa-cart-shopping"></i> <span class="t-key" data-k="buy_medicines_btn">మందులు కొనండి</span>
                  </button>
                </div>
                <p id="res-medicine-text" style="font-size:0.88rem; white-space:pre-line; margin-top:8px; line-height:1.5; color:var(--text);"></p>
                <!-- Medicine Purchase & Store Links -->
                <div id="res-medicine-links-box" style="display:none; margin-top:14px; border-top:1px dashed rgba(255,255,255,0.15); padding-top:10px;">
                  <div style="font-size:0.78rem; font-weight:700; color:#fca5a5; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-arrow-up-right-from-square"></i> <span class="t-key" data-k="med_links_title">మందులు లభించే అధికారిక వెబ్‌సైట్లు & స్టోర్ లింకులు:</span>
                  </div>
                  <div id="res-medicine-links-list" class="provider-links-grid"></div>
                </div>
              </div>

              <!-- 2. FOOD (MEETHA) CARD -->
              <div class="action-card" style="border-left:4px solid var(--amber); background:rgba(245,158,11,0.06); margin-top:12px;">
                <h4 style="color:#fbbf24;">
                  <i class="fa-solid fa-wheat-awn"></i> <span class="t-key" data-k="food_title">ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO)</span>
                </h4>
                <p id="res-food-text" style="font-size:0.88rem; white-space:pre-line; margin-top:8px; line-height:1.5; color:var(--text);"></p>
                <!-- Food Purchase & Brand Links -->
                <div id="res-food-links-box" style="display:none; margin-top:14px; border-top:1px dashed rgba(255,255,255,0.15); padding-top:10px;">
                  <div style="font-size:0.78rem; font-weight:700; color:#fde68a; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-arrow-up-right-from-square"></i> <span class="t-key" data-k="food_links_title">మేత & సప్లిమెంట్లు లభించే అధికారిక వెబ్‌సైట్లు:</span>
                  </div>
                  <div id="res-food-links-list" class="provider-links-grid"></div>
                </div>
              </div>
            </div>

            <!-- HEALTHY DETECTED: SELLING & HARVEST PRICING -->
            <div id="healthy-selling-view" style="display:none;">
              <div class="action-card" style="border-left:4px solid var(--emerald); background:rgba(16,185,129,0.08);">
                <h4 style="color:#34d399;">
                  <i class="fa-solid fa-circle-check"></i> <span class="t-key" data-k="sell_title">మార్కెట్ అమ్మకాలు & రాబడి వివరాలు</span>
                </h4>
                <p id="res-selling-status" style="font-weight:600; color:var(--emerald); margin-bottom:8px; font-size:0.9rem;"></p>
                <div id="res-selling-rates-list" style="margin-bottom:10px;"></div>
                <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">
                  <strong style="color:var(--accent); font-size:0.84rem;" class="t-key" data-k="harvest_tips_title">హార్వెస్టింగ్ & మార్కెట్ సలహాలు:</strong>
                  <p id="res-harvest-guide" style="font-size:0.86rem; color:var(--text-muted); white-space:pre-line; margin-top:4px;"></p>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:8px; margin-top:8px;">
                  <strong style="color:#fbbf24; font-size:0.84rem;" class="t-key" data-k="food_title">ఆహారం & మేత యాజమాన్యం (Food):</strong>
                  <p id="res-healthy-food" style="font-size:0.86rem; color:var(--text); white-space:pre-line; margin-top:4px;"></p>
                </div>
                <!-- Healthy Market Trading Links -->
                <div id="res-market-links-box" style="display:none; border-top:1px dashed rgba(255,255,255,0.15); padding-top:10px; margin-top:12px;">
                  <div style="font-size:0.78rem; font-weight:700; color:#6ee7b7; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-building-columns"></i> <span class="t-key" data-k="market_links_title">ప్రభుత్వ మార్కెట్ & ట్రేడింగ్ వెబ్‌సైట్లు:</span>
                  </div>
                  <div id="res-market-links-list" class="provider-links-grid"></div>
                </div>
                <!-- Healthy Feeds Purchase Links -->
                <div id="res-healthy-food-links-box" style="display:none; border-top:1px dashed rgba(255,255,255,0.15); padding-top:10px; margin-top:12px;">
                  <div style="font-size:0.78rem; font-weight:700; color:#fde68a; margin-bottom:6px; display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-wheat-awn"></i> <span class="t-key" data-k="food_links_title">మేత & సప్లిమెంట్లు లభించే అధికారిక వెబ్‌సైట్లు:</span>
                  </div>
                  <div id="res-healthy-food-links-list" class="provider-links-grid"></div>
                </div>
              </div>
            </div>
          </div>

          <!-- PROMINENT ERROR ALERT BANNER (Shown when NOT a fish or prawn) -->
          <div id="detect-error" style="display:none; padding:18px; border-radius:14px; background:rgba(239,68,68,0.18); border:2px solid #ef4444; color:#fecaca; margin-top:16px;">
            <div style="display:flex; align-items:flex-start; gap:12px;">
              <i class="fa-solid fa-triangle-exclamation" style="color:#ef4444; font-size:1.8rem; margin-top:2px;"></i>
              <div>
                <strong style="font-size:1.05rem; color:#f87171; display:block; margin-bottom:4px;" class="t-key" data-k="err_banner_title">
                  🚫 కేవలం చేప లేదా రొయ్య ఫోటో మాత్రమే ఇవ్వండి
                </strong>
                <p id="detect-error-msg" style="font-size:0.92rem; line-height:1.4; color:#fecaca; margin-bottom:8px;"></p>
                <p style="font-size:0.82rem; color:#fca5a5; opacity:0.9;" class="t-key" data-k="err_tip">
                  💡 సూచన: కెమెరాను నేరుగా చేప లేదా రొయ్య వైపు ఉంచండి. మనుషులు, ముఖాలు, గది వస్తువులు స్వయంచాలకంగా తిరస్కరించబడతాయి.
                </p>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- ======================================================================= -->
    <!-- TAB 2: DISSOLVED OXYGEN (DO) PREDICTOR                                  -->
    <!-- ======================================================================= -->
    <div id="tab-do" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <h2 class="card-title"><i class="fa-solid fa-gauge"></i> <span class="t-key" data-k="do_title">నీటి నాణ్యత & ఆక్సిజన్ పరీక్ష</span></h2>
          <p class="card-subtitle t-key" data-k="do_sub">చెరువు నీటి పారామితులను నమోదు చేసి ఆక్సిజన్ తెలుసుకోండి.</p>

          <form id="do-form">
            <div class="form-group">
              <label class="t-key" data-k="pond_species">చెరువు రకం</label>
              <select name="species" id="do-species">
                <option value="fish">🐟 చేపల చెరువు (Fish Pond)</option>
                <option value="prawn">🦐 రొయ్యల చెరువు (Prawn Pond)</option>
              </select>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label class="t-key" data-k="water_temp">నీటి ఉష్ణోగ్రత (°C)</label>
                <input type="number" step="0.1" name="water_temp_C" value="27.5" required>
              </div>
              <div class="form-group">
                <label class="t-key" data-k="water_ph">నీటి pH (6.5 - 9.0)</label>
                <input type="number" step="0.05" name="pH" value="7.8" required>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label class="t-key" data-k="salinity">సెలైనిటీ (ppt)</label>
                <input type="number" step="0.1" name="salinity_ppt" value="0.5" required>
              </div>
              <div class="form-group">
                <label class="t-key" data-k="ammonia">అమ్మోనియా (mg/L)</label>
                <input type="number" step="0.01" name="ammonia_mg_L" value="0.06" required>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label class="t-key" data-k="nitrite">నైట్రైట్ (mg/L)</label>
                <input type="number" step="0.01" name="nitrite_mg_L" value="0.03" required>
              </div>
              <div class="form-group">
                <label class="t-key" data-k="turbidity">టర్బిడిటీ (NTU)</label>
                <input type="number" step="0.5" name="turbidity_NTU" value="28.0" required>
              </div>
            </div>

            <div class="form-row">
              <div class="form-group">
                <label class="t-key" data-k="daily_feed">రోజువారీ మేత (kg)</label>
                <input type="number" step="0.5" name="feed_kg" value="8.0" required>
              </div>
              <div class="form-group">
                <label class="t-key" data-k="rainfall">వర్షపాతం (mm)</label>
                <input type="number" step="0.5" name="rainfall_mm" value="0.0" required>
              </div>
            </div>

            <div class="form-group">
              <label class="t-key" data-k="aerator_status">ఎరేటర్ స్థితి</label>
              <select name="aerator_on" id="do-aerator-select">
                <option value="1" selected>Aerators ON (నడుస్తున్నాయి)</option>
                <option value="0">Aerators OFF (ఆపివేయబడ్డాయి)</option>
              </select>
            </div>

            <button type="submit" class="btn btn-primary">
              <i class="fa-solid fa-calculator"></i> <span class="t-key" data-k="calc_do_btn">ఆక్సిజన్ & ఎరేటర్లను లెక్కించండి</span>
            </button>
          </form>
        </div>

        <div class="card" id="do-result-card">
          <h2 class="card-title"><i class="fa-solid fa-chart-line"></i> <span class="t-key" data-k="do_forecast_title">ఆక్సిజన్ నివేదిక & అంచనా</span></h2>
          <div id="do-placeholder" style="text-align:center; padding:50px 20px; color:var(--text-muted);">
            <i class="fa-solid fa-droplet" style="font-size:3rem; margin-bottom:12px; opacity:0.3;"></i>
            <p class="t-key" data-k="do_placeholder">చెరువు పారామితులను నమోదు చేసి గణించండి.</p>
          </div>
          <div id="do-loading" style="display:none; text-align:center; padding:40px 20px;">
            <i class="fa-solid fa-circle-notch fa-spin" style="font-size:2.5rem; color:var(--primary); margin-bottom:12px;"></i>
            <p style="font-weight:600;" class="t-key" data-k="do_calc_wait">ఆక్సిజన్ స్థాయి లెక్కింపబడుతోంది...</p>
          </div>
          <div id="do-result-panel" style="display:none;">
            <div style="background:rgba(255,255,255,0.03); border:1px solid var(--card-border); border-radius:14px; padding:18px; text-align:center; margin-bottom:16px;">
              <span style="font-size:0.8rem; text-transform:uppercase; color:var(--text-muted); font-weight:700;">PREDICTED DISSOLVED OXYGEN</span>
              <div id="res-do-value" style="font-size:2.8rem; font-weight:800; color:var(--primary); margin:6px 0;">-- mg/L</div>
              <span id="res-do-status" class="result-badge safe">Optimal</span>
            </div>
            <div class="action-card" style="border-left:4px solid var(--primary);">
              <h4 style="color:var(--primary);"><i class="fa-solid fa-clipboard-check"></i> <span class="t-key" data-k="do_assessment_title">నీటి స్థితి & సలహా</span></h4>
              <p id="res-do-advice" style="font-size:0.9rem; color:var(--text);"></p>
            </div>
            <div class="action-card" style="border-left:4px solid var(--accent); margin-top:10px;">
              <h4 style="color:var(--accent);"><i class="fa-solid fa-fan"></i> <span class="t-key" data-k="aerator_sched_title">ఎరేటర్ నిర్వహణ షెడ్యూల్</span></h4>
              <p id="res-do-aerator" style="font-size:0.9rem; color:var(--text);"></p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================================= -->
    <!-- TAB 3: FARMER VOICE ASSISTANT                                           -->
    <!-- ======================================================================= -->
    <div id="tab-assistant" class="tab-pane">
      <div class="card" style="max-width:850px; margin:0 auto;">
        <h2 class="card-title"><i class="fa-solid fa-headset"></i> <span class="t-key" data-k="assistant_title">రైతు వాయిస్ అసిస్టెంట్</span></h2>
        <p class="card-subtitle t-key" data-k="assistant_sub">చేపలు, రొయ్యల వ్యాధులు, మందులు మరియు ఆక్సిజన్ సమస్యలపై అడగండి.</p>
        
        <div id="chat-box" style="height:320px; overflow-y:auto; background:rgba(0,0,0,0.25); border:1px solid var(--card-border); border-radius:12px; padding:16px; margin-bottom:16px; display:flex; flex-direction:column; gap:10px;">
          <div style="background:rgba(0,240,255,0.1); border:1px solid var(--card-border); padding:10px 14px; border-radius:10px; max-width:85%;">
            👨‍🌾 <strong>AquaCare Assistant:</strong> <span id="chat-welcome-text">నమస్కారం! నేను ఆక్వాకేర్ రైతు సహాయకుడిని. చేపలు, రొయ్యల వ్యాధులు, మందులు మరియు ఆక్సిజన్ సమస్యలపై సహాయం చేయగలను. మైక్రోఫోన్ నొక్కి మాట్లాడండి లేదా టైప్ చేయండి!</span>
          </div>
        </div>

        <div style="display:flex; gap:10px; align-items:center; margin-bottom:10px; flex-wrap:wrap;">
          <label style="font-size:0.82rem; color:var(--text-muted); font-weight:700;">🎙 Voice Language</label>
          <select id="assistant-voice-lang" style="background:rgba(255,255,255,0.06); color:white; border:1px solid var(--card-border); border-radius:10px; padding:8px 12px;">
            <option value="te">తెలుగు</option>
            <option value="en">English</option>
            <option value="hi">हिन्दी</option>
          </select>
          <span style="font-size:0.78rem; color:var(--text-muted);">Mic is available only in Farmer Assistant.</span>
        </div>
        <form id="chat-form" style="display:flex; gap:10px;">
          <input type="text" id="chat-input" placeholder="మీ ప్రశ్నను ఇక్కడ అడగండి..." style="flex:1; background:rgba(255,255,255,0.06); border:1px solid var(--card-border); color:white; padding:10px 14px; border-radius:10px; outline:none;">
          <button type="button" id="mic-btn" class="btn btn-secondary" style="width:auto; padding:0 16px;" title="Speak only in Farmer Assistant">
            <i class="fa-solid fa-microphone"></i>
          </button>
          <button type="submit" class="btn btn-primary" style="width:auto; padding:0 22px;">
            <i class="fa-solid fa-paper-plane"></i>
          </button>
        </form>
      </div>
    </div>

    <!-- ======================================================================= -->
    <!-- TAB 4: SELLING & RATES                                                 -->
    <!-- ======================================================================= -->
    <div id="tab-selling" class="tab-pane">
      <div class="grid-2">
        <div class="card">
          <h2 class="card-title"><i class="fa-solid fa-calculator"></i> <span class="t-key" data-k="calc_title">హార్వెస్ట్ రాబడి కాలిక్యులేటర్</span></h2>
          <div class="form-group">
            <label class="t-key" data-k="calc_species_label">జాతి & సైజు ఎంపిక</label>
            <select id="calc-species-select" onchange="updateCalcRate()">
              <option value="165">🐟 Rohu / Catla (1.5-2.5 kg) - ₹165/kg</option>
              <option value="185">🐟 Catla Large (3.0+ kg) - ₹185/kg</option>
              <option value="415">🦐 Vannamei 30 Count (Large) - ₹415/kg</option>
              <option value="355" selected>🦐 Vannamei 40 Count (Medium) - ₹355/kg</option>
              <option value="305">🦐 Vannamei 50 Count (Standard) - ₹305/kg</option>
              <option value="520">🐟 Murrel / Korameenu - ₹520/kg</option>
            </select>
          </div>
          <div class="form-row">
            <div class="form-group">
              <label class="t-key" data-k="calc_biomass_label">మొత్తం బరువు / బయోమాస్ (kg)</label>
              <input type="number" id="calc-weight" value="2500" oninput="calculateTotalIncome()">
            </div>
            <div class="form-group">
              <label class="t-key" data-k="calc_rate_label">ధర రేటు (₹ / kg)</label>
              <input type="number" id="calc-price" value="355" oninput="calculateTotalIncome()">
            </div>
          </div>
          <div style="background:rgba(255,255,255,0.03); border:1px solid var(--card-border); border-radius:12px; padding:16px; margin-top:14px; text-align:center;">
            <span style="font-size:0.8rem; color:var(--text-muted); font-weight:700;" class="t-key" data-k="gross_revenue_label">అంచనా మొత్తం రాబడి</span>
            <div id="calc-gross-revenue" style="font-size:2.2rem; font-weight:800; color:var(--emerald); margin:6px 0;">₹ 8,87,500</div>
            <div style="font-size:0.85rem; color:var(--text-muted);">
              <span class="t-key" data-k="net_profit_label">అంచనా నికర లాభం (~60% ఖర్చులు తీసివేయగా):</span> <strong id="calc-net-profit" style="color:var(--primary);">₹ 3,55,000</strong>
            </div>
          </div>
        </div>

        <div class="card">
          <h2 class="card-title"><i class="fa-solid fa-chart-simple"></i> <span class="t-key" data-k="commercial_rates_title">ప్రస్తుత మార్కెట్ ధరలు</span></h2>
          <div style="margin-bottom:12px;">
            <h4 style="color:var(--primary); font-size:0.95rem; margin-bottom:6px;"><span class="t-key" data-k="fish_rates_title">🐟 చేపల వ్యాపార ధరలు:</span></h4>
            {% for item, data in MARKET_SELLING_RATES["fish"].items() %}
            <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.06); font-size:0.88rem;">
              <span>{{ item }}</span>
              <strong style="color:var(--emerald);">{{ data.price_range }}</strong>
            </div>
            {% endfor %}
          </div>
          <div style="margin-top:16px;">
            <h4 style="color:var(--primary); font-size:0.95rem; margin-bottom:6px;"><span class="t-key" data-k="prawn_rates_title">🦐 వనామి రొయ్యల కౌంట్ ధరలు:</span></h4>
            {% for item, data in MARKET_SELLING_RATES["prawn"].items() %}
            <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.06); font-size:0.88rem;">
              <span>{{ item }}</span>
              <strong style="color:var(--emerald);">{{ data.price_range }}</strong>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>
    </div>

    <!-- ======================================================================= -->
    <!-- TAB 5: BUYING HUB                                                       -->
    <!-- ======================================================================= -->
    <div id="tab-buying" class="tab-pane">
      <div class="card" style="margin-bottom:20px;">
        <h2 class="card-title"><i class="fa-solid fa-store"></i> <span class="t-key" data-k="buy_title">AquaCare స్టోర్ • ధృవీకరించబడిన మందులు & సీడ్</span></h2>
        <p class="card-subtitle t-key" data-k="buy_sub">ఆంధ్రప్రదేశ్ మరియు సమీప ఆక్వా ప్రాంతాలకు నేరుగా సరఫరా. వాట్సాప్ లేదా విచారణ ద్వారా ఆర్డర్ చేయండి.</p>
      </div>

      <div class="catalog-grid" id="catalog-container">
        {% for item in buy_items %}
        <div class="product-card" data-prod-id="{{ item.id }}">
          <div>
            <span style="font-size:0.75rem; padding:2px 8px; border-radius:10px; background:rgba(0,240,255,0.12); color:var(--primary); font-weight:700; text-transform:uppercase;">{{ item.category }}</span>
            <h3 class="prod-title" style="font-size:1.05rem; margin:8px 0 4px 0; color:var(--text);">{{ item.title_te }}</h3>
            <p class="prod-purpose" style="font-size:0.8rem; color:var(--text-muted); margin-bottom:10px;">{{ item.purpose_te }}</p>
          </div>
          <div>
            <div style="font-size:1.1rem; font-weight:800; color:var(--primary); margin-bottom:10px;">{{ item.price }}</div>
            <div style="display:flex; gap:8px;">
              <button class="btn btn-primary" onclick="openBuyModal('{{ item.title_te }}', '{{ item.price }}')" style="font-size:0.85rem; padding:8px 12px;">
                <i class="fa-solid fa-cart-plus"></i> <span class="t-key" data-k="order_btn">ఆర్డర్ చేయండి</span>
              </button>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

  </div>

  <!-- ORDER MODAL -->
  <div class="modal-backdrop" id="buy-modal">
    <div class="modal-card">
      <button onclick="closeBuyModal()" style="position:absolute; top:16px; right:16px; background:transparent; border:none; color:white; font-size:1.2rem; cursor:pointer;">✕</button>
      <h3 style="color:var(--primary); margin-bottom:6px;"><i class="fa-solid fa-cart-shopping"></i> <span class="t-key" data-k="order_modal_title">ఆర్డర్ విచారణ ఫారం</span></h3>
      <p id="modal-product-name" style="font-weight:700; margin-bottom:2px; color:white;"></p>
      <p id="modal-product-price" style="font-size:0.9rem; color:var(--emerald); margin-bottom:14px;"></p>
      <form onsubmit="submitOrder(event)">
        <input type="hidden" id="order-item-input">
        <div class="form-group">
          <label class="t-key" data-k="farmer_name_label">రైతు పేరు (Farmer Name)</label>
          <input type="text" id="order-farmer-name" required>
        </div>
        <div class="form-group">
          <label class="t-key" data-k="phone_label">ఫోన్ నెంబర్ (Phone Number)</label>
          <input type="tel" id="order-phone" required placeholder="9876543210">
        </div>
        <div class="form-group">
          <label class="t-key" data-k="location_label">గ్రామం / జిల్లా (Village / District)</label>
          <input type="text" id="order-location" required placeholder="Bhimavaram, W.G. Dist">
        </div>
        <div class="form-group">
          <label class="t-key" data-k="qty_label">కావలసిన పరిమాణం (Quantity Needed)</label>
          <input type="text" id="order-quantity" required placeholder="5 kg / 2 bags / 50,000 seed">
        </div>
        <div style="display:flex; gap:10px; margin-top:14px;">
          <button type="submit" class="btn btn-primary" style="flex:1;" class="t-key" data-k="submit_inquiry_btn">సమర్పించండి</button>
          <button type="button" class="btn btn-secondary" onclick="orderViaWhatsApp()" style="background:#25d366; color:white; border:none;">
            <i class="fa-brands fa-whatsapp"></i> WhatsApp
          </button>
        </div>
      </form>
    </div>
  </div>

  <!-- JAVASCRIPT LOGIC -->
  <script>
    let currentLang = 'te';
    const synth = window.speechSynthesis;
    let lastDetectionData = null;
    let lastDOData = null;

    // CATALOG DATA IN JS FOR DYNAMIC TRANSLATION
    const CATALOG_DATA = {{ buy_items | tojson }};

    // 3-LANGUAGE TRANSLATION DICTIONARY: Telugu / English / Hindi
    const TRANSLATIONS = {
      te: {
        header_sub: "చేపలు & రొయ్యల AI ప్లాట్‌ఫామ్ • లైవ్ కెమెరా • మందులు & మేత • బహుభాషా మద్దతు",
        guard_badge: "చేప & రొయ్య గార్డ్ రక్షణలో ఉంది",
        tab_detection: "వ్యాధి గుర్తింపు & కెమెరా",
        tab_do: "ఆక్సిజన్ & నీటి నాణ్యత",
        tab_assistant: "రైతు వాయిస్ అసిస్టెంట్",
        tab_selling: "అమ్మకాలు & మార్కెట్ ధరలు",
        tab_buying: "మందులు కొనడానికి (Buy Hub)",
        detect_title: "ఫోటో అప్‌లోడ్ చేయండి లేదా లైవ్ కెమెరా వాడండి",
        detect_sub: "కేవలం చేప మరియు రొయ్యల ఫోటోలు మాత్రమే అనుమతించబడతాయి. వ్యాధి ఉన్న ప్రదేశంపై బౌండింగ్ బాక్స్‌లు వస్తాయి.",
        open_front_cam: "🤳 ఫ్రంట్ కెమెరా (Webcam)",
        open_back_cam: "📷 బ్యాక్ / వెబ్‌క్యామ్ కెమెరా (చేప / రొయ్య)",
        flip_camera: "కెమెరా మార్చండి",
        capture_photo: "ఫోటో తీసి స్కాన్ చేయండి",
        close_camera: "కెమెరా ఆపివేయండి",
        species_label: "జాతి ఎంపిక",
        quick_samples_title: "⚡ శాంపిల్ ఫోటోలు (వెంటనే టెస్ట్ చేయడానికి):",
        sample_healthy_fish: "🐟 ఆరోగ్యకరమైన చేప",
        sample_bacterial: "🐟 బాక్టీరియల్ పుండ్లు",
        sample_wssv: "🦐 తెల్ల మచ్చ వైరస్",
        sample_blackgill: "🦐 నల్ల మొప్ప రొయ్య",
        drag_title: "చేప లేదా రొయ్య ఫోటో ఇక్కడ వేయండి",
        drag_sub: "లేదా ఫైల్ ఎంచుకోవడానికి ఇక్కడ క్లిక్ చేయండి (JPG, PNG, WEBP)",
        detect_btn: "వ్యాధి & ప్రదేశాన్ని గుర్తించండి",
        report_title: "రోగ నిర్ధారణ & చికిత్స నివేదిక",
        report_sub: "శరీరంపై వ్యాధి ఉన్న ఖచ్చితమైన ప్రదేశాలు, సరైన మందులు మరియు ఆహార సూచనలు.",
        placeholder_text: "ఫోటో అప్‌లోడ్ చేయండి లేదా కెమెరాతో తీయండి.",
        scanning_text: "YOLO బౌండింగ్ బాక్స్‌లతో స్కాన్ చేయబడుతోంది...",
        listen_btn: "వాయిస్ వినండి (Listen)",
        spots_title: "శరీరంపై వ్యాధి ఉన్న ఖచ్చితమైన ప్రదేశాలు:",
        identified_class: "గుర్తించిన వ్యాధి / స్థితి",
        confidence_score: "ఖచ్చితత్వం (కాన్ఫిడెన్స్)",
        med_title: "మందులు & రసాయన చికిత్స (MANDHULU)",
        med_links_title: "మందులు లభించే అధికారిక వెబ్‌సైట్లు & స్టోర్ లింకులు:",
        food_title: "ఆహారం & మేత యాజమాన్యం (FOOD EM CHEYALO)",
        food_links_title: "మేత & సప్లిమెంట్లు లభించే అధికారిక వెబ్‌సైట్లు:",
        sell_title: "మార్కెట్ అమ్మకాలు & రాబడి వివరాలు",
        market_links_title: "ప్రభుత్వ మార్కెట్ & ట్రేడింగ్ వెబ్‌సైట్లు:",
        harvest_tips_title: "హార్వెస్టింగ్ & మార్కెట్ సలహాలు:",
        buy_medicines_btn: "మందులు కొనండి",
        err_banner_title: "🚫 కేవలం చేప లేదా రొయ్య ఫోటోలు మాత్రమే అనుమతించబడతాయి",
        err_tip: "💡 సూచన: కెమెరాను నేరుగా చేప లేదా రొయ్య వైపు ఉంచండి. మనుషులు, ముఖాలు, గది వస్తువులు స్వయంచాలకంగా తిరస్కరించబడతాయి.",
        do_title: "నీటి నాణ్యత & ఆక్సిజన్ పరీక్ష",
        do_sub: "చెరువు నీటి పారామితులను నమోదు చేసి ఆక్సిజన్ తెలుసుకోండి.",
        pond_species: "చెరువు రకం",
        water_temp: "నీటి ఉష్ణోగ్రత (°C)",
        water_ph: "నీటి pH (6.5 - 9.0)",
        salinity: "సెలైనిటీ (ppt)",
        ammonia: "అమ్మోనియా (mg/L)",
        nitrite: "నైట్రైట్ (mg/L)",
        turbidity: "టర్బిడిటీ (NTU)",
        daily_feed: "రోజువారీ మేత (kg)",
        rainfall: "వర్షపాతం (mm)",
        aerator_status: "ఎరేటర్ స్థితి",
        calc_do_btn: "ఆక్సిజన్ & ఎరేటర్లను లెక్కించండి",
        do_forecast_title: "ఆక్సిజన్ నివేదిక & అంచనా",
        do_placeholder: "చెరువు పారామితులను నమోదు చేసి గణించండి.",
        do_calc_wait: "ఆక్సిజన్ స్థాయి లెక్కింపబడుతోంది...",
        do_assessment_title: "నీటి స్థితి & సలహా",
        aerator_sched_title: "ఎరేటర్ నిర్వహణ షెడ్యూల్",
        assistant_title: "రైతు వాయిస్ అసిస్టెంట్",
        assistant_sub: "చేపలు, రొయ్యల వ్యాధులు, మందులు మరియు ఆక్సిజన్ సమస్యలపై అడగండి.",
        chat_welcome: "నమస్కారం! నేను ఆక్వాకేర్ రైతు సహాయకుడిని. చేపలు, రొయ్యల వ్యాధులు, మందులు మరియు ఆక్సిజన్ సమస్యలపై సహాయం చేయగలను. మైక్రోఫోన్ నొక్కి మాట్లాడండి లేదా టైప్ చేయండి!",
        chat_placeholder: "మీ ప్రశ్నను ఇక్కడ అడగండి (ఉదా: ఆక్సిజన్ పడిపోయింది, తెల్ల మచ్చ)...",
        calc_title: "హార్వెస్ట్ రాబడి కాలిక్యులేటర్",
        calc_species_label: "జాతి & సైజు ఎంపిక",
        calc_biomass_label: "మొత్తం బరువు / బయోమాస్ (kg)",
        calc_rate_label: "ధర రేటు (₹ / kg)",
        gross_revenue_label: "అంచనా మొత్తం రాబడి",
        net_profit_label: "అంచనా నికర లాభం (~60% ఖర్చులు తీసివేయగా):",
        commercial_rates_title: "ప్రస్తుత మార్కెట్ ధరలు",
        fish_rates_title: "🐟 చేపల వ్యాపార ధరలు:",
        prawn_rates_title: "🦐 వనామి రొయ్యల కౌంట్ ధరలు:",
        buy_title: "AquaCare స్టోర్ • ధృవీకరించబడిన మందులు & సీడ్",
        buy_sub: "ఆంధ్రప్రదేశ్ మరియు సమీప ఆక్వా ప్రాంతాలకు నేరుగా సరఫరా. వాట్సాప్ లేదా విచారణ ద్వారా ఆర్డర్ చేయండి.",
        order_btn: "ఆర్డర్ చేయండి",
        order_modal_title: "ఆర్డర్ విచారణ ఫారం",
        farmer_name_label: "రైతు పేరు (Farmer Name)",
        phone_label: "ఫోన్ నెంబర్ (Phone Number)",
        location_label: "గ్రామం / జిల్లా (Village / District)",
        qty_label: "కావలసిన పరిమాణం (Quantity Needed)",
        submit_inquiry_btn: "సమర్పించండి"
      },
      en: {
        header_sub: "Fish & Prawn AI Platform • Live Camera • Disease & Feeding • Multilingual",
        guard_badge: "Fish & Prawn Guard Active",
        tab_detection: "Disease & Live Camera",
        tab_do: "DO & Water Quality",
        tab_assistant: "Farmer Voice Assistant",
        tab_selling: "Selling & Market Rates",
        tab_buying: "Buy Hub (Store)",
        detect_title: "Upload Image or Use Live Camera",
        detect_sub: "Select Fish or Prawn. The selected species is locked; opposite-species images are rejected. Disease boxes mark the model-detected affected region.",
        open_front_cam: "🤳 Front Camera (Webcam)",
        open_back_cam: "📷 Back / Webcam Camera (Fish / Prawn)",
        flip_camera: "Switch Camera",
        capture_photo: "Snapshot",
        close_camera: "Close Camera",
        species_label: "Species Selection",
        quick_samples_title: "⚡ Quick Test Samples:",
        sample_healthy_fish: "🐟 Healthy Fish",
        sample_bacterial: "🐟 Bacterial Ulcer",
        sample_wssv: "🦐 WSSV White Spot",
        sample_blackgill: "🦐 Black Gill Prawn",
        drag_title: "Drag & Drop Fish or Prawn Image",
        drag_sub: "or click to browse from device (JPG, PNG, WEBP)",
        detect_btn: "Detect Disease & Exact Locations",
        report_title: "Diagnostic & Treatment Report",
        report_sub: "Pinpoint localization of infected areas with custom medicine, food, and market guides.",
        placeholder_text: "Upload an image or take a live camera photo to begin.",
        scanning_text: "Scanning Fish/Prawn + disease region with AI bounding boxes...",
        listen_btn: "Listen Voice Advice",
        spots_title: "Disease Spot Locations (corrected disease-region boxes):",
        identified_class: "Condition Identified",
        confidence_score: "Confidence Score",
        med_title: "Medicine & Chemical Treatment (MANDHULU)",
        med_links_title: "Medicine & Treatment Purchase Links (Official Stores):",
        food_title: "Food & Feeding Guidelines (FOOD EM CHEYALO)",
        food_links_title: "Feed & Nutrition Purchase Links (Official Websites):",
        sell_title: "Selling & Commercial Market Rates",
        market_links_title: "Government Aqua Trading & Market Portals:",
        harvest_tips_title: "Harvesting & Selling Strategy:",
        buy_medicines_btn: "Buy Medicines",
        err_banner_title: "⚠️ Reliable Fish/Prawn Detection Required",
        err_tip: "💡 Tip: Use a clear fish/prawn photo. Full-animal and disease close-up images are supported; weak disease evidence is not shown as a fake diagnosis.",
        do_title: "Water Quality & DO Simulation",
        do_sub: "Input pond water sensor parameters to predict Dissolved Oxygen and aerator requirements.",
        pond_species: "Pond Species",
        water_temp: "Water Temp (°C)",
        water_ph: "Water pH (6.5 - 9.0)",
        salinity: "Salinity (ppt)",
        ammonia: "Ammonia (mg/L)",
        nitrite: "Nitrite (mg/L)",
        turbidity: "Turbidity (NTU)",
        daily_feed: "Daily Feed (kg)",
        rainfall: "Rainfall (mm)",
        aerator_status: "Aerator Status",
        calc_do_btn: "Calculate DO & Forecast Aeration",
        do_forecast_title: "DO Forecast & Health Impact",
        do_placeholder: "Enter pond parameters on the left to simulate DO.",
        do_calc_wait: "Calculating dissolved oxygen level...",
        do_assessment_title: "Water Assessment & Advice",
        aerator_sched_title: "Aerator Schedule",
        assistant_title: "Farmer Voice Assistant",
        assistant_sub: "Ask questions about fish/prawn diseases, DO management, or prices.",
        chat_welcome: "Hello! I am AquaCare Farmer Assistant. I can assist you with fish & prawn diseases, medicine treatments, and oxygen levels. Speak or type your question!",
        chat_placeholder: "Ask your question here (e.g., low oxygen, white spot, bacterial ulcer)...",
        calc_title: "Harvest Revenue Calculator",
        calc_species_label: "Select Species & Count",
        calc_biomass_label: "Total Harvest Biomass (kg)",
        calc_rate_label: "Price Rate (₹ / kg)",
        gross_revenue_label: "ESTIMATED GROSS REVENUE",
        net_profit_label: "Estimated Net Profit (After ~60% feed/power costs):",
        commercial_rates_title: "Live Market Selling Rates",
        fish_rates_title: "🐟 Fish Commercial Rates:",
        prawn_rates_title: "🦐 Vannamei Prawn Rates:",
        buy_title: "AquaCare Store • Buy Certified Medicines & Feed",
        buy_sub: "Direct delivery across aquaculture belts. Order via WhatsApp or submit inquiry.",
        order_btn: "Order Now",
        order_modal_title: "Product Order Inquiry",
        farmer_name_label: "Farmer Name",
        phone_label: "Phone Number",
        location_label: "Village / District",
        qty_label: "Quantity Needed",
        submit_inquiry_btn: "Submit Inquiry"
      },
      hi: {
        header_sub: "मछली और झींगा एआई प्लेटफॉर्म • लाइव कैमरा • रोग व चारा मार्गदर्शन • बहुभाषी",
        guard_badge: "मछली व झींगा सुरक्षा सक्रिय",
        tab_detection: "रोग पहचान और कैमरा",
        tab_do: "ऑक्सीजन और जल गुणवत्ता",
        tab_assistant: "किसान वॉइस सहायक",
        tab_selling: "बिक्री और बाजार दरें",
        tab_buying: "दवाइयां खरीदें (Buy Hub)",
        detect_title: "फोटो अपलोड करें या लाइव कैमरा उपयोग करें",
        detect_sub: "केवल मछली और झींगा की तस्वीरें मान्य हैं। रोग प्रभावित क्षेत्र पर सटीक बाउंडिंग बॉक्स आते हैं।",
        open_front_cam: "🤳 फ्रंट कैमरा (सेल्फी / आप)",
        open_back_cam: "📷 बैक कैमरा (पीछे / मछली)",
        flip_camera: "कैमरा बदलें",
        capture_photo: "फोटो खींचकर जांचें",
        close_camera: "कैमरा बंद करें",
        species_label: "प्रजाति चयन",
        quick_samples_title: "⚡ त्वरित परीक्षण नमूने:",
        sample_healthy_fish: "🐟 स्वस्थ मछली",
        sample_bacterial: "🐟 बैक्टीरियल घाव",
        sample_wssv: "🦐 सफेद धब्बा वायरस",
        sample_blackgill: "🦐 काली गलफड़ झींगा",
        drag_title: "मछली या झींगा की फोटो यहाँ डालें",
        drag_sub: "या फ़ाइल चुनने के लिए यहाँ क्लिक करें (JPG, PNG, WEBP)",
        detect_btn: "रोग और सटीक स्थान पहचानें",
        report_title: "रोग निदान और उपचार रिपोर्ट",
        report_sub: "संक्रमित क्षेत्रों का सटीक स्थान, उचित दवाइयां और आहार संबंधी दिशानिर्देश।",
        placeholder_text: "शुरू करने के लिए फोटो अपलोड करें या लाइव कैमरे से लें।",
        scanning_text: "YOLO बाउंडिंग बॉक्स से स्कैन किया जा रहा है...",
        listen_btn: "सलाह सुनें (Listen)",
        spots_title: "शरीर पर बीमारी के सटीक स्थान:",
        identified_class: "पहचाना गया रोग / स्थिति",
        confidence_score: "सटीकता दर (Confidence)",
        med_title: "दवाइयां और रासायनिक उपचार (MANDHULU)",
        med_links_title: "दवाइयां खरीदने के आधिकारिक वेबसाइट और स्टोर लिंक:",
        food_title: "आहार और चारा प्रबंधन (FOOD EM CHEYALO)",
        food_links_title: "चारा और पोषण उत्पाद खरीदने के आधिकारिक वेबसाइट लिंक:",
        sell_title: "बिक्री और व्यावसायिक बाजार दरें",
        market_links_title: "सरकारी मत्स्य व्यापार और बाजार पोर्टल:",
        harvest_tips_title: "कटाई और बिक्री रणनीति:",
        buy_medicines_btn: "दवाइयां खरीदें",
        err_banner_title: "🚫 केवल मछली और झींगा की तस्वीरें मान्य हैं",
        err_tip: "💡 सुझाव: कैमरे को सीधे मछली या झींगा की ओर रखें। इंसानों या कमरों को अस्वीकार कर दिया जाता है।",
        do_title: "जल गुणवत्ता और ऑक्सीजन सिमुलेशन",
        do_sub: "घुलित ऑक्सीजन और एरेटर आवश्यकता जानने के लिए सेंसर मान दर्ज करें।",
        pond_species: "तालाब का प्रकार",
        water_temp: "पानी का तापमान (°C)",
        water_ph: "पानी का pH (6.5 - 9.0)",
        salinity: "लवणता / सेलिनीटी (ppt)",
        ammonia: "अमोनिया (mg/L)",
        nitrite: "नाइट्राइट (mg/L)",
        turbidity: "गंदलापन (NTU)",
        daily_feed: "दैनिक चारा (kg)",
        rainfall: "वर्षा (mm)",
        aerator_status: "एरेटर की स्थिति",
        calc_do_btn: "ऑक्सीजन और एरेटर की गणना करें",
        do_forecast_title: "ऑक्सीजन पूर्वानुमान और स्वास्थ्य प्रभाव",
        do_placeholder: "ऑक्सीजन जानने के लिए बाईं ओर मान भरें।",
        do_calc_wait: "ऑक्सीजन की गणना हो रही है...",
        do_assessment_title: "पानी का आकलन और सलाह",
        aerator_sched_title: "एरेटर अनुसूची",
        assistant_title: "किसान वॉइस सहायक",
        assistant_sub: "मछली/झींगा रोगों, दवाओं और ऑक्सीजन स्तर पर प्रश्न पूछें।",
        chat_welcome: "नमस्ते! मैं एक्वाकेयर किसान सहायक हूँ। मछली और झींगा रोगों, दवाइयों और ऑक्सीजन समस्याओं पर आपकी सहायता कर सकता हूँ। बोलें या लिखें!",
        chat_placeholder: "अपना प्रश्न यहाँ पूछें (उदा: ऑक्सीजन कम हो गई, सफेद धब्बा)...",
        calc_title: "कटाई आय कैलकुलेटर",
        calc_species_label: "प्रजाति और काउंट चुनें",
        calc_biomass_label: "कुल वजन / बायोमास (kg)",
        calc_rate_label: "मूल्य दर (₹ / kg)",
        gross_revenue_label: "अनुमानित कुल आय",
        net_profit_label: "अनुमानित शुद्ध लाभ (~60% लागत घटाकर):",
        commercial_rates_title: "वर्तमान बाजार दरें",
        fish_rates_title: "🐟 मछली बाजार दरें:",
        prawn_rates_title: "🦐 वेनामेई झींगा काउंट दरें:",
        buy_title: "एक्वाकेयर स्टोर • प्रमाणित दवाइयां और चारा",
        buy_sub: "तालाब तक सीधी डिलीवरी। व्हाट्सएप या पूछताछ फॉर्म द्वारा ऑर्डर करें।",
        order_btn: "ऑर्डर करें",
        order_modal_title: "उत्पाद ऑर्डर पूछताछ",
        farmer_name_label: "किसान का नाम",
        phone_label: "फोन नंबर",
        location_label: "गांव / जिला",
        qty_label: "आवश्यक मात्रा",
        submit_inquiry_btn: "जमा करें"
      },
      tel: {
        header_sub: "Chepalu & Royyala AI Platform • Live Camera • Mandulu & Meetha Guidance",
        guard_badge: "Fish & Prawn Guard Active",
        tab_detection: "Disease & Camera",
        tab_do: "DO & Water Quality",
        tab_assistant: "Raitu Voice Assistant",
        tab_selling: "Selling & Rates (Ammakalu)",
        tab_buying: "Buy Hub (Konadaniki)",
        detect_title: "Photo Upload Cheyandi leda Live Camera Vadandi",
        detect_sub: "Kevalam Chepa mariyu Royya photos matrame allow cheyabadathayi. Exact spots meeda bounding boxes vasthayi.",
        open_front_cam: "🤳 Front Camera (Nenu kanipisthanu)",
        open_back_cam: "📷 Back Camera (Venuka unnavallu / Chepa)",
        flip_camera: "Camera Marchandi",
        capture_photo: "Photo Theesi Scan Cheyandi",
        close_camera: "Camera Aapeyandi",
        species_label: "Species Select Cheyandi",
        quick_samples_title: "⚡ Quick Test Samples:",
        sample_healthy_fish: "🐟 Healthy Chepa",
        sample_bacterial: "🐟 Bacterial Pundlu",
        sample_wssv: "🦐 Thella Macha (WSSV)",
        sample_blackgill: "🦐 Nalla Moppa Royya",
        drag_title: "Chepa leda Royya Photo Ikkada Drop Cheyandi",
        drag_sub: "leda browse cheyadaniki ikkada click cheyandi (JPG, PNG, WEBP)",
        detect_btn: "Disease & Spots Gurthinchandi",
        report_title: "Diagnostic & Treatment Report",
        report_sub: "Infection spots, mandulu mariyu food guidelines.",
        placeholder_text: "Photo upload chesi leda Camera tho photo theesi testing start cheyandi.",
        scanning_text: "YOLO boxes tho scan avthundi...",
        listen_btn: "Voice Vinandi (Listen)",
        spots_title: "Shareeram meeda rogam unna spots:",
        identified_class: "Identified Condition",
        confidence_score: "Confidence Score",
        med_title: "Mandulu & Chemical Treatment",
        med_links_title: "Mandulu labhinche official website & store links:",
        food_title: "Food & Meetha Management",
        food_links_title: "Meetha & nutrition labhinche official website links:",
        sell_title: "Market Selling & Rates Vivaralu",
        market_links_title: "Govt Aqua Trading & Market Portals:",
        harvest_tips_title: "Harvest & Selling Strategy:",
        buy_medicines_btn: "Mandulu Konandi",
        err_banner_title: "🚫 Kevalam Chepa leda Royya Photos Matrame Allow",
        err_tip: "💡 Tip: Camera ni direct ga chepa/royya vaipu unchandi. Manushulu, rooms allow cheyaru.",
        do_title: "Water Quality & Oxygen Test",
        do_sub: "Cheruvu parameters enter chesi DO telusukondi.",
        pond_species: "Cheruvu Rakam",
        water_temp: "Water Temp (°C)",
        water_ph: "Water pH",
        salinity: "Salinity (ppt)",
        ammonia: "Ammonia (mg/L)",
        nitrite: "Nitrite (mg/L)",
        turbidity: "Turbidity (NTU)",
        daily_feed: "Daily Feed (kg)",
        rainfall: "Rainfall (mm)",
        aerator_status: "Aerator Status",
        calc_do_btn: "DO & Aeration Calculate Cheyandi",
        do_forecast_title: "DO Forecast & Health Impact",
        do_placeholder: "Cheruvu parameters enter chesi test cheyandi.",
        do_calc_wait: "Oxygen level calculate avthundi...",
        do_assessment_title: "Water Assessment & Advice",
        aerator_sched_title: "Aerator Schedule",
        assistant_title: "Raitu Voice Assistant",
        assistant_sub: "Chepalu, royyala vyadhulu, mandulu mariyu DO meeda prashnalu adagandi.",
        chat_welcome: "Namaskaram! Nenu AquaCare raitu sahayakudini. Chepalu, royyala vyadhulu, mandulu mariyu oxygen meeda help chesthanu. Matladandi leda type cheyandi!",
        chat_placeholder: "Mee prashna ikkada adagandi (e.g., oxygen padipoindi, thella macha)...",
        calc_title: "Harvest Revenue Calculator",
        calc_species_label: "Species & Count Select Cheyandi",
        calc_biomass_label: "Total Harvest Weight (kg)",
        calc_rate_label: "Rate (₹ / kg)",
        gross_revenue_label: "ESTIMATED TOTAL INCOME",
        net_profit_label: "Estimated Net Profit (~60% karchulu poga):",
        commercial_rates_title: "Current Market Rates",
        fish_rates_title: "🐟 Chepala Market Rates:",
        prawn_rates_title: "🦐 Vannamei Royyala Rates:",
        buy_title: "AquaCare Store • Certified Mandulu & Seed",
        buy_sub: "Direct ga farm delivery. WhatsApp leda inquiry form dwara order cheyandi.",
        order_btn: "Order Cheyandi",
        order_modal_title: "Product Order Inquiry",
        farmer_name_label: "Raitu Peru",
        phone_label: "Phone Number",
        location_label: "Gramam / District",
        qty_label: "Kavalasina Quantity",
        submit_inquiry_btn: "Submit Cheyandi"
      }
    };

    function switchLanguage(lang) {
      // AquaCare UI supports only English, Telugu and Hindi.
      if (!["en", "te", "hi"].includes(lang)) lang = "te";
      currentLang = lang;
      const dict = TRANSLATIONS[lang] || TRANSLATIONS.te;

      // Update all elements with data-k
      document.querySelectorAll('.t-key').forEach(el => {
        const k = el.getAttribute('data-k');
        if (dict[k]) el.innerText = dict[k];
      });

      // Update placeholders
      const chatIn = document.getElementById('chat-input');
      if (chatIn && dict.chat_placeholder) chatIn.placeholder = dict.chat_placeholder;

      // Update welcome text in chat
      const welcome = document.getElementById('chat-welcome-text');
      if (welcome && dict.chat_welcome) welcome.innerText = dict.chat_welcome;

      // Update dropdown options
      const spSelect = document.getElementById('species-select');
      if (spSelect) {
        if (lang === 'te') {
          spSelect.options[0].text = "🐟 చేప (Fish)";
          spSelect.options[1].text = "🦐 రొయ్య (Prawn / Shrimp)";
        } else if (lang === 'hi') {
          spSelect.options[0].text = "🐟 मछली (Fish)";
          spSelect.options[1].text = "🦐 झींगा (Prawn / Shrimp)";
        } else {
          spSelect.options[0].text = "🐟 Fish";
          spSelect.options[1].text = "🦐 Prawn / Shrimp";
        }
      }

      // Update Catalog Product Cards text
      document.querySelectorAll('.product-card').forEach(card => {
        const id = card.getAttribute('data-prod-id');
        const prod = CATALOG_DATA.find(p => p.id === id);
        if (prod) {
          const titleEl = card.querySelector('.prod-title');
          const purpEl = card.querySelector('.prod-purpose');
          if (titleEl) {
            titleEl.innerText = lang === 'te' ? prod.title_te : (lang === 'hi' ? (prod.title_hi || prod.title_en) : (lang === 'tel' ? prod.title_tel : prod.title_en));
          }
          if (purpEl) {
            purpEl.innerText = lang === 'te' ? prod.purpose_te : (lang === 'hi' ? (prod.purpose_hi || prod.purpose_en) : (lang === 'tel' ? prod.purpose_tel : prod.purpose_en));
          }
        }
      });

      // Re-render detection results if already present
      if (lastDetectionData && lastDetectionData.success) {
        renderDetectionResult(lastDetectionData);
      }

      // Re-render DO results if already present
      if (lastDOData && lastDOData.success) {
        renderDOResult(lastDOData);
      }

      document.getElementById('lang-select').value = lang;
      try { localStorage.setItem('aquacare_lang', lang); } catch (e) {}
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      const target = document.getElementById(tabId);
      if (target) target.classList.add('active');
      const btn = document.querySelector(`[onclick="switchTab('${tabId}')"]`);
      if (btn) btn.classList.add('active');
    }

    function speakText(text, langOverride = null) {
      if (!synth || !text) return;
      const lang = ["en", "te", "hi"].includes(langOverride) ? langOverride : currentLang;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(String(text));
      utterance.lang = lang === 'hi' ? 'hi-IN' : (lang === 'te' ? 'te-IN' : 'en-IN');
      utterance.rate = 0.95;
      synth.speak(utterance);
    }

    // =========================================================================
    // LIVE CAMERA SYSTEM: STABLE WEBCAM + LIVE SPECIES-LOCKED SCAN
    let currentStream = null;
    let currentFacingMode = 'environment';
    let activeDeviceId = null;
    let liveScanTimer = null;
    let liveScanBusy = false;

    function stopLiveAutoScan() {
      if (liveScanTimer) {
        clearInterval(liveScanTimer);
        liveScanTimer = null;
      }
      liveScanBusy = false;
    }

    function startLiveAutoScan() {
      stopLiveAutoScan();
      liveScanTimer = setInterval(() => liveScanFrame(), 1800);
      setTimeout(() => liveScanFrame(), 500);
    }

    async function enumerateVideoDevices() {
      const select = document.getElementById('camera-device-select');
      const container = document.getElementById('camera-select-container');
      if (!select || !container || !navigator.mediaDevices?.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        if (!cameras.length) {
          container.style.display = 'none';
          return;
        }
        select.innerHTML = '';
        cameras.forEach((device, index) => {
          const opt = document.createElement('option');
          opt.value = device.deviceId;
          opt.textContent = device.label || `Camera ${index + 1}`;
          if (device.deviceId && device.deviceId === activeDeviceId) opt.selected = true;
          select.appendChild(opt);
        });
        container.style.display = cameras.length > 1 ? 'block' : 'none';
      } catch (err) {
        console.warn('[CAMERA] enumerateDevices:', err);
      }
    }

    async function startCameraMode(facingMode = 'environment', deviceId = null) {
      stopLiveAutoScan();
      if (currentStream) {
        currentStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
        currentStream = null;
      }

      const video = document.getElementById('camera-stream');
      const container = document.getElementById('camera-container');
      const msg = document.getElementById('camera-message');
      const status = document.getElementById('camera-status-text');
      const flipBtn = document.getElementById('flip-camera-btn');
      if (!video || !container) return;

      if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(location.hostname)) {
        container.style.display = 'block';
        if (msg) msg.textContent = '⚠ Camera needs HTTPS or http://127.0.0.1:5000';
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia) {
        container.style.display = 'block';
        if (msg) msg.textContent = '⚠ Browser camera API unavailable. Use Chrome/Edge.';
        return;
      }

      container.style.display = 'block';
      document.getElementById('drop-zone').style.display = 'none';
      if (status) status.textContent = 'Starting camera...';
      if (msg) msg.textContent = '⏳ Camera permission అడుగుతుంది — Allow నొక్కండి.';

      // First request: generic webcam. This works better on Windows laptops than
      // requiring an exact environment/user facingMode.
      const attempts = [];
      if (deviceId) attempts.push({video:{deviceId:{exact:deviceId}, width:{ideal:1280}, height:{ideal:720}}, audio:false});
      attempts.push({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
      attempts.push({video:{facingMode:{ideal:facingMode},width:{ideal:1280},height:{ideal:720}},audio:false});
      attempts.push({video:true,audio:false});

      let stream = null;
      let lastErr = null;
      for (const constraints of attempts) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (stream) break;
        } catch (err) {
          lastErr = err;
        }
      }

      if (!stream) {
        let text = '⚠ Camera could not be opened.';
        if (lastErr?.name === 'NotAllowedError' || lastErr?.name === 'PermissionDeniedError') {
          text = '⚠ Camera blocked. Chrome address bar → Camera → Allow → reload.';
        } else if (lastErr?.name === 'NotFoundError') {
          text = '⚠ No webcam found. Connect/enable the laptop webcam.';
        } else if (lastErr?.name === 'NotReadableError') {
          text = '⚠ Camera is busy. Close Camera/Teams/Zoom/EShare and try again.';
        }
        if (msg) msg.textContent = text;
        if (status) status.textContent = 'Camera Error';
        return;
      }

      currentStream = stream;
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings?.() || {};
      activeDeviceId = settings.deviceId || deviceId || null;
      currentFacingMode = facingMode;

      video.srcObject = stream;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;
      video.setAttribute('playsinline','true');
      video.setAttribute('autoplay','true');
      video.style.display = 'block';
      video.style.width = '100%';
      video.style.maxHeight = '420px';
      video.style.objectFit = 'contain';
      video.style.background = '#050b0f';
      video.style.transform = facingMode === 'user' ? 'scaleX(-1)' : 'none';

      try { await video.play(); } catch (_) {}

      const deadline = Date.now() + 5000;
      while ((!video.videoWidth || !video.videoHeight) && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 200));
        try { await video.play(); } catch (_) {}
      }

      if (!video.videoWidth || !video.videoHeight) {
        if (msg) msg.textContent = '⚠ Camera opened but no video frames arrived. Select another webcam from the camera list.';
        if (status) status.textContent = 'No Video Frames';
        if (currentStream) currentStream.getTracks().forEach(t => { try { t.stop(); } catch (_) {} });
        currentStream = null;
        return;
      }

      if (flipBtn) flipBtn.style.display = 'inline-flex';
      if (status) status.textContent = facingMode === 'user' ? '🤳 Front Camera Active' : '📷 Camera Active';
      if (msg) msg.textContent = '🟢 Live AI Scan ON — upload cheyyalsina avasaram ledu. Fish/Prawn ni frame lo clear ga pettandi.';

      await enumerateVideoDevices();
      startLiveAutoScan();
    }

    function flipCamera() {
      startCameraMode(currentFacingMode === 'user' ? 'environment' : 'user');
    }

    function switchCameraDevice(deviceId) {
      if (deviceId) startCameraMode(currentFacingMode, deviceId);
    }

    function stopCamera() {
      stopLiveAutoScan();
      if (currentStream) {
        currentStream.getTracks().forEach(track => { try { track.stop(); } catch (_) {} });
        currentStream = null;
      }
      const video = document.getElementById('camera-stream');
      if (video) {
        try { video.pause(); } catch (_) {}
        video.srcObject = null;
        video.style.transform = 'none';
      }
      const container = document.getElementById('camera-container');
      const flipBtn = document.getElementById('flip-camera-btn');
      const dropZone = document.getElementById('drop-zone');
      if (container) container.style.display = 'none';
      if (flipBtn) flipBtn.style.display = 'none';
      if (dropZone) dropZone.style.display = 'block';
    }

    async function liveScanFrame() {
      if (liveScanBusy || !currentStream) return;
      const video = document.getElementById('camera-stream');
      const canvas = document.getElementById('camera-canvas');
      if (!video || !canvas || !video.videoWidth || !video.videoHeight) return;

      liveScanBusy = true;
      try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d', {alpha:false});
        if (!ctx) return;
        ctx.setTransform(1,0,0,1,0,0);
        ctx.drawImage(video,0,0,canvas.width,canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve,'image/jpeg',0.88));
        if (!blob) return;

        currentSelectedFile = new File([blob],`live_camera_${Date.now()}.jpg`,{type:'image/jpeg'});
        await submitDiseaseDetection(null,true);
      } catch (err) {
        console.warn('[LIVE SCAN]', err);
      } finally {
        liveScanBusy = false;
      }
    }

    async function capturePhoto() {
      const video = document.getElementById('camera-stream');
      const canvas = document.getElementById('camera-canvas');
      if (!video || !canvas || !currentStream || !video.videoWidth || !video.videoHeight) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d', {alpha:false});
      if (!ctx) return;
      ctx.setTransform(1,0,0,1,0,0);
      ctx.drawImage(video,0,0,canvas.width,canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) return;
        currentSelectedFile = new File([blob],`camera_capture_${Date.now()}.jpg`,{type:'image/jpeg'});
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(currentSelectedFile);
        try { document.getElementById('image-input').files = dataTransfer.files; } catch (_) {}
        const preview = document.getElementById('preview-img');
        if (preview) preview.src = canvas.toDataURL('image/jpeg',0.95);
        const box = document.getElementById('preview-box');
        if (box) box.style.display = 'block';
        stopCamera();
        submitDiseaseDetection(null,false);
      },'image/jpeg',0.95);
    }

    // SAMPLE PHOTOS & FILE UPLOAD
    // =========================================================================
    let currentSelectedFile = null;

    async function loadSample(sampleKey, species) {
      document.getElementById('species-select').value = species;
      const loader = document.getElementById('detect-loading');
      const placeholder = document.getElementById('detect-placeholder');
      const panel = document.getElementById('detect-result-panel');
      const errorBanner = document.getElementById('detect-error');
      placeholder.style.display = 'none';
      panel.style.display = 'none';
      errorBanner.style.display = 'none';
      if (!silent) loader.style.display = 'block';

      try {
        const res = await fetch(`/api/sample/${sampleKey}`);
        if (!res.ok) throw new Error("Sample file not found on server.");
        const blob = await res.blob();
        const file = new File([blob], `${sampleKey}.jpg`, { type: 'image/jpeg' });
        currentSelectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
          previewImg.src = e.target.result;
          previewBox.style.display = 'block';
        };
        reader.readAsDataURL(file);

        setTimeout(() => {
          document.getElementById('disease-form').dispatchEvent(new Event('submit'));
        }, 120);
      } catch (err) {
        loader.style.display = 'none';
        alert("Failed to load sample: " + err.message);
      }
    }

    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');

    dropZone.addEventListener('click', () => imageInput.click());

    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect(files[0]);
      }
    });

    imageInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
      }
    });

    function handleFileSelect(file) {
      currentSelectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewBox.style.display = 'block';
      };
      reader.readAsDataURL(file);
    }

    // =========================================================================
    // DETECTION SUBMISSION & RESULT RENDERING
    // =========================================================================
    async function submitDiseaseDetection(e, silent = false) {
      if (e && e.preventDefault) e.preventDefault();
      const file = currentSelectedFile || (imageInput.files && imageInput.files[0]);
      if (!file) {
        if (!silent) alert("Please select or capture a fish/prawn image first.");
        return;
      }

      const species = document.getElementById('species-select').value;
      const formData = new FormData();
      formData.append('image', file);
      formData.append('species', species);

      const placeholder = document.getElementById('detect-placeholder');
      const loader = document.getElementById('detect-loading');
      const panel = document.getElementById('detect-result-panel');
      const errorBanner = document.getElementById('detect-error');

      placeholder.style.display = 'none';
      panel.style.display = 'none';
      errorBanner.style.display = 'none';
      loader.style.display = 'block';

      try {
        // One endpoint, but the selected species is sent explicitly.
        // Backend is locked to that species and never auto-switches.
        const endpoint = '/predict/auto-disease';
        const res = await fetch(endpoint, {
          method: 'POST',
          body: formData,
          credentials: 'same-origin',
          cache: 'no-store'
        });

        // Read text first. A Flask 4xx/5xx HTML response should not be converted
        // into the misleading generic "Network error" message.
        const rawText = await res.text();
        let data = null;
        try { data = rawText ? JSON.parse(rawText) : null; } catch (_) { data = null; }

        if (!res.ok) {
          const serverMessage = data?.message_en || data?.message || data?.error ||
            `Detection server returned HTTP ${res.status}.`;
          throw new Error(serverMessage);
        }
        if (!data) {
          throw new Error('Detection backend returned an invalid response.');
        }

        if (!silent) loader.style.display = 'none';

        // STRICT REQUIREMENT: If NOT a fish/prawn, or human detected, reject!
        if (!data.success || !data.is_valid_aquatic) {
          if (silent) {
            // During live scanning, do not flash an error every 1.8 seconds.
            panel.style.display = 'none';
            return;
          }
          panel.style.display = 'none';
          errorBanner.style.display = 'block';

          let msg = "";
          if (currentLang === 'te') msg = data.message_te || data.message;
          else if (currentLang === 'hi') msg = data.message_hi || data.message_en || data.message;
          else if (currentLang === 'tel') msg = data.message_tel || data.message;
          else msg = data.message_en || data.message;

          document.getElementById('detect-error-msg').innerText = msg || "❌ No Fish or Prawn detected! Only fish and prawn images are supported.";

          // Reset outputs
          document.getElementById('annotated-img').src = '';
          document.getElementById('annotated-container').style.display = 'none';
          document.getElementById('spots-card').style.display = 'none';
          document.getElementById('res-class-name').innerText = '-';
          document.getElementById('res-confidence').innerText = '-';

          // Speak error warning
          const speechMsg = (currentLang === 'hi') ? (data.voice_script_hi || msg) : ((currentLang === 'en') ? (data.voice_script_en || msg) : (data.voice_script_te || msg));
          if (!silent) speakText(speechMsg);
          lastDetectionData = null;
          return;
        }

        // SECOND HARD LOCK IN THE BROWSER:
        // The backend is expected to return exactly the selected species.
        // Never display a result for the opposite species.
        if (String(data.species || '').toLowerCase() !== String(species).toLowerCase()) {
          panel.style.display = 'none';
          errorBanner.style.display = 'block';
          document.getElementById('detect-error-msg').innerText =
            `❌ Species mismatch blocked. You selected ${species === 'fish' ? 'Fish' : 'Prawn'}, but the backend returned another species.`;
          document.getElementById('annotated-img').src = '';
          document.getElementById('annotated-container').style.display = 'none';
          document.getElementById('spots-card').style.display = 'none';
          document.getElementById('res-class-name').innerText = '-';
          document.getElementById('res-confidence').innerText = '-';
          lastDetectionData = null;
          return;
        }

        // Valid Fish / Prawn detected!
        errorBanner.style.display = 'none';
        lastDetectionData = data;
        renderDetectionResult(data);

      } catch (err) {
        if (!silent) {
          loader.style.display = 'none';
          errorBanner.style.display = 'block';
          const message = (err && err.message)
            ? err.message
            : 'Could not reach the detection backend. Make sure python app.py is still running.';
          document.getElementById('detect-error-msg').innerText = '❌ ' + message;
        }
      }
    }

    function openBuyModalById(prodId) {
      const item = CATALOG_DATA.find(x => x.id === prodId);
      if (item) {
        let name = item.title_te;
        if (currentLang === 'en') name = item.title_en;
        else if (currentLang === 'hi') name = item.title_hi || item.title_en;
        else if (currentLang === 'tel') name = item.title_tel;
        openBuyModal(name, item.price);
      } else {
        switchTab('tab-buying');
      }
    }

    function createLinkPill(link) {
      let title = link.title_en;
      let badge = link.badge_en;
      if (currentLang === 'te') {
        title = link.title_te || title;
        badge = link.badge_te || badge;
      } else if (currentLang === 'hi') {
        title = link.title_hi || title;
        badge = link.badge_hi || badge;
      } else if (currentLang === 'tel') {
        title = link.title_tel || title;
        badge = link.badge_tel || badge;
      }

      const isInternal = link.url && link.url.startsWith('internal:');
      const pill = document.createElement(isInternal ? 'button' : 'a');
      pill.className = `provider-link-pill ${isInternal ? 'store-pill' : ''}`;

      if (isInternal) {
        pill.type = 'button';
        const prodId = link.url.replace('internal:', '');
        pill.onclick = (e) => {
          e.preventDefault();
          openBuyModalById(prodId);
        };
      } else {
        pill.href = link.url;
        pill.target = '_blank';
        pill.rel = 'noopener noreferrer';
      }

      let badgeClass = 'badge-external';
      if (isInternal) badgeClass = 'badge-store-order';
      else if (link.type === 'govt') badgeClass = 'badge-govt';

      pill.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px; overflow:hidden; min-width:0;">
          <i class="${link.icon || (isInternal ? 'fa-solid fa-cart-shopping' : 'fa-solid fa-arrow-up-right-from-square')}" style="color:${isInternal ? '#10b981' : (link.type === 'govt' ? '#fbbf24' : 'var(--primary)')}; font-size:0.95rem; flex-shrink:0;"></i>
          <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-weight:600;">${title}</span>
        </div>
        <span class="provider-pill-badge ${badgeClass}">${badge}</span>
      `;
      return pill;
    }

    // FINAL CLIENT-SIDE DISEASE LOCALIZATION
    // Uses the clean uploaded preview, not the already-annotated output.
    // For Fish + Bacterial Diseases, locate the visible red/hemorrhagic
    // lesion directly and use that box for display.
    function findFishBacterialUlcerBox(canvas) {
      // FINAL DISPLAY LOCALIZER
      // This operates on the CLEAN uploaded/captured image. It does not use
      // the raw YOLO disease bbox. YOLO supplies the disease CLASS; this
      // function supplies the visible lesion LOCATION for display.
      try {
        const MAX_SIDE = 640;
        const scale = Math.min(1, MAX_SIDE / Math.max(canvas.width, canvas.height));
        const w = Math.max(1, Math.round(canvas.width * scale));
        const h = Math.max(1, Math.round(canvas.height * scale));

        const source = document.createElement('canvas');
        source.width = w;
        source.height = h;
        const sctx = source.getContext('2d', { willReadFrequently: true });
        if (!sctx) return null;
        sctx.drawImage(canvas, 0, 0, w, h);

        const pix = sctx.getImageData(0, 0, w, h).data;
        const mask = new Uint8Array(w * h);

        // Red / hemorrhagic ulcer detector.
        // Broad enough for dark-red and bright-red ulcer tissue, but much
        // stricter than a generic red detector so yellow/orange scales don't win.
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            const r = pix[i], g = pix[i + 1], b = pix[i + 2];
            const maxc = Math.max(r, g, b);
            const minc = Math.min(r, g, b);
            const sat = maxc > 0 ? (maxc - minc) / maxc : 0;

            const redDominant =
              r >= 78 &&
              (r - g) >= 22 &&
              (r - b) >= 18 &&
              r >= g * 1.08 &&
              r >= b * 1.10 &&
              sat >= 0.18;

            const darkRed =
              r >= 48 &&
              r > g * 1.14 &&
              r > b * 1.12 &&
              (r - g) >= 16 &&
              (r - b) >= 14;

            if (redDominant || darkRed) mask[y * w + x] = 1;
          }
        }

        // Morphological close: merge holes and contiguous red pieces of the
        // same ulcer into one lesion.
        const joined = new Uint8Array(mask.length);
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            let n = 0;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                n += mask[(y + dy) * w + (x + dx)];
              }
            }
            if (n >= 3) joined[y * w + x] = 1;
          }
        }

        // One more horizontal/vertical joining pass handles a white highlight
        // that cuts through the middle of a red ulcer.
        const closed = new Uint8Array(joined.length);
        for (let y = 2; y < h - 2; y++) {
          for (let x = 2; x < w - 2; x++) {
            let n = 0;
            for (let dy = -2; dy <= 2; dy++) {
              for (let dx = -2; dx <= 2; dx++) {
                n += joined[(y + dy) * w + (x + dx)];
              }
            }
            if (n >= 7) closed[y * w + x] = 1;
          }
        }

        const visited = new Uint8Array(closed.length);
        const qx = new Int32Array(w * h);
        const qy = new Int32Array(w * h);
        let best = null;

        for (let sy = 0; sy < h; sy++) {
          for (let sx = 0; sx < w; sx++) {
            const startPos = sy * w + sx;
            if (!closed[startPos] || visited[startPos]) continue;

            let head = 0, tail = 0;
            qx[tail] = sx;
            qy[tail] = sy;
            tail++;
            visited[startPos] = 1;

            let area = 0;
            let minX = sx, minY = sy, maxX = sx, maxY = sy;

            while (head < tail) {
              const x = qx[head], y = qy[head];
              head++;
              area++;
              if (x < minX) minX = x;
              if (y < minY) minY = y;
              if (x > maxX) maxX = x;
              if (y > maxY) maxY = y;

              for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                  if (dx === 0 && dy === 0) continue;
                  const xx = x + dx, yy = y + dy;
                  if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                  const pos = yy * w + xx;
                  if (closed[pos] && !visited[pos]) {
                    visited[pos] = 1;
                    qx[tail] = xx;
                    qy[tail] = yy;
                    tail++;
                  }
                }
              }
            }

            const bw = maxX - minX + 1;
            const bh = maxY - minY + 1;
            const imageArea = w * h;
            if (area < Math.max(40, imageArea * 0.0005)) continue;
            if (area > imageArea * 0.45) continue;
            if (bw < 10 || bh < 10) continue;

            const fill = area / Math.max(1, bw * bh);
            // A true ulcer normally has a substantial filled region. This
            // prevents isolated red dots from beating a large lesion.
            if (fill < 0.06) continue;

            const score = area * (0.55 + fill) * (1 + Math.min(2, Math.sqrt(area) / 30));

            if (!best || score > best.score) {
              best = { minX, minY, maxX, maxY, area, fill, score };
            }
          }
        }

        if (!best) return null;

        // Convert back to the real image coordinates with a small border pad.
        const padX = Math.max(8, Math.round((best.maxX - best.minX + 1) * 0.05));
        const padY = Math.max(8, Math.round((best.maxY - best.minY + 1) * 0.05));

        const x1 = Math.max(0, (best.minX - padX) / scale);
        const y1 = Math.max(0, (best.minY - padY) / scale);
        const x2 = Math.min(canvas.width, (best.maxX + 1 + padX) / scale);
        const y2 = Math.min(canvas.height, (best.maxY + 1 + padY) / scale);

        if (x2 - x1 < canvas.width * 0.04 || y2 - y1 < canvas.height * 0.04) {
          return null;
        }
        return [x1, y1, x2, y2];
      } catch (err) {
        console.warn('[CLIENT EXACT BACTERIAL LOCALIZER]', err);
        return null;
      }
    }

    function drawClientBoxes(data) {
      const img = document.getElementById('annotated-img');
      const container = document.getElementById('annotated-container');
      const src = document.getElementById('preview-img')?.src;

      if (!src) {
        container.style.display = 'none';
        return;
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const base = new Image();

      base.onload = () => {
        canvas.width = base.naturalWidth;
        canvas.height = base.naturalHeight;
        ctx.drawImage(base, 0, 0);

        // Fish/Prawn organism boxes.
        for (const ob of (data.object_detections || [])) {
          const b = ob.bbox || [];
          if (b.length !== 4) continue;

          ctx.strokeStyle = '#00d9ff';
          ctx.lineWidth = Math.max(4, Math.round(canvas.width / 190));
          ctx.strokeRect(
            b[0], b[1],
            b[2] - b[0],
            b[3] - b[1]
          );

          ctx.font = `bold ${Math.max(15, Math.round(canvas.width / 48))}px Arial`;
          const label =
            `${ob.label || 'Selected species'} ` +
            `${(Number(ob.confidence || 0) * 100).toFixed(1)}%`;

          const tw = ctx.measureText(label).width + 16;
          const th = 28;
          const ty = Math.max(0, b[1] - th);

          ctx.fillStyle = 'rgba(0,127,153,.94)';
          ctx.fillRect(b[0], ty, tw, th);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, b[0] + 8, ty + 20);
        }

        // Copy disease detections.
        const finalDiseaseDetections = (data.detections || [])
          .filter(
            d => !['Healthy', 'Healthy Fish'].includes(
              String(d.class_name || '')
            )
          )
          .map(d => ({ ...d }));

        // ==========================================================
        // HARD FINAL DISPLAY OVERRIDE
        // ==========================================================
        // For Fish + Bacterial Diseases, DO NOT display the raw YOLO box.
        // Find the actual visible ulcer on the clean uploaded image.
        for (const d of finalDiseaseDetections) {
          const cls = String(d.class_name || '')
            .trim()
            .toLowerCase();

          if (String(data.species || '').toLowerCase() === 'fish' && cls === 'bacterial diseases') {
            const exactBox = findFishBacterialUlcerBox(canvas);

            if (exactBox) {
              d.original_yolo_bbox = d.bbox || null;
              d.bbox = exactBox;
              d.localization_source =
                'CLIENT_EXACT_VISIBLE_BACTERIAL_ULCER';
            }
          }
        }

        // Draw ONLY FINAL disease boxes.
        finalDiseaseDetections.forEach((d, idx) => {
          const b = d.bbox || [];
          if (b.length !== 4) return;

          ctx.strokeStyle = '#ff3344';
          ctx.lineWidth = Math.max(
            5,
            Math.round(canvas.width / 170)
          );

          ctx.strokeRect(
            b[0], b[1],
            b[2] - b[0],
            b[3] - b[1]
          );

          const label =
            `${d.class_name || 'Disease'} ` +
            `${(Number(d.confidence || 0) * 100).toFixed(1)}%`;

          ctx.font =
            `bold ${Math.max(16, Math.round(canvas.width / 45))}px Arial`;

          const tw = ctx.measureText(label).width + 16;
          const th = 30;
          const ty = Math.max(0, b[1] - th);

          ctx.fillStyle = 'rgba(255,51,68,.92)';
          ctx.fillRect(b[0], ty, tw, th);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, b[0] + 8, ty + 21);
        });

        // Keep backend data synchronized with what is visible.
        data.detections = finalDiseaseDetections;
        data.final_bbox_source = 'clean-image-disease-localizer';

        if (
          Array.isArray(data.spots) &&
          finalDiseaseDetections.length
        ) {
          finalDiseaseDetections.forEach((d, idx) => {
            const sp = data.spots[idx];
            if (!sp || !d.bbox || d.bbox.length !== 4) return;

            sp.box_coords = {
              x1: d.bbox[0],
              y1: d.bbox[1],
              x2: d.bbox[2],
              y2: d.bbox[3]
            };
          });
        }

        img.src = canvas.toDataURL('image/jpeg', .94);
        container.style.display = 'block';
      };

      base.src = src;
    }

    function renderDetectionResult(data) {
      const panel = document.getElementById('detect-result-panel');
      panel.style.display = 'block';

      const info = data.clinical_data || {};

      // 1. ALWAYS redraw from the clean uploaded preview using the FINAL
      // corrected detection boxes. Do not prefer a stale backend annotation.
      if (data.detections && data.detections.length > 0) {
        drawClientBoxes(data);
      } else if (data.annotated_image) {
        document.getElementById('annotated-img').src = data.annotated_image;
        document.getElementById('annotated-container').style.display = 'block';
      } else {
        document.getElementById('annotated-container').style.display = 'none';
      }

      // 2. Render Detected Spots List with localized body region
      const spotsList = document.getElementById('spots-list');
      spotsList.innerHTML = '';
      if (data.spots && data.spots.length > 0) {
        data.spots.forEach(sp => {
          let loc = sp.location_en;
          if (currentLang === 'te') loc = sp.location_te;
          else if (currentLang === 'hi') loc = sp.location_hi || sp.location_en;
          else if (currentLang === 'tel') loc = sp.location_tel || sp.location_te;

          const item = document.createElement('div');
          item.className = 'spot-item';
          item.innerHTML = `
            <span><i class="fa-solid fa-location-dot" style="color:var(--rose); margin-right:6px;"></i> <strong>Spot ${sp.spot_num}:</strong> ${loc}</span>
            <span style="color:var(--emerald); font-weight:700;">${sp.confidence_pct}</span>
          `;
          spotsList.appendChild(item);
        });
        document.getElementById('spots-card').style.display = 'block';
      } else {
        document.getElementById('spots-card').style.display = 'none';
      }

      // 3. Condition name & Confidence
      let conditionTitle = info.title_en || data.predicted_class;
      if (currentLang === 'te') conditionTitle = info.title_te || conditionTitle;
      else if (currentLang === 'hi') conditionTitle = info.title_hi || conditionTitle;
      else if (currentLang === 'tel') conditionTitle = info.title_tel || conditionTitle;

      document.getElementById('res-class-name').innerText = conditionTitle;
      document.getElementById('res-confidence').innerText = data.confidence_pct;

      const badge = document.getElementById('result-status-badge');
      badge.innerText = `${info.severity || 'Detected'} • ${data.species.toUpperCase()}`;
      badge.className = `result-badge ${info.is_healthy ? 'safe' : (info.severity === 'Critical' ? 'danger' : 'warning')}`;

      // 4. MANDHULU (MEDICINES) & FOOD (MEETHA) GUIDANCE
      const diseaseView = document.getElementById('disease-treatment-view');
      const healthyView = document.getElementById('healthy-selling-view');

      if (info.is_healthy) {
        diseaseView.style.display = 'none';
        healthyView.style.display = 'block';

        let statusText = info.market_status_en;
        let harvestText = info.harvest_guide_en;
        let foodText = info.food_guide_en;
        if (currentLang === 'te') {
          statusText = info.market_status_te;
          harvestText = info.harvest_guide_te;
          foodText = info.food_guide_te;
        } else if (currentLang === 'hi') {
          statusText = info.market_status_hi || statusText;
          harvestText = info.harvest_guide_hi || harvestText;
          foodText = info.food_guide_hi || foodText;
        } else if (currentLang === 'tel') {
          statusText = info.market_status_tel;
          harvestText = info.harvest_guide_tel;
          foodText = info.food_guide_tel;
        }

        document.getElementById('res-selling-status').innerText = statusText;
        document.getElementById('res-harvest-guide').innerText = harvestText;
        document.getElementById('res-healthy-food').innerText = foodText;

        const ratesList = document.getElementById('res-selling-rates-list');
        ratesList.innerHTML = '';
        if (info.selling_rates) {
          Object.entries(info.selling_rates).forEach(([item, obj]) => {
            const rDiv = document.createElement('div');
            rDiv.style.cssText = 'display:flex; justify-content:space-between; font-size:0.86rem; padding:3px 0; border-bottom:1px solid rgba(255,255,255,0.05);';
            rDiv.innerHTML = `<span>${item}:</span> <strong style="color:var(--primary);">${obj.price_range}</strong>`;
            ratesList.appendChild(rDiv);
          });
        }

        // Render Govt & Market Trading Links
        const mktBox = document.getElementById('res-market-links-box');
        const mktList = document.getElementById('res-market-links-list');
        if (mktBox && mktList) {
          mktList.innerHTML = '';
          const mktLinks = info.market_links || [];
          if (mktLinks.length > 0) {
            mktBox.style.display = 'block';
            mktLinks.forEach(lnk => mktList.appendChild(createLinkPill(lnk)));
          } else {
            mktBox.style.display = 'none';
          }
        }

        // Render Healthy Feeds Links
        const healthyFeedBox = document.getElementById('res-healthy-food-links-box');
        const healthyFeedList = document.getElementById('res-healthy-food-links-list');
        if (healthyFeedBox && healthyFeedList) {
          healthyFeedList.innerHTML = '';
          const fLinks = info.food_links || [];
          if (fLinks.length > 0) {
            healthyFeedBox.style.display = 'block';
            fLinks.forEach(lnk => healthyFeedList.appendChild(createLinkPill(lnk)));
          } else {
            healthyFeedBox.style.display = 'none';
          }
        }
      } else {
        healthyView.style.display = 'none';
        diseaseView.style.display = 'block';

        let medText = info.medicine_en;
        let foodText = info.food_guide_en;
        if (currentLang === 'te') {
          medText = info.medicine_te;
          foodText = info.food_guide_te;
        } else if (currentLang === 'hi') {
          medText = info.medicine_hi || medText;
          foodText = info.food_guide_hi || foodText;
        } else if (currentLang === 'tel') {
          medText = info.medicine_tel;
          foodText = info.food_guide_tel;
        }

        document.getElementById('res-medicine-text').innerText = medText;
        document.getElementById('res-food-text').innerText = foodText;

        // Render Medicine Purchase & Store Links
        const medBox = document.getElementById('res-medicine-links-box');
        const medList = document.getElementById('res-medicine-links-list');
        if (medBox && medList) {
          medList.innerHTML = '';
          const medLinks = info.medicine_links || [];
          if (medLinks.length > 0) {
            medBox.style.display = 'block';
            medLinks.forEach(lnk => medList.appendChild(createLinkPill(lnk)));
          } else {
            medBox.style.display = 'none';
          }
        }

        // Render Food & Feed Purchase Links
        const foodBox = document.getElementById('res-food-links-box');
        const foodList = document.getElementById('res-food-links-list');
        if (foodBox && foodList) {
          foodList.innerHTML = '';
          const foodLinks = info.food_links || [];
          if (foodLinks.length > 0) {
            foodBox.style.display = 'block';
            foodLinks.forEach(lnk => foodList.appendChild(createLinkPill(lnk)));
          } else {
            foodBox.style.display = 'none';
          }
        }
      }

      // Voice Script in active language
      let voiceScript = info.voice_script_en || `${conditionTitle} detected.`;
      if (currentLang === 'te' || currentLang === 'tel') {
        voiceScript = info.voice_script_te || voiceScript;
      } else if (currentLang === 'hi') {
        voiceScript = info.voice_script_hi || voiceScript;
      }

      document.getElementById('speak-diagnosis-btn').onclick = () => speakText(voiceScript);
    }

    // =========================================================================
    // DISSOLVED OXYGEN FORM SUBMIT
    // =========================================================================
    document.getElementById('do-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const form = e.target;
      const payload = {
        species: form.species.value,
        water_temp_C: form.water_temp_C.value,
        pH: form.pH.value,
        salinity_ppt: form.salinity_ppt.value,
        ammonia_mg_L: form.ammonia_mg_L.value,
        nitrite_mg_L: form.nitrite_mg_L.value,
        turbidity_NTU: form.turbidity_NTU.value,
        feed_kg: form.feed_kg.value,
        rainfall_mm: form.rainfall_mm.value,
        aerator_on: form.aerator_on.value
      };

      const placeholder = document.getElementById('do-placeholder');
      const loader = document.getElementById('do-loading');
      const panel = document.getElementById('do-result-panel');

      placeholder.style.display = 'none';
      panel.style.display = 'none';
      loader.style.display = 'block';

      try {
        const res = await fetch('/predict/do', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        loader.style.display = 'none';

        if (data.success) {
          lastDOData = data;
          renderDOResult(data);
        } else {
          alert(data.message || "Failed to calculate DO.");
        }
      } catch (err) {
        loader.style.display = 'none';
        alert("Error connecting to DO predictor.");
      }
    });

    function renderDOResult(data) {
      const panel = document.getElementById('do-result-panel');
      panel.style.display = 'block';
      document.getElementById('res-do-value').innerText = `${data.predicted_do_mg_l} mg/L`;
      const badge = document.getElementById('res-do-status');
      badge.innerText = data.status;
      badge.style.background = data.status_color + '33';
      badge.style.color = data.status_color;
      badge.style.borderColor = data.status_color;

      let advice = data.advice_en;
      let aerator = data.aerator_en;
      if (currentLang === 'te') {
        advice = data.advice_te;
        aerator = data.aerator_te;
      } else if (currentLang === 'hi') {
        advice = data.advice_hi || data.advice_en;
        aerator = data.aerator_hi || data.aerator_en;
      } else if (currentLang === 'tel') {
        advice = data.advice_tel;
        aerator = data.aerator_tel;
      }

      document.getElementById('res-do-advice').innerText = advice;
      document.getElementById('res-do-aerator').innerText = aerator;

      const voice = (currentLang === 'hi') ? data.voice_script_hi : (data.voice_script_te || advice);
      if (voice) speakText(voice);
    }

    // =========================================================================
    // CHAT ASSISTANT — ONLY PLACE WHERE VOICE INPUT IS ENABLED
    // The backend detects the language of the actual question and returns the answer
    // in that same language: English -> English, Telugu -> Telugu, Hindi -> Hindi.
    const assistantVoiceLang = document.getElementById('assistant-voice-lang');

    document.getElementById('chat-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const input = document.getElementById('chat-input');
      const question = input.value.trim();
      if (!question) return;

      appendChatMessage('Farmer', question, '#00f0ff');
      input.value = '';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: question,
            lang: currentLang,
            voice_lang: assistantVoiceLang ? assistantVoiceLang.value : currentLang
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'Chat service failed.');

        const replyLang = data.reply_language || 'en';
        const reply = replyLang === 'te' ? data.reply_telugu
                    : replyLang === 'hi' ? data.reply_hindi
                    : data.reply_english;

        appendChatMessage('AquaCare Assistant', reply || data.reply_telugu || data.reply_hindi || data.reply_english || 'No answer.', '#34d399');

        // Speak the answer in the language of the question, not the UI language.
        if (data.voice_text) speakText(data.voice_text, replyLang);
      } catch (err) {
        appendChatMessage('System', 'Chat service offline. Please try again.', '#ef4444');
      }
    });

    function appendChatMessage(sender, text, color) {
      const box = document.getElementById('chat-box');
      const div = document.createElement('div');
      div.style.cssText = `background:rgba(255,255,255,0.05); border:1px solid var(--card-border); padding:10px 14px; border-radius:10px; max-width:85%; margin-bottom:8px; line-height:1.4;`;
      div.innerHTML = `<strong style="color:${color};">${sender}:</strong> <span style="white-space:pre-line;"></span>`;
      div.querySelector('span').textContent = String(text ?? '');
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    // Browser speech recognition is available only inside Farmer Assistant.
    const micBtn = document.getElementById('mic-btn');
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (micBtn && SpeechRecognitionAPI) {
      const recognizer = new SpeechRecognitionAPI();
      recognizer.continuous = false;
      recognizer.interimResults = false;
      recognizer.maxAlternatives = 1;

      micBtn.addEventListener('click', () => {
        try {
          const vLang = assistantVoiceLang ? assistantVoiceLang.value : currentLang;
          recognizer.lang = vLang === 'hi' ? 'hi-IN' : (vLang === 'te' ? 'te-IN' : 'en-IN');
          recognizer.start();
          micBtn.style.color = '#ef4444';
          micBtn.title = 'Listening...';
        } catch (e) {
          try { recognizer.stop(); } catch (_) {}
        }
      });

      recognizer.onresult = (e) => {
        const spoken = e.results?.[0]?.[0]?.transcript || '';
        document.getElementById('chat-input').value = spoken;
        micBtn.style.color = '';
        micBtn.title = 'Speak only in Farmer Assistant';
        if (spoken) document.getElementById('chat-form').requestSubmit();
      };

      recognizer.onerror = () => {
        micBtn.style.color = '';
        micBtn.title = 'Speak only in Farmer Assistant';
      };
      recognizer.onend = () => {
        micBtn.style.color = '';
        micBtn.title = 'Speak only in Farmer Assistant';
      };
    } else if (micBtn) {
      micBtn.addEventListener('click', () => {
        alert('Voice input is not supported in this browser. Use Chrome or Edge.');
      });
    }

    // =========================================================================
    // SELLING REVENUE CALCULATOR
    // =========================================================================
    function updateCalcRate() {
      const select = document.getElementById('calc-species-select');
      document.getElementById('calc-price').value = select.value;
      calculateTotalIncome();
    }

    function calculateTotalIncome() {
      const weight = parseFloat(document.getElementById('calc-weight').value) || 0;
      const price = parseFloat(document.getElementById('calc-price').value) || 0;
      const gross = weight * price;
      const profit = gross * 0.40;
      document.getElementById('calc-gross-revenue').innerText = `₹ ${gross.toLocaleString('en-IN')}`;
      document.getElementById('calc-net-profit').innerText = `₹ ${profit.toLocaleString('en-IN')}`;
    }

    // =========================================================================
    // BUYING MODAL & ORDER
    // =========================================================================
    function openBuyModal(productName, price) {
      document.getElementById('modal-product-name').innerText = productName;
      document.getElementById('modal-product-price').innerText = price;
      document.getElementById('order-item-input').value = productName;
      document.getElementById('buy-modal').style.display = 'flex';
    }

    function closeBuyModal() {
      document.getElementById('buy-modal').style.display = 'none';
    }

    async function submitOrder(e) {
      e.preventDefault();
      const payload = {
        product_name: document.getElementById('order-item-input').value,
        farmer_name: document.getElementById('order-farmer-name').value,
        phone: document.getElementById('order-phone').value,
        location: document.getElementById('order-location').value,
        quantity: document.getElementById('order-quantity').value
      };
      try {
        const res = await fetch('/api/order', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
          alert("Order submitted successfully! Our local aqua officer will contact you.");
          closeBuyModal();
          e.target.reset();
        } else {
          alert(data.message || "Failed to submit inquiry.");
        }
      } catch (err) {
        alert("Failed to submit inquiry. Try via WhatsApp.");
      }
    }

    function orderViaWhatsApp() {
      const product = document.getElementById('modal-product-name').innerText;
      const price = document.getElementById('modal-product-price').innerText;
      const name = document.getElementById('order-farmer-name').value || "Farmer";
      const location = document.getElementById('order-location').value || "";
      const qty = document.getElementById('order-quantity').value || "1 unit";
      const text = encodeURIComponent(`Namaskaram! Naku AquaCare app dwara order kavali:\nProduct: ${product}\nPrice: ${price}\nQuantity: ${qty}\nName: ${name}\nLocation: ${location}`);
      window.open(`https://wa.me/919959269947?text=${text}`, '_blank');
    }

    // Species changes invalidate the previous result so a Fish result can never
    // remain visible after the user switches to Prawn (and vice versa).
    document.getElementById('species-select')?.addEventListener('change', () => {
      currentSelectedFile = null;
      if (imageInput) imageInput.value = '';
      const previewBoxEl = document.getElementById('preview-box');
      const panelEl = document.getElementById('detect-result-panel');
      const errorEl = document.getElementById('detect-error');
      if (previewBoxEl) previewBoxEl.style.display = 'none';
      if (panelEl) panelEl.style.display = 'none';
      if (errorEl) errorEl.style.display = 'none';
      lastDetectionData = null;
    });

    window.addEventListener('beforeunload', () => { try { stopCamera(); } catch (_) {} });

    // INITIALIZATION
    const storedLang = localStorage.getItem('aquacare_lang') || 'te';
    const savedLang = ['en', 'te', 'hi'].includes(storedLang) ? storedLang : 'te';
    switchLanguage(savedLang);
    if (assistantVoiceLang) assistantVoiceLang.value = savedLang;
    calculateTotalIncome();
  </script>
</body>
</html>
"""

# ==============================================================================
# 8. FLASK API ENDPOINTS & CONTROLLERS
# ==============================================================================
# Discover true species object gates after all helper functions are defined.
_discover_true_species_object_models()

@app.route("/")
def index():
    models_status = {
        "fish_disease": FISH_DISEASE_MODEL is not None,
        "prawn_disease": PRAWN_DISEASE_MODEL is not None,
        "fish_object": FISH_OBJECT_MODEL is not None,
        "prawn_object": PRAWN_OBJECT_MODEL is not None,
        "fish_do": FISH_DO_MODEL is not None,
        "prawn_do": PRAWN_DO_MODEL is not None,
    }
    return send_from_directory(os.path.join(BASE_DIR, "static_ui"), "index.html")

@app.route("/static/<path:filepath>")
def static_files(filepath: str):
    """Serve static assets (images, css, js) from static_ui/."""
    static_dir = os.path.join(BASE_DIR, "static_ui")
    return send_from_directory(static_dir, filepath)

@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "AquaCare AI Platform v4.7 contextual-assistant",
        "models_loaded": {
            "fish_disease_model": FISH_DISEASE_MODEL is not None,
            "prawn_disease_model": PRAWN_DISEASE_MODEL is not None,
            "fish_object_model": FISH_OBJECT_MODEL is not None,
            "prawn_object_model": PRAWN_OBJECT_MODEL is not None,
            "fish_do_model": FISH_DO_MODEL is not None,
            "prawn_do_model": PRAWN_DO_MODEL is not None,
            "species_gate_model": SPECIES_GATE_MODEL is not None,
            "species_gate_required": True,
            "coco_detector": COCO_DETECTOR is not None
        },
        "assistant": {
            "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
            "gemini_api_key_format_valid": os.getenv("GEMINI_API_KEY", "").strip().startswith("AIza"),
            "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        },
        "model_paths": {
            "fish_weights": FISH_MODEL_PATHS,
            "prawn_weights": PRAWN_MODEL_PATHS,
            "fish_object_weights": FISH_OBJECT_MODEL_PATH,
            "prawn_object_weights": PRAWN_OBJECT_MODEL_PATH,
            "coco_weights": COCO_MODEL_PATH,
            "fish_do_pkl": FISH_DO_PATH,
            "prawn_do_pkl": PRAWN_DO_PATH,
            "species_gate_pkl": SPECIES_GATE_PATH,
        },
        "errors": MODEL_LOAD_ERRORS
    })

@app.errorhandler(413)
def handle_request_too_large(_error):
    return jsonify({
        "success": False,
        "is_valid_aquatic": False,
        "error": "Uploaded image is too large. Please use an image below 32 MB.",
        "message": "Uploaded image is too large. Please use an image below 32 MB."
    }), 413


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    # Always return JSON for API routes so the browser never turns a server-side
    # exception into the misleading generic "Network error" message.
    path = request.path if request else ""
    print(f"[API ERROR] {path}: {type(error).__name__}: {error}")
    if path.startswith("/predict/"):
        return jsonify({
            "success": False,
            "is_valid_aquatic": False,
            "error": f"{type(error).__name__}: {error}",
            "message": "Detection backend error. See terminal for the exact error.",
            "message_en": "Detection backend error. See the terminal for the exact error.",
            "message_te": "Detection backend లో error వచ్చింది. Terminal లో exact error చూడండి.",
            "message_hi": "Detection backend में error आया। Terminal में exact error देखें.",
            "voice_script_en": "Detection backend error.",
            "voice_script_te": "Detection backend లో error వచ్చింది."
        }), 500
    from werkzeug.exceptions import HTTPException
    if isinstance(error, HTTPException):
        return error
    raise error


def _save_uploaded_image(file, prefix: str):
    if file is None or not file.filename:
        return None, {"success": False, "message": "Please upload an image."}
    original = os.path.basename(file.filename)
    ext = os.path.splitext(original)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        return None, {"success": False, "message": "Only JPG, JPEG, PNG and WEBP images are allowed."}
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", original)
    filename = f"{prefix}_{uuid.uuid4().hex}_{safe_name}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        raw = file.read()
        if not raw:
            return None, {"success": False, "message": "Uploaded image is empty."}
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with open(image_path, "wb") as out:
            out.write(raw)
        return image_path, None
    except Exception as exc:
        return None, {"success": False, "message": f"Invalid image: {type(exc).__name__}: {exc}"}

@app.route("/predict/fish-disease", methods=["POST"])
def predict_fish_disease():
    if "image" not in request.files:
        return jsonify({"success": False, "is_valid_aquatic": False, "message": "No fish image uploaded."}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "is_valid_aquatic": False, "message": "Invalid file uploaded."}), 400

    image_path, error = _save_uploaded_image(file, "fish")
    if error:
        return jsonify(error), 400

    result = _attach_result_guidance(analyze_species_disease(image_path, target_species="fish"))
    return jsonify(result)

@app.route("/predict/prawn-disease", methods=["POST"])
def predict_prawn_disease():
    if "image" not in request.files:
        return jsonify({"success": False, "is_valid_aquatic": False, "message": "No prawn image uploaded."}), 400
    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"success": False, "is_valid_aquatic": False, "message": "Invalid file uploaded."}), 400

    image_path, error = _save_uploaded_image(file, "prawn")
    if error:
        return jsonify(error), 400

    result = _attach_result_guidance(analyze_species_disease(image_path, target_species="prawn"))
    return jsonify(result)

@app.route("/predict/auto-disease", methods=["POST"])
def predict_auto_disease():
    """Stable API boundary for the image detector.

    Every failure is converted to JSON so the frontend can show the real reason
    instead of reporting a generic network error. The selected species is copied
    into the analysis call and is never auto-switched.
    """
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "is_valid_aquatic": False,
                "message": "No image uploaded."
            }), 400

        requested_species = str(request.form.get("species", "")).strip().lower()
        if requested_species not in {"fish", "prawn"}:
            return jsonify({
                "success": False,
                "is_valid_aquatic": False,
                "message": "Please select Fish or Prawn before detection.",
                "message_en": "Please select Fish or Prawn before detection.",
                "message_te": "Detection ముందు Fish లేదా Prawn select చేయండి."
            }), 400

        file = request.files["image"]
        image_path, error = _save_uploaded_image(file, requested_species)
        if error:
            return jsonify(error), 400

        print(f"[API] Detection request | species={requested_species} | file={os.path.basename(image_path)}")
        result = _attach_result_guidance(
            analyze_species_disease(image_path, target_species=requested_species)
        )

        if not isinstance(result, dict):
            raise RuntimeError("Detector returned an invalid response object.")

        # Never let a detector implementation return the opposite species.
        if result.get("success") and str(result.get("species", "")).strip().lower() != requested_species:
            print(f"[API] Species mismatch blocked: requested={requested_species}, returned={result.get('species')}")
            return jsonify({
                "success": False,
                "is_valid_aquatic": False,
                "species": requested_species,
                "message": f"Species mismatch blocked. You selected {requested_species.capitalize()}.",
                "message_en": f"Species mismatch blocked. You selected {requested_species.capitalize()}.",
                "message_te": f"Species mismatch block అయింది. మీరు {('చేప' if requested_species == 'fish' else 'రొయ్య')} select చేశారు."
            }), 200

        return jsonify(result), 200

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "is_valid_aquatic": False,
            "message": "Detection backend failed while processing the image.",
            "message_en": "Detection backend failed while processing the image.",
            "message_te": "Image process చేస్తున్నప్పుడు detection backend error వచ్చింది.",
            "error": f"{type(exc).__name__}: {exc}"
        }), 500

@app.route("/predict/do", methods=["POST"])
def predict_do():
    payload = request.get_json(force=True, silent=True) or {}
    if not payload:
        return jsonify({"success": False, "message": "Missing water parameter payload."})
    return jsonify(predict_do_level(payload))

def _attach_result_guidance(result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every successful disease route returns the guidance used by the UI."""
    if not result.get("success"):
        return result

    species = str(result.get("species", "")).strip().lower()
    if species not in {"fish", "prawn"}:
        return result

    clinical_data = dict(result.get("clinical_data") or {})
    healthy = bool(clinical_data.get("is_healthy")) or result.get("predicted_class") in {
        "Healthy Fish",
        "Healthy",
    }
    clinical_data["is_healthy"] = healthy
    if healthy:
        clinical_data["selling_rates"] = MARKET_SELLING_RATES[species]
        clinical_data["market_rate_note_en"] = (
            "Indicative reference ranges only; actual prices vary by location, size, quality, "
            "season, and buyer. Confirm today's rate with local buyers."
        )
        clinical_data["market_rate_note_te"] = (
            "ఇవి సూచిక ధరల శ్రేణులు మాత్రమే. ప్రాంతం, సైజు, నాణ్యత, సీజన్‌ను బట్టి ధర మారుతుంది; "
            "ఈరోజు ధరను స్థానిక కొనుగోలుదారులతో నిర్ధారించండి."
        )
        clinical_data["market_rate_note_hi"] = (
            "ये केवल संकेतात्मक दरें हैं। स्थान, आकार, गुणवत्ता और मौसम से कीमत बदलती है; "
            "आज की दर स्थानीय खरीदार से पुष्टि करें।"
        )
    else:
        is_wssv = species == "prawn" and str(result.get("predicted_class", "")).lower() in {
            "wssv",
            "wssv_black_gill",
        }
        if is_wssv:
            precautions = {
                "en": (
                    "Treat this as a suspected viral-screening result, not confirmation. Separate "
                    "equipment for the affected pond, limit animal movement, do not release pond "
                    "water into canals, and contact a fisheries officer or diagnostic lab promptly. "
                    "Do not rely on antibiotics as a cure for a virus."
                ),
                "te": (
                    "ఇది వైరస్‌పై అనుమాన సూచన మాత్రమే; నిర్ధారిత diagnosis కాదు. ప్రభావిత చెరువు "
                    "పరికరాలను వేరుగా ఉంచండి, రొయ్యల తరలింపును తగ్గించండి, చెరువు నీటిని కాలువల్లో "
                    "విడవకండి; వెంటనే మత్స్య అధికారిని లేదా diagnostic lab‌ను సంప్రదించండి. వైరస్‌కు "
                    "antibiotics‌ను చికిత్సగా వాడకండి."
                ),
                "hi": (
                    "इसे केवल संभावित वायरल screening मानें, पुष्टि नहीं। प्रभावित तालाब के उपकरण "
                    "अलग रखें, पशुओं की आवाजाही सीमित करें, तालाब का पानी नहर में न छोड़ें और "
                    "मत्स्य अधिकारी या diagnostic lab से शीघ्र संपर्क करें। वायरस के इलाज के रूप में "
                    "antibiotics पर भरोसा न करें।"
                ),
                "tel": (
                    "Idi suspected viral screening matrame, confirm diagnosis kaadu. Affected pond "
                    "equipment separate ga unchandi, prawns movement thagginchandi, pond water canals "
                    "lo vadalakandi; fisheries officer/diagnostic lab ni contact cheyandi."
                ),
            }
        else:
            precautions = {
                "en": (
                    "Separate nets and equipment for affected stock, avoid moving animals between "
                    "ponds, remove dead animals safely, and check dissolved oxygen, temperature, pH, "
                    "ammonia and nitrite. Avoid overfeeding. Confirm the disease and any medicine or "
                    "dose with a qualified fisheries professional before treatment."
                ),
                "te": (
                    "ప్రభావిత చేపలు/రొయ్యల కోసం వలలు, పరికరాలు వేరుగా వాడండి; చెరువుల మధ్య తరలింపును "
                    "తగ్గించండి; చనిపోయిన వాటిని సురక్షితంగా తొలగించండి. DO, ఉష్ణోగ్రత, pH, ammonia, "
                    "nitrite తనిఖీ చేయండి; అధిక మేత వేయకండి. మందు లేదా మోతాదుకు ముందు మత్స్య నిపుణుడితో "
                    "వ్యాధిని నిర్ధారించండి."
                ),
                "hi": (
                    "प्रभावित stock के जाल और उपकरण अलग रखें, तालाबों के बीच जीवों को न ले जाएं, "
                    "मृत जीवों को सुरक्षित हटाएं और DO, तापमान, pH, ammonia तथा nitrite जांचें। "
                    "अधिक चारा न दें। उपचार से पहले रोग और दवा/खुराक की पुष्टि योग्य मत्स्य विशेषज्ञ से करें।"
                ),
                "tel": (
                    "Affected stock kosam nets/equipment separate ga vadandi, ponds madhya movement "
                    "avoid cheyandi, dead animals safe ga remove cheyandi. DO, temperature, pH, ammonia, "
                    "nitrite check cheyandi; overfeeding vaddu. Treatment mundu fisheries expert tho confirm cheyandi."
                ),
            }
        clinical_data["precautions_en"] = precautions["en"]
        clinical_data["precautions_te"] = precautions["te"]
        clinical_data["precautions_hi"] = precautions["hi"]
        clinical_data["precautions_tel"] = precautions["tel"]
        clinical_data.setdefault(
            "medicine_en",
            "No medicine can be safely selected from an image alone. Confirm the diagnosis and "
            "treatment with a qualified fisheries professional.",
        )
        clinical_data.setdefault(
            "food_guide_en",
            "Use species-appropriate quality feed, avoid overfeeding, and observe feed intake daily.",
        )

    result["clinical_data"] = clinical_data
    return result


@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    payload = request.get_json(force=True, silent=True) or {}
    message = str(payload.get("message") or payload.get("query") or "").strip()
    preferred = str(payload.get("lang", "te")).strip().lower()
    context = payload.get("context")

    if not message:
        return jsonify({
            "reply_language": preferred if preferred in {"en", "te", "hi"} else "te",
            "reply_english": "Please ask your farmer/aquaculture question.",
            "reply_telugu": "దయచేసి మీ రైతు / ఆక్వాకల్చర్ ప్రశ్నను అడగండి.",
            "reply_hindi": "कृपया अपना किसान या एक्वाकल्चर प्रश्न पूछें।",
            "voice_text": "దయచేసి మీ రైతు లేదా ఆక్వాకల్చర్ ప్రశ్నను అడగండి।"
        })

    reply = process_farmer_query(message, preferred, context)
    ai_reply = None
    try:
        ai_reply = generate_application_answer(message, context, reply["reply_language"])
    except (
        requests.RequestException,
        urllib.error.URLError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print(
            "[ASSISTANT] Gemini unavailable; returned built-in AquaCare answer "
            f"(error={type(exc).__name__}, http_status={status})."
        )
    if ai_reply:
        reply["reply_english"] = ai_reply
        reply["reply_telugu"] = ai_reply
        reply["reply_hindi"] = ai_reply
        reply["reply"] = ai_reply
        reply["voice_text"] = ai_reply
    return jsonify(reply)

@app.route("/api/market/orders", methods=["POST"])
def place_market_order():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Send order details as JSON."}), 400

    def required_text(field: str, limit: int) -> Optional[str]:
        value = payload.get(field)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if value and len(value) <= limit else None

    buyer_name = required_text("buyer_name", 80)
    phone_text = required_text("phone", 24)
    delivery_location = required_text("delivery_location", 160)
    if not buyer_name or not phone_text or not delivery_location:
        return jsonify({
            "success": False,
            "message": "Buyer name, valid phone number and delivery location are required.",
        }), 400

    phone = re.sub(r"\D", "", phone_text)
    if len(phone) == 12 and phone.startswith("91"):
        phone = phone[2:]
    if len(phone) != 10:
        return jsonify({"success": False, "message": "Enter a valid 10-digit Indian phone number."}), 400

    raw_quantity = payload.get("quantity")
    if isinstance(raw_quantity, bool):
        return jsonify({"success": False, "message": "Order quantity must be a positive number."}), 400
    try:
        quantity = float(raw_quantity)
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Order quantity must be a positive number."}), 400
    if not np.isfinite(quantity) or quantity <= 0 or quantity > 1_000_000:
        return jsonify({"success": False, "message": "Order quantity must be between 0 and 1,000,000."}), 400

    order_type = payload.get("order_type")
    if not isinstance(order_type, str) or order_type not in {"harvest", "feed", "medicine"}:
        return jsonify({"success": False, "message": "Choose a valid order category."}), 400

    seller_listing_id = None
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("BEGIN IMMEDIATE")
            if order_type == "harvest":
                raw_listing_id = payload.get("listing_id")
                if isinstance(raw_listing_id, bool):
                    return jsonify({"success": False, "message": "Choose an available fish or prawn listing."}), 400
                try:
                    seller_listing_id = int(raw_listing_id)
                except (TypeError, ValueError):
                    return jsonify({"success": False, "message": "Choose an available fish or prawn listing."}), 400
                listing = conn.execute("""
                    SELECT species, variety, quantity_kg, quantity_sold_kg,
                           quantity_reserved_kg, price_per_kg
                    FROM harvest_listings
                    WHERE id = ?
                """, (seller_listing_id,)).fetchone()
                if not listing:
                    return jsonify({"success": False, "message": "This harvest listing is no longer available."}), 404
                available_quantity = float(listing[2]) - float(listing[3]) - float(listing[4])
                if available_quantity <= 0:
                    return jsonify({"success": False, "message": "This harvest listing is sold out."}), 409
                if quantity > available_quantity:
                    return jsonify({"success": False, "message": "Order quantity exceeds the available harvest quantity."}), 400
                stock_update = conn.execute("""
                    UPDATE harvest_listings
                    SET quantity_reserved_kg = quantity_reserved_kg + ?
                    WHERE id = ? AND quantity_kg - quantity_sold_kg - quantity_reserved_kg >= ?
                """, (quantity, seller_listing_id, quantity))
                if stock_update.rowcount != 1:
                    return jsonify({"success": False, "message": "This harvest quantity is no longer available."}), 409
                species_name = "Prawn" if listing[0] == "prawn" else "Fish"
                product_name = f"{species_name}: {listing[1]}"
                unit_price = float(listing[5])
            else:
                product_id = payload.get("product_id")
                if not isinstance(product_id, str):
                    return jsonify({"success": False, "message": "Choose a product from the catalog."}), 400
                product = next((
                    item for item in BUYING_CATALOG
                    if item.get("id") == product_id and item.get("category") == order_type
                ), None)
                if not product:
                    return jsonify({"success": False, "message": "Choose a product from the selected catalog."}), 400
                language = payload.get("lang")
                if not isinstance(language, str) or language not in {"en", "te", "hi"}:
                    language = "en"
                product_name = product.get(f"title_{language}") or product["title_en"]
                unit_price = float(product["unit_price"])

            estimated_total = round(unit_price * quantity, 2)
            cursor = conn.execute("""
                INSERT INTO marketplace_orders
                    (buyer_name, phone, delivery_location, order_type, product_name,
                     seller_listing_id, quantity, unit_price, estimated_total, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Placed')
            """, (
                buyer_name, phone, delivery_location, order_type, product_name,
                seller_listing_id, quantity, unit_price, estimated_total,
            ))
            order_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        return jsonify({
            "success": True,
            "order_id": order_id,
            "product_name": product_name,
            "quantity": quantity,
            "estimated_total": estimated_total,
            "status": "Placed",
            "contact_phone": MARKETPLACE_CONTACT_PHONE,
        }), 201
    except sqlite3.Error as exc:
        print(f"[MARKETPLACE] Could not place order: {exc}")
        return jsonify({"success": False, "message": "Could not place order. Please try again."}), 500


@app.route("/api/farmer/dashboard", methods=["GET"])
@farmer_login_required
def get_farmer_dashboard():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            listings = conn.execute("""
                SELECT id, created_at, seller_name, phone, location, species,
                       variety, quantity_kg, quantity_sold_kg, quantity_reserved_kg,
                       quantity_kg - quantity_sold_kg - quantity_reserved_kg AS available_quantity_kg,
                       price_per_kg, notes
                FROM harvest_listings
                ORDER BY id DESC
            """).fetchall()
            orders = conn.execute("""
                SELECT o.id, o.created_at, o.buyer_name, o.phone, o.delivery_location,
                       o.order_type, o.product_name, o.seller_listing_id, o.quantity,
                       o.unit_price, o.estimated_total, o.status,
                       l.seller_name AS seller_name, l.phone AS seller_phone,
                       l.species AS species
                FROM marketplace_orders AS o
                LEFT JOIN harvest_listings AS l ON l.id = o.seller_listing_id
                ORDER BY o.id DESC
                LIMIT 200
            """).fetchall()
        finally:
            conn.close()
        return jsonify({
            "success": True,
            "username": session.get("farmer_username", ""),
            "listings": [dict(row) for row in listings],
            "orders": [dict(row) for row in orders],
        })
    except sqlite3.Error as exc:
        print(f"[FARMER DASHBOARD] Could not load dashboard data: {exc}")
        return jsonify({"success": False, "message": "Could not load farmer dashboard."}), 500


@app.route("/api/market/orders/<int:order_id>/status", methods=["PATCH"])
@farmer_login_required
def update_market_order_status(order_id: int):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("status") != "Accepted":
        return jsonify({"success": False, "message": "Only a Placed order can be accepted."}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute("""
                SELECT order_type, seller_listing_id, quantity, status
                FROM marketplace_orders
                WHERE id = ?
            """, (order_id,))
            order = cursor.fetchone()
            if not order:
                return jsonify({"success": False, "message": "Order not found."}), 404
            if order[3] != "Placed":
                return jsonify({"success": False, "message": "This order is already accepted."}), 409
            if order[0] == "harvest" and order[1] is not None:
                stock_update = conn.execute("""
                    UPDATE harvest_listings
                    SET quantity_reserved_kg = quantity_reserved_kg - ?,
                        quantity_sold_kg = quantity_sold_kg + ?
                    WHERE id = ? AND quantity_reserved_kg >= ?
                """, (order[2], order[2], order[1], order[2]))
                if stock_update.rowcount != 1:
                    return jsonify({"success": False, "message": "Reserved harvest stock is unavailable; refresh the dashboard."}), 409
            status_update = conn.execute("""
                UPDATE marketplace_orders
                SET status = 'Accepted'
                WHERE id = ? AND status = 'Placed'
            """, (order_id,))
            if status_update.rowcount != 1:
                return jsonify({"success": False, "message": "This order is already accepted."}), 409
            conn.commit()
        finally:
            conn.close()
        return jsonify({"success": True, "order_id": order_id, "status": "Accepted"})
    except sqlite3.Error as exc:
        print(f"[FARMER DASHBOARD] Could not update order {order_id}: {exc}")
        return jsonify({"success": False, "message": "Could not update order status."}), 500


@app.route("/api/catalog", methods=["GET"])
def get_market_catalog():
    category = request.args.get("category", "").strip().lower()
    language = request.args.get("lang", "en").strip().lower()
    if category not in {"feed", "medicine"}:
        return jsonify({"success": False, "message": "Choose feed or medicine catalog."}), 400
    if language not in {"en", "te", "hi"}:
        language = "en"

    products = []
    for item in BUYING_CATALOG:
        if item.get("category") != category:
            continue
        search_query = f"{item['title_en']} aquaculture"
        encoded_query = quote_plus(search_query)
        products.append({
            "id": item["id"],
            "title": item.get(f"title_{language}") or item["title_en"],
            "price": item["price"],
            "purpose": item.get(f"purpose_{language}") or item["purpose_en"],
            "website_url": f"https://dir.indiamart.com/search.mp?ss={encoded_query}",
            "website_links": [
                {
                    "label": "IndiaMART",
                    "url": f"https://dir.indiamart.com/search.mp?ss={encoded_query}",
                },
                {
                    "label": "TradeIndia",
                    "url": f"https://www.tradeindia.com/search.html?keyword={encoded_query}",
                },
                {
                    "label": "Google Search",
                    "url": f"https://www.google.com/search?q={encoded_query}",
                },
            ],
        })
    return jsonify({
        "success": True,
        "category": category,
        "contact_phone": MARKETPLACE_CONTACT_PHONE,
        "products": products,
    })


@app.route("/api/market/listings", methods=["GET", "POST"])
def harvest_market_listings():
    if request.method == "GET":
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute("""
                    SELECT id, created_at, seller_name, phone, location, species,
                           variety, quantity_kg - quantity_sold_kg - quantity_reserved_kg AS quantity_kg,
                           price_per_kg, notes
                    FROM harvest_listings
                    WHERE quantity_kg - quantity_sold_kg - quantity_reserved_kg > 0
                    ORDER BY id DESC
                    LIMIT 100
                """).fetchall()
            finally:
                conn.close()
            return jsonify({
                "success": True,
                "contact_phone": MARKETPLACE_CONTACT_PHONE,
                "listings": [dict(row) for row in rows],
            })
        except sqlite3.Error as exc:
            print(f"[MARKETPLACE] Could not load harvest listings: {exc}")
            return jsonify({"success": False, "message": "Could not load listings."}), 500

    if not session.get("farmer_authenticated"):
        return jsonify({"success": False, "message": "Farmer login is required to post harvest."}), 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "message": "Send listing details as JSON."}), 400

    text_fields = {
        "seller_name": (80, "Seller name"),
        "location": (120, "Location"),
        "variety": (80, "Fish/prawn variety"),
        "notes": (500, "Notes"),
        "phone": (24, "Phone number"),
    }
    values = {}
    for field, (max_length, label) in text_fields.items():
        value = payload.get(field, "")
        if not isinstance(value, str):
            return jsonify({"success": False, "message": f"{label} must be text."}), 400
        value = value.strip()
        if field != "notes" and not value:
            return jsonify({"success": False, "message": f"{label} is required."}), 400
        if len(value) > max_length:
            return jsonify({"success": False, "message": f"{label} is too long."}), 400
        values[field] = value

    species = payload.get("species")
    if not isinstance(species, str) or species not in {"fish", "prawn"}:
        return jsonify({"success": False, "message": "Species must be fish or prawn."}), 400

    phone_digits = re.sub(r"\D", "", values["phone"])
    if len(phone_digits) == 12 and phone_digits.startswith("91"):
        phone_digits = phone_digits[2:]
    if len(phone_digits) != 10:
        return jsonify({"success": False, "message": "Enter a valid 10-digit Indian phone number."}), 400
    values["phone"] = phone_digits

    amounts = {}
    for field, label in (("quantity_kg", "Quantity"), ("price_per_kg", "Price per kg")):
        raw_value = payload.get(field)
        if isinstance(raw_value, bool):
            return jsonify({"success": False, "message": f"{label} must be a positive number."}), 400
        try:
            amount = float(raw_value)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": f"{label} must be a positive number."}), 400
        if not np.isfinite(amount) or amount <= 0 or amount > 1_000_000:
            return jsonify({"success": False, "message": f"{label} must be between 0 and 1,000,000."}), 400
        amounts[field] = amount

    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.execute("""
                INSERT INTO harvest_listings
                    (seller_name, phone, location, species, variety, quantity_kg, price_per_kg, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                values["seller_name"], values["phone"], values["location"], species,
                values["variety"], amounts["quantity_kg"], amounts["price_per_kg"],
                values["notes"],
            ))
            listing_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        return jsonify({"success": True, "listing_id": listing_id, "message": "Listing posted."}), 201
    except sqlite3.Error as exc:
        print(f"[MARKETPLACE] Could not save harvest listing: {exc}")
        return jsonify({"success": False, "message": "Could not post listing."}), 500


@app.route("/api/order", methods=["POST"])
def submit_order_inquiry():
    payload = request.get_json(force=True, silent=True) or {}
    product_name = payload.get("product_name", "")
    farmer_name = payload.get("farmer_name", "")
    phone = payload.get("phone", "")
    location = payload.get("location", "")
    quantity = payload.get("quantity", "")

    if not phone or not product_name:
        return jsonify({"success": False, "message": "Phone number and product name are required."})

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO purchase_inquiries (farmer_name, phone, location, product_name, quantity)
            VALUES (?, ?, ?, ?, ?)
        """, (farmer_name, phone, location, product_name, quantity))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": "Inquiry recorded successfully."})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

SAMPLE_IMAGES = {
    "fish_healthy": ("fish", "fish_yolo", "test", "images", "healthy_fish_koki_129_jpg.rf.3bba32005b9978177054a0bf80c6cf51.jpg"),
    "fish_bacterial": ("fish", "fish_yolo", "test", "images", "bacterial_diseases_koi_002_jpg.rf.8dd1af9266169f4a8392cc328ca2e1ef.jpg"),
    "prawn_wssv": ("prawn", "prawn_yolo", "test", "images", "WSSV_BG-5-img-2_jpg.rf.3e15116a6064907451cb08ecd8db9c6a.jpg"),
    "prawn_blackgill": ("prawn", "prawn_yolo", "test", "images", "BG-1-img-1_jpg.rf.1211405422581d61308e7d5d95de844f.jpg"),
}

@app.route("/api/sample/<sample_key>")
def get_sample_image(sample_key: str):
    if sample_key in SAMPLE_IMAGES:
        rel_parts = SAMPLE_IMAGES[sample_key]
        directory = os.path.join(BASE_DIR, *rel_parts[:-1])
        filename = rel_parts[-1]
        return send_from_directory(directory, filename)
    return jsonify({"error": "Sample not found"}), 404

# ==============================================================================
# 9. RUN APPLICATION
# ==============================================================================
from assistant_blueprint import (
    bp as assistant_bp,
    generate_application_answer,
)
app.register_blueprint(assistant_bp)

def run_local_server():
    selected_port = None
    for candidate_port in range(5001, 5100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate_port))
            except OSError:
                url = f"http://127.0.0.1:{candidate_port}"
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                        health = json.load(response)
                except (OSError, ValueError, urllib.error.URLError):
                    print(
                        f"[AQUACARE] Port {candidate_port} is occupied by another service; "
                        "trying the next port."
                    )
                    continue

                if (
                    isinstance(health, dict)
                    and health.get("service") == "AquaCare AI Platform v4.7 contextual-assistant"
                    and isinstance(health.get("models_loaded"), dict)
                    and health["models_loaded"].get("species_gate_model") is True
                ):
                    print(f"[AQUACARE] Verified server is already running at {url}")
                    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
                    return

                print(
                    f"[AQUACARE] Port {candidate_port} has an older/incompatible server; "
                    "trying the next port."
                )
                continue
            else:
                selected_port = candidate_port
                break

    if selected_port is None:
        raise RuntimeError(
            "No free AquaCare port is available between 5001 and 5099. "
            "Close an unused local server and run `python app.py` again."
        )

    url = f"http://127.0.0.1:{selected_port}"

    print(f"[AQUACARE] Starting verified Fish/Prawn server at {url}")
    print(f"[AQUACARE] Species gate loaded: {SPECIES_GATE_MODEL is not None}")
    threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    app.run(
        debug=False,
        use_reloader=False,
        host="127.0.0.1",
        port=selected_port,
        threaded=True,
    )


if __name__ == "__main__":
    run_local_server()

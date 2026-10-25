"""Secure AquaCare application assistant. The API key lives only in .env."""
import json
import os
import requests
from flask import Blueprint, jsonify, request

bp = Blueprint("assistant", __name__)

SYSTEM = (
    "You are the help assistant for the AquaCare Fish & Prawn application. Answer only questions "
    "about using AquaCare, its Fish/Prawn disease image screening, bounding boxes, healthy-result "
    "market-rate guides, DO estimates, supported languages, result screens, and the guidance shown "
    "inside the app. Use the supplied application facts and current result JSON as your only source "
    "of product-specific facts. Do not invent features, live prices, diagnoses, or data not present "
    "in that source. If a detail is not documented, say so and explain where in the app to check. "
    "For unrelated questions, politely say you can only help with AquaCare and invite an "
    "AquaCare-related question. Never treat an image-screening result as a confirmed diagnosis. "
    "When discussing medicines or chemical guidance displayed by the app, tell the user to confirm "
    "with a qualified fisheries professional before use. Answer in the language requested."
)


APPLICATION_FACTS = (
    "AquaCare is a local web application for Fish and Prawn disease-image screening and bounding "
    "boxes, healthy-stock guidance and indicative selling-rate ranges, dissolved-oxygen (DO) "
    "estimates from entered pond parameters, and a farmer assistant. Fish and Prawn have separate "
    "tabs; an image that does not match the selected species is rejected. The UI supports English, "
    "Telugu, and Hindi. Fish screening classes are Healthy Fish, Bacterial Diseases, Parasitic "
    "Diseases, White Tail Diseases, and Fungal Diseases. Prawn classes are Healthy, WSSV, "
    "Black Gill, and WSSV + Black Gill. A disease result can show localized image regions, severity, "
    "precautions, medicine information, and food/feed guidance. Healthy results show species-specific "
    "indicative selling-rate ranges. DO is estimated from entered pond inputs; it is not a live meter "
    "reading. Disease advice and prices are screening/reference guidance, not a confirmed diagnosis "
    "or guaranteed live market quote. Disease medicines/doses must be verified with a qualified "
    "fisheries professional. The launcher selects the first free local port from 5001-5099; "
    "the terminal and opened browser show the current URL, and app URL + /health reports status."
)


def assistant_error_hint(exc, language="en"):
    """Explain common Gemini failures without exposing the configured key."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    language = str(language or "en").lower()

    if status == 400:
        messages = {
            "en": (
                "Gemini rejected the request (HTTP 400). Check that GEMINI_API_KEY is a valid "
                "Google AI Studio Gemini API key and GEMINI_MODEL is gemini-2.5-flash. If the key "
                "is not a Google Gemini key, create a Gemini API key and replace it in .env."
            ),
            "te": (
                "Gemini request‌ను తిరస్కరించింది (HTTP 400). GEMINI_API_KEY Google AI Studio Gemini "
                "API keyనా, GEMINI_MODEL gemini-2.5-flashనా తనిఖీ చేయండి. Key Geminiది కాకపోతే "
                "కొత్త Gemini API key సృష్టించి .env‌లో మార్చండి."
            ),
            "hi": (
                "Gemini ने request अस्वीकार की (HTTP 400)। जांचें कि GEMINI_API_KEY Google AI Studio "
                "की Gemini API key है और GEMINI_MODEL gemini-2.5-flash है। यदि key Gemini की नहीं है, "
                "तो नई Gemini API key बनाकर .env में बदलें।"
            ),
        }
    elif status in {401, 403}:
        messages = {
            "en": "Gemini authorization failed. Create or enable a Gemini API key in Google AI Studio and replace GEMINI_API_KEY in .env.",
            "te": "Gemini అనుమతి విఫలమైంది. Google AI Studioలో Gemini API key సృష్టించండి/enable చేసి .env‌లో GEMINI_API_KEY మార్చండి.",
            "hi": "Gemini authorization विफल हुआ। Google AI Studio में Gemini API key बनाएं/enable करें और .env में GEMINI_API_KEY बदलें।",
        }
    elif status == 404:
        messages = {
            "en": "Gemini could not find the configured model. Set GEMINI_MODEL=gemini-2.5-flash in .env.",
            "te": "Gemini model కనిపించలేదు. .env‌లో GEMINI_MODEL=gemini-2.5-flash సెట్ చేయండి.",
            "hi": "Gemini को configured model नहीं मिला। .env में GEMINI_MODEL=gemini-2.5-flash सेट करें।",
        }
    elif status == 429:
        messages = {
            "en": "Gemini rate limit or quota was reached. Check the Google AI Studio project quota and try again later.",
            "te": "Gemini request పరిమితి/quota చేరింది. Google AI Studio quota తనిఖీ చేసి కొద్దిసేపటి తర్వాత ప్రయత్నించండి.",
            "hi": "Gemini की request सीमा/quota पूरी हो गई। Google AI Studio quota जांचें और बाद में प्रयास करें।",
        }
    else:
        messages = {
            "en": f"Gemini is unavailable{f' (HTTP {status})' if status else ''}. Check network access and the key/model settings in .env.",
            "te": f"Gemini అందుబాటులో లేదు{f' (HTTP {status})' if status else ''}. Network మరియు .env‌లో key/model settings తనిఖీ చేయండి.",
            "hi": f"Gemini उपलब्ध नहीं है{f' (HTTP {status})' if status else ''}। Network और .env में key/model settings जांचें।",
        }
    return messages.get(language, messages["en"])


def generate_application_answer(question, context=None, language="en"):
    """Return a Gemini answer, or None when no API key is configured."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key.startswith("AIza"):
        return None

    prompt_context = json.dumps(context or {}, ensure_ascii=False)[:6000]
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{
            "role": "user",
            "parts": [{
                "text": (
                    f"APPLICATION FACTS:\n{APPLICATION_FACTS}\n\n"
                    f"CURRENT RESULT CONTEXT JSON:\n{prompt_context}\n\n"
                    f"REPLY LANGUAGE: {language}\nQUESTION: {str(question)[:1000]}"
                )
            }],
        }],
    }
    response = requests.post(
        url,
        headers={"x-goog-api-key": key},
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["candidates"][0]["content"]["parts"][0]["text"].strip()


@bp.route("/api/assistant", methods=["POST"])
def assistant():
    d = request.get_json(silent=True) or {}
    q = str(d.get("question", "")).strip()[:1000]
    if not q:
        return jsonify(success=False, message="Empty question."), 400

    try:
        text = generate_application_answer(
            q,
            d.get("context") or {},
            str(d.get("language", "en"))[:10],
        )
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        return jsonify(
            success=False,
            message=assistant_error_hint(exc, str(d.get("language", "en"))),
        ), 502

    if text is None:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        message = (
            "GEMINI_API_KEY is configured but does not look like a Google AI Studio key. "
            "Use a Gemini API key from Google AI Studio."
            if key
            else "GEMINI_API_KEY is not set in .env."
        )
        return jsonify(success=False, message=message), 503
    return jsonify(success=True, reply=text)

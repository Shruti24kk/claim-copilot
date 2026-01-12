from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any
import joblib

from PIL import Image
import numpy as np
import imagehash
import shutil

app = FastAPI(title="Claim Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS = Path("app/ml/artifacts")
UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)

tfidf = joblib.load(ARTIFACTS / "tfidf.joblib")
model = joblib.load(ARTIFACTS / "text_model.joblib")

# In-memory store of image hashes (demo). Resets when server restarts.
KNOWN_HASHES: Dict[str, str] = {}

@app.get("/health")
def health():
    return {"status": "ok"}

def save_image(upload: UploadFile) -> Path:
    ext = Path(upload.filename).suffix.lower() or ".jpg"
    out = UPLOADS / f"{uuid4().hex}{ext}"
    with out.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out

def image_signals(path: Path) -> Dict[str, Any]:
    """
    Starter 'damage' signals (not a true dent detector).
    These are useful for triage + quality checks + fraud signals.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0  # 0..1

    # Brightness
    brightness = float(arr.mean())

    # Simple blur score: variance of Laplacian-ish using finite differences
    # (proxy: if very low => blurry)
    gray = arr.mean(axis=2)
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_strength = float((gx.mean() + gy.mean()) / 2.0)

    # "Complexity" proxy: higher edges => more texture (sometimes damage/parts complexity)
    complexity = edge_strength

    # Perceptual hash for duplicate detection
    ph = str(imagehash.phash(img))

    return {
        "width": img.size[0],
        "height": img.size[1],
        "brightness": round(brightness, 4),
        "edge_strength": round(edge_strength, 4),
        "complexity": round(complexity, 4),
        "phash": ph,
        "quality_flag_blurry": edge_strength < 0.010,  # tweakable threshold
        "quality_flag_too_dark": brightness < 0.12,
        "quality_flag_too_bright": brightness > 0.88,
    }

def image_risk_from_signals(img_infos: List[Dict[str, Any]]) -> (float, List[str], List[Dict[str, Any]]):
    reasons = []
    matches = []

    # Duplicate within the same claim
    hashes = [x["phash"] for x in img_infos if "phash" in x]
    if len(hashes) >= 2 and len(set(hashes)) < len(hashes):
        reasons.append("Duplicate image(s) detected in this upload (possible reuse).")

    # Similar to previously uploaded images (demo)
    for info in img_infos:
        h = info.get("phash")
        if not h:
            continue
        for prev_id, prev_h in KNOWN_HASHES.items():
            # Compare hashes via Hamming distance
            d = (int(h, 16) ^ int(prev_h, 16)).bit_count()
            if d <= 6:
                matches.append({"previous_image_id": prev_id, "hamming_distance": d})
                reasons.append("Image similar to a previously uploaded claim image (possible reuse).")
                break

    # Quality reasons
    for info in img_infos:
        if info.get("quality_flag_blurry"):
            reasons.append("At least one image looks blurry/low detail; request clearer photos.")
            break
    for info in img_infos:
        if info.get("quality_flag_too_dark") or info.get("quality_flag_too_bright"):
            reasons.append("At least one image is poorly exposed (too dark/bright); request retake.")
            break

    # Convert signals to a simple score 0..1 (starter)
    score = 0.10
    if any("similar to a previously" in r.lower() for r in reasons):
        score += 0.35
    if any("duplicate" in r.lower() for r in reasons):
        score += 0.20
    if any("blurry" in r.lower() for r in reasons):
        score += 0.10
    if any("exposed" in r.lower() for r in reasons):
        score += 0.10

    score = max(0.0, min(1.0, score))
    return score, reasons, matches

@app.post("/analyze")
async def analyze(claim_text: str = Form(...), images: List[UploadFile] = File([])):
    claim_id = uuid4().hex[:8]

    # --- TEXT MODEL ---
    X = tfidf.transform([claim_text])
    text_prob = float(model.predict_proba(X)[0, 1])  # 0..1
    text_score = int(round(text_prob * 100))

    # --- IMAGE SIGNALS ---
    img_infos = []
    saved_paths = []
    for img in images:
        try:
            p = save_image(img)
            saved_paths.append(p)
            info = image_signals(p)
            img_infos.append(info)
        except Exception:
            img_infos.append({"error": f"Could not process image: {img.filename}"})

    img_prob = 0.0
    img_reasons = []
    img_matches = []

    if img_infos:
        img_prob, img_reasons, img_matches = image_risk_from_signals(img_infos)

        # Store hashes for future reuse detection (demo)
        for i, info in enumerate(img_infos):
            if "phash" in info:
                KNOWN_HASHES[f"{claim_id}-{i}"] = info["phash"]

    # --- FUSION ---
    final_prob = min(1.0, 0.65 * text_prob + 0.35 * img_prob)
    final_score = int(round(final_prob * 100))

    if final_score >= 75:
        triage = "Fraud Investigation"
    elif final_score >= 45:
        triage = "Adjuster Review"
    else:
        triage = "Straight Through Processing"

    reasons = []
    if text_score >= 70:
        reasons.append("Text model indicates elevated risk based on suspicious narrative patterns.")
    reasons.extend(img_reasons[:5])

    questions = [
        "Upload 2–3 wide-angle photos (front/side/rear).",
        "Provide date/time and exact location of incident.",
        "Provide repair estimate or shop quote if available.",
    ]
    if final_score >= 45:
        questions.append("Provide police report number or incident reference (if available).")

    return {
        "claim_id": claim_id,
        "risk_score": final_score,
        "triage": triage,
        "text_model_score": text_score,
        "image_score_component": int(round(img_prob * 100)),
        "reasons": reasons if reasons else ["No strong risk signals found."],
        "questions_to_ask": questions,
        "image_analysis": img_infos,
        "image_matches": img_matches,
    }

"""
FastAPI Microservice for ResQFlow ML Flood Severity Assessment.
Drop-in replacement for flood-severity-analyzer in Capstone-Sprint-1.
Directly interfaces with Spring Boot VictimService via /api/v1/score-base64.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import io
import base64
import tempfile
from typing import List, Optional
import torch
torch.set_num_threads(1)
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import FloodDamagePipeline

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = FloodDamagePipeline()
    return pipeline

app = FastAPI(
    title="ResQFlow ML Flood Severity Analyzer",
    description="Custom PyTorch EfficientNet-B0 + Gated Damage Model API for ResQFlow Disaster Management",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Java / Spring Boot Matching DTOs (Strict 0 for non-flood, 0-100 for flood)
# ---------------------------------------------------------------------------
class FloodScoreRequestDTO(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded image string (with or without 'data:image/...;base64,' prefix)",
    )


class FloodScoreResponseDTO(BaseModel):
    damage_score: int = Field(0, description="Damage severity score 0-100 (strict 0 for non-flooded)")
    severity_score: int = Field(0, description="Damage severity score 0-100 (strict 0 for non-flooded)")
    severity_level: str = Field(..., description="NOT_FLOOD, VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH, CRITICAL")
    rescue_priority: str = Field(..., description="NONE, LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., description="Confidence between 0.0 and 1.0")
    title: str = Field(..., description="Incident headline")
    description: str = Field(..., description="Assessment details")
    what_is_in_image: str = Field(..., description="Identified visual elements")
    estimated_depth: str = Field(..., description="Depth range e.g. '0 cm (Dry Surface)'")
    road_access: str = Field(..., description="CLEAR, PASSABLE_CAUTION, IMPASSABLE, COMPLETELY_BLOCKED")
    recommendations: List[str] = Field(..., description="Safety and triage recommendations")
    flood_detected: bool = Field(False, description="Binary flood flag from Stage 1 gate")
    flood_probability: float = Field(0.0, description="Raw model probability (0.0 for non-flooded)")
    damage_score_100: int = Field(0, description="Damage severity score 0-100 (strict 0 for non-flooded)")

    class Config:
        populate_by_name = True


def format_model_result(result: dict) -> dict:
    flood_detected = bool(result.get("flood_detected", False))
    prob = float(result.get("flood_probability", 0.0))
    raw_score = int(result.get("damage_score", 0))
    reason = result.get("reason", "")
    
    if not flood_detected or raw_score == 0:
        is_doc = "document" in reason.lower() or "text" in reason.lower()
        title = "Non-Hazard: Document / Text Detected" if is_doc else "Normal Condition: No Flood Inundation Detected"
        what_is_in_image = "Document / printed paper; no flood inundation detected." if is_doc else "Dry surface; no standing water, road submersion, or structural flooding."
        
        return {
            "damage_score": 0,
            "severity_score": 0,
            "severity_level": "NOT_FLOOD",
            "rescue_priority": "NONE",
            "confidence": float(round(1.0 - prob if prob < 0.5 else 0.99, 4)),
            "title": title,
            "description": reason if reason else "Safe: No flood inundation detected.",
            "what_is_in_image": what_is_in_image,
            "estimated_depth": "0 cm (Dry Surface)",
            "road_access": "CLEAR",
            "recommendations": [
                "Standard civilian transit permitted.",
                "No rescue mobilization required.",
                "Routine monitoring continues."
            ],
            "flood_detected": False,
            "flood_probability": 0.0,
            "damage_score_100": 0
        }
    else:
        score = max(1, min(100, raw_score))
        if score <= 20:
            level = "VERY_LOW"
            priority = "LOW"
            depth = "5 - 15 cm (Curbside Puddles)"
            access = "CLEAR"
            what = "Shallow water accumulation, surface runoff, or curbside puddles."
            recs = [
                "Roads are safe for transit. Drive with standard caution.",
                "Monitor local rainfall alerts.",
                "Ensure local street drains are unblocked."
            ]
        elif score <= 40:
            level = "LOW"
            priority = "LOW"
            depth = "15 - 30 cm (Ankle Deep)"
            access = "PASSABLE_CAUTION"
            what = "Shallow road flooding; water pooling along streets and curbs."
            recs = [
                "Proceed with caution on low-lying roads.",
                "Avoid parking near storm drains.",
                "Small passenger vehicles should slow down."
            ]
        elif score <= 60:
            level = "MODERATE"
            priority = "MEDIUM"
            depth = "30 - 65 cm (Knee to Wheel Height)"
            access = "PASSABLE_CAUTION"
            what = "Submerged curbs, water reaching vehicle tires, and water accumulating around residential boundaries."
            recs = [
                "Restrict light vehicle transit through flooded sections.",
                "Elevate electrical appliances and critical documents.",
                "Prepare household emergency evacuation kits."
            ]
        elif score <= 80:
            level = "HIGH"
            priority = "HIGH"
            depth = "70 - 130 cm (Waist to Lower-Door Height)"
            access = "IMPASSABLE"
            what = "Significant street inundation, partially submerged vehicles, ground floor water ingress."
            recs = [
                "Deploy high-clearance rescue vehicles and dinghies.",
                "Evacuate ground-floor occupants to higher stories or relief shelters.",
                "Disconnect local electricity grid sections in affected streets."
            ]
        elif score <= 95:
            level = "VERY_HIGH"
            priority = "CRITICAL"
            depth = "130 - 180 cm (Chest Deep Water)"
            access = "IMPASSABLE"
            what = "Severe street and residential inundation; vehicles floating or deeply submerged."
            recs = [
                "Deploy emergency rescue boats and dinghies immediately.",
                "Evacuate vulnerable individuals and elderly immediately.",
                "Secure emergency shelter locations."
            ]
        else:
            level = "CRITICAL"
            priority = "CRITICAL"
            depth = "> 180 cm (Chest Deep to Roof Inundation)"
            access = "COMPLETELY_BLOCKED"
            what = "Severe catastrophic flooding, deep fast-flowing water, structural submersion."
            recs = [
                "Immediate tactical rescue deployment (boats, helicopters, emergency divers).",
                "Aerial distribution of clean drinking water, rations, and first aid.",
                "Establish medical triage stations at nearest elevated staging areas."
            ]
            
        return {
            "damage_score": score,
            "severity_score": score,
            "severity_level": level,
            "rescue_priority": priority,
            "confidence": float(round(prob, 4)),
            "title": f"Flood Warning: {level} Severity Inundation (Score {score}/100)",
            "description": reason if reason else f"Inundation detected with damage score {score}/100.",
            "what_is_in_image": what,
            "estimated_depth": depth,
            "road_access": access,
            "recommendations": recs,
            "flood_detected": True,
            "flood_probability": prob,
            "damage_score_100": score
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.get("/", response_class=HTMLResponse)
def index_page():
    if os.path.exists(HTML_PATH):
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ResQFlow Flood Severity Assessment API</h1>")


@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "UP",
        "service": "ResQFlow-ML-Vision",
        "model": "EfficientNet-B0 + Stage 1/2 Gated Neural Pipeline",
        "deployment": "Custom ML PyTorch API"
    }


@app.post("/api/v1/score-base64", response_model=FloodScoreResponseDTO)
@app.post("/score-base64", response_model=FloodScoreResponseDTO)
@app.post("/api/v1/flood/score", response_model=FloodScoreResponseDTO)
@app.post("/api/v1/score", response_model=FloodScoreResponseDTO)
@app.post("/score", response_model=FloodScoreResponseDTO)
def score_flood_image(request: FloodScoreRequestDTO):
    raw_b64 = request.image_base64
    if not raw_b64 or not raw_b64.strip():
        fallback = {
            "damage_score": 0,
            "severity_score": 0,
            "severity_level": "NOT_FLOOD",
            "rescue_priority": "NONE",
            "confidence": 0.0,
            "title": "Empty Image Payload",
            "description": "No valid image data was attached to the assessment request.",
            "what_is_in_image": "No image content detected.",
            "estimated_depth": "0 cm (Dry Surface)",
            "road_access": "CLEAR",
            "recommendations": ["Ensure photo is attached when filing reports."],
            "flood_detected": False,
            "flood_probability": 0.0,
            "damage_score_100": 0
        }
        return FloodScoreResponseDTO(**fallback)

    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
        # Open with PIL and resize large images to save RAM on free-tier containers
        with Image.open(io.BytesIO(img_bytes)) as pil_img:
            pil_img = pil_img.convert("RGB")
            if pil_img.width > 1024 or pil_img.height > 1024:
                pil_img.thumbnail((1024, 1024))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                pil_img.save(tmp.name, format="JPEG", quality=85)
                tmp_path = tmp.name

        result = get_pipeline().predict(tmp_path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        dto = format_model_result(result)
        return FloodScoreResponseDTO(**dto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

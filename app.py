"""
FastAPI Microservice for ResQFlow ML Flood Severity Assessment.
Drop-in replacement for flood-severity-analyzer in Capstone-Sprint-1.
Directly interfaces with Spring Boot VictimService via /api/v1/score-base64.
"""

import os
import io
import base64
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from predict import FloodDamagePipeline

# Load ML Pipeline on startup
pipeline = FloodDamagePipeline()

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
# Java / Spring Boot Matching DTOs
# ---------------------------------------------------------------------------
class FloodScoreRequestDTO(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded image string (with or without 'data:image/...;base64,' prefix)",
    )


class FloodScoreResponseDTO(BaseModel):
    severity_score: int = Field(..., description="Integer severity score out of 10 (1-10)")
    severity_level: str = Field(..., description="LOW, MODERATE, HIGH, EXTREME")
    rescue_priority: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., description="Confidence between 0.0 and 1.0")
    title: str = Field(..., description="Incident headline")
    description: str = Field(..., description="Assessment details")
    what_is_in_image: str = Field(..., description="Identified visual elements")
    estimated_depth: str = Field(..., description="Depth range e.g. '0 cm (Dry Surface)'")
    road_access: str = Field(..., description="CLEAR, PASSABLE_CAUTION, IMPASSABLE, COMPLETELY_BLOCKED")
    recommendations: List[str] = Field(..., description="Safety and triage recommendations")
    flood_detected: Optional[bool] = Field(None, description="Binary flood flag from Stage 1 gate")
    flood_probability: Optional[float] = Field(None, description="Raw model probability")
    damage_score_100: Optional[int] = Field(None, description="Damage severity score 0-100")

    class Config:
        populate_by_name = True


def format_model_result(result: dict) -> dict:
    flood_detected = result.get("flood_detected", False)
    prob = float(result.get("flood_probability", 0.0))
    raw_score = int(result.get("damage_score", 0))
    reason = result.get("reason", "")
    
    if not flood_detected:
        is_doc = "document" in reason.lower() or "text" in reason.lower()
        title = "Non-Hazard: Document / Text Detected" if is_doc else "Normal Condition: No Flood Inundation Detected"
        what_is_in_image = "Document / printed paper; no flood inundation detected." if is_doc else "Dry surface; no standing water, road submersion, or structural flooding."
        
        return {
            "severity_score": 1,
            "severity_level": "LOW",
            "rescue_priority": "LOW",
            "confidence": float(round(1.0 - prob if prob < 0.5 else 0.99, 4)),
            "title": title,
            "description": reason,
            "what_is_in_image": what_is_in_image,
            "estimated_depth": "0 cm (Dry Surface)",
            "road_access": "CLEAR",
            "recommendations": [
                "Standard civilian transit permitted.",
                "No rescue mobilization required.",
                "Routine monitoring continues."
            ],
            "flood_detected": False,
            "flood_probability": prob,
            "damage_score_100": 0
        }
    else:
        # Scale 1-100 to 1-10
        severity_10 = max(2, min(10, int(round(raw_score / 10.0))))
        
        if raw_score <= 30:
            level = "LOW"
            priority = "LOW"
            depth = "10 - 25 cm (Ankle Deep)"
            access = "PASSABLE_CAUTION"
            what = "Shallow water accumulation, surface runoff, or curbside puddles."
            recs = [
                "Proceed with caution on low-lying roads.",
                "Monitor local rainfall alerts.",
                "Avoid parking near storm drains."
            ]
        elif raw_score <= 60:
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
        elif raw_score <= 80:
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
        else:
            level = "EXTREME"
            priority = "CRITICAL"
            depth = "> 150 cm (Chest Deep to Roof Inundation)"
            access = "COMPLETELY_BLOCKED"
            what = "Severe catastrophic flooding, deep fast-flowing water, structural submersion."
            recs = [
                "Immediate tactical rescue deployment (boats, helicopters, emergency divers).",
                "Aerial distribution of clean drinking water, rations, and first aid.",
                "Establish medical triage stations at nearest elevated staging areas."
            ]
            
        return {
            "severity_score": severity_10,
            "severity_level": level,
            "rescue_priority": priority,
            "confidence": float(round(prob, 4)),
            "title": f"Flood Warning: {level} Severity Inundation (Score {severity_10}/10)",
            "description": reason,
            "what_is_in_image": what,
            "estimated_depth": depth,
            "road_access": access,
            "recommendations": recs,
            "flood_detected": True,
            "flood_probability": prob,
            "damage_score_100": raw_score
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
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
            "severity_score": 1,
            "severity_level": "LOW",
            "rescue_priority": "LOW",
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        result = pipeline.predict(tmp_path)
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

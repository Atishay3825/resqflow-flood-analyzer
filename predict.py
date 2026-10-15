import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Recreate architecture definitions so they can be loaded easily
class FloodClassifier(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class DamageRegressor(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x) * 100.0

class OrdinalDamageModel(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.reg_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.clf_head = nn.Linear(64, 6)
    def forward(self, x):
        features = self.shared(x)
        reg_out = self.reg_head(features) * 100.0
        clf_out = self.clf_head(features)
        return reg_out, clf_out

# Hard-coded reasoning mapping for the closest score
REASONS_MAP = {
    5: "Minor waterlogging; water is shallow and there is little visible disruption.",
    15: "Minor street flooding with shallow water and limited impact on traffic.",
    25: "Shallow flooding is clearly visible, but structures remain largely unaffected.",
    35: "Moderate water coverage affects the road and begins restricting normal movement.",
    45: "Water reaches vehicle lower-body level and significantly disrupts traffic.",
    55: "High water levels affect ground floors and residential access.",
    65: "Significant inundation covers a large residential area.",
    75: "Severe flooding leaves vehicles largely submerged and causes major disruption.",
    85: "Extensive urban inundation affects a very large area and causes severe disruption.",
    95: "Extreme widespread flooding with extensive inundation and severe community impact."
}

def get_reason(score):
    closest_score = min(REASONS_MAP.keys(), key=lambda x: abs(x - score))
    return REASONS_MAP[closest_score]

def check_document_or_non_scene(image):
    """
    Tier 1 Domain & Document Gating:
    Detects if an image is a printed/scanned document, application letter,
    text sheet, or non-natural graphic with low saturation and high paper luminance.
    """
    rgb = np.array(image, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return True, "Grayscale or single-channel document detected. Non-flood incident."
        
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = (max_c - min_c) / (max_c + 1e-5)
    
    # White/near-white paper pixels
    white_mask = (r > 190) & (g > 190) & (b > 190) & (saturation < 0.15)
    white_ratio = float(np.mean(white_mask))
    low_sat_ratio = float(np.mean(saturation < 0.18))
    dark_mask = (max_c < 120) & (saturation < 0.25)
    dark_ratio = float(np.mean(dark_mask))
    
    # Document detection criteria:
    # 1. High paper background (>55% white) and predominantly achromatic (>75%)
    # 2. Or scanned document with text (>40% white, dark text strokes >0.5%, >85% achromatic)
    if (white_ratio > 0.55 and low_sat_ratio > 0.75) or (white_ratio > 0.40 and dark_ratio > 0.005 and low_sat_ratio > 0.85):
        return True, "Document / text sheet detected (high paper luminance, low color saturation). No flood hazard."
        
    return False, None

def get_severity(score):
    if score <= 20: return "VERY_LOW"
    elif score <= 40: return "LOW"
    elif score <= 60: return "MODERATE"
    elif score <= 80: return "HIGH"
    elif score <= 95: return "VERY_HIGH"
    else: return "CRITICAL"

class FloodDamagePipeline:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        config_path = os.path.join(models_dir, "pipeline_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.threshold = self.config["threshold"]
        
        # Load feature extractor backbone
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.backbone.eval()
        self.backbone.to(self.device)
        
        # Load classifier
        self.classifier = FloodClassifier(input_dim=self.config["input_dim"]).to(self.device)
        self.classifier.load_state_dict(torch.load(os.path.join(models_dir, "flood_classifier.pth"), map_location=self.device))
        self.classifier.eval()
        
        # Load regressor
        if self.config["use_ordinal"]:
            self.damage_model = OrdinalDamageModel(input_dim=self.config["input_dim"]).to(self.device)
        else:
            self.damage_model = DamageRegressor(input_dim=self.config["input_dim"]).to(self.device)
            
        self.damage_model.load_state_dict(torch.load(os.path.join(models_dir, "damage_model.pth"), map_location=self.device))
        self.damage_model.eval()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        image = Image.open(image_path).convert('RGB')
        
        # Tier 1: Visual Domain & Document Gating
        is_doc, doc_reason = check_document_or_non_scene(image)
        if is_doc:
            return {
                "flood_detected": False,
                "flood_probability": 0.0,
                "damage_score": 0,
                "severity": "NOT_FLOOD",
                "reason": doc_reason
            }
            
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            features = self.backbone(img_tensor)
            
            # Predict flood probability
            clf_logits = self.classifier(features).squeeze(1)
            prob = torch.sigmoid(clf_logits).item()
            
            if prob < self.threshold:
                reason = "Safe: No reliable visual evidence of flooding or flood aftermath." if prob < 0.1 else "No reliable visual evidence of flooding or flood aftermath."
                return {
                    "flood_detected": False,
                    "flood_probability": float(prob),
                    "damage_score": 0,
                    "severity": "NOT_FLOOD",
                    "reason": reason
                }
            
            # If flood detected, run damage model
            if self.config["use_ordinal"]:
                score_tensor, _ = self.damage_model(features)
            else:
                score_tensor = self.damage_model(features)
                
            score = float(score_tensor.squeeze(0).item())
            # Clamp between 1 and 100 for flood images
            score = max(1.0, min(100.0, score))
            
            return {
                "flood_detected": True,
                "flood_probability": float(prob),
                "damage_score": int(round(score)),
                "severity": get_severity(score),
                "reason": get_reason(score)
            }

def main():
    parser = argparse.ArgumentParser(description="Inference script for Flood Damage ML model")
    parser.add_argument("--image", required=True, help="Path to the image to analyze")
    parser.add_argument("--models_dir", default=r"c:\Users\Atishay jain\Desktop\PROJECT\models", help="Directory where models are saved")
    args = parser.parse_args()
    
    try:
        pipeline = FloodDamagePipeline(models_dir=args.models_dir)
        result = pipeline.predict(args.image)
        
        print("\n=== Prediction Result ===")
        print(f"Flood Detected: {'YES' if result['flood_detected'] else 'NO'}")
        print(f"Flood Confidence: {result['flood_probability']*100:.1f}%")
        print(f"Damage Score: {result['damage_score']}/100")
        print(f"Severity: {result['severity']}")
        print(f"\nReason:\n{result['reason']}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

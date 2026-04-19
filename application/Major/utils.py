import os
import json
import pickle
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 224


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_pseudo_mask(rgb_img: np.ndarray) -> np.ndarray:
    """
    Foreground mask using Otsu thresholding + morphological operations.
    Returns a binary mask (0/1) of shape (H, W).
    """
    if rgb_img.dtype != np.uint8:
        rgb_img = rgb_img.astype(np.uint8)

    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If the mask covers more than 60 % of pixels it's likely inverted
    if np.mean(thresh > 0) > 0.6:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 1, 0).astype(np.uint8)
    else:
        mask = (opened > 0).astype(np.uint8)

    # Fallback: never return an empty mask
    if mask.sum() == 0:
        mask = np.ones_like(mask, dtype=np.uint8)

    return mask


def build_full_classifier(num_classes: int):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_feature_extractor():
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Identity()
    return model


def load_all_models(base_dir: str):
    model_dir = os.path.join(base_dir, "models")

    label_map_path  = os.path.join(model_dir, "label_mapping.json")
    full_model_path = os.path.join(model_dir, "efficientnetv2s_full_best.pth")
    feat_model_path = os.path.join(model_dir, "efficientnetv2s_feature_extractor.pth")
    xgb_path        = os.path.join(model_dir, "xgboost_brain_multiclass.pkl")

    for p in [label_map_path, full_model_path, feat_model_path, xgb_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(label_map_path, "r") as f:
        label_data = json.load(f)

    idx2label = {int(k): v for k, v in label_data["idx2label"].items()}
    num_classes = len(idx2label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_model = build_full_classifier(num_classes)
    full_model.load_state_dict(torch.load(full_model_path, map_location=device))
    full_model.to(device).eval()

    feature_extractor = build_feature_extractor()
    feature_extractor.load_state_dict(torch.load(feat_model_path, map_location=device))
    feature_extractor.to(device).eval()

    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=EfficientNet_V2_S_Weights.DEFAULT.transforms().mean,
            std=EfficientNet_V2_S_Weights.DEFAULT.transforms().std
        )
    ])

    target_layer = full_model.features[-1]

    return {
        "device": device,
        "idx2label": idx2label,
        "full_model": full_model,
        "feature_extractor": feature_extractor,
        "xgb_model": xgb_model,
        "transform": transform,
        "target_layer": target_layer
    }


def preprocess_for_model(image_path: str, transform):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    img_np = np.array(img).astype(np.uint8)
    mask   = create_pseudo_mask(img_np)

    # Apply mask to suppress background before classification
    masked = img_np.astype(np.float32) * np.expand_dims(mask, axis=-1)
    masked = np.clip(masked, 0, 255).astype(np.uint8)
    masked_pil = Image.fromarray(masked)

    tensor = transform(masked_pil).unsqueeze(0)
    return tensor, masked_pil, img_np   # also return original numpy for overlay


@torch.no_grad()
def predict_image(image_path: str, bundle: dict):
    x, masked_pil, original_np = preprocess_for_model(image_path, bundle["transform"])
    x = x.to(bundle["device"])

    features    = bundle["feature_extractor"](x)
    features_np = features.detach().cpu().numpy()

    probs       = bundle["xgb_model"].predict_proba(features_np)[0]
    pred_idx    = int(np.argmax(probs))
    confidence  = float(np.max(probs))
    predicted_label = bundle["idx2label"][pred_idx]

    sorted_idx = np.argsort(probs)[::-1][:3]
    top_predictions = [
        {
            "label": bundle["idx2label"][int(i)],
            "probability": round(float(probs[i]) * 100, 2)
        }
        for i in sorted_idx
    ]

    return {
        "pred_index":      pred_idx,
        "predicted_label": predicted_label,
        "confidence":      confidence,
        "top_predictions": top_predictions,
        "input_tensor":    x,
        "masked_pil":      masked_pil,
        "original_np":     original_np,   # original unmasked RGB image for overlay
    }


# ── Disease metadata for the result page ──────────────────────────────────────
DISEASE_INFO = {
    "Glioma": {
        "icon": "🧠",
        "color": "#e74c3c",
        "description": (
            "Glioma is a type of tumour that starts in the glial cells of the brain or spinal cord. "
            "It is the most common type of primary brain tumour. Treatment typically involves surgery, "
            "radiation therapy, and chemotherapy."
        ),
        "urgency": "High — please consult a neuro-oncologist promptly."
    },
    "Meningioma": {
        "icon": "🔬",
        "color": "#e67e22",
        "description": (
            "Meningioma is a tumour that arises from the meninges — the membranes surrounding the "
            "brain and spinal cord. Most meningiomas are benign and slow-growing. Regular monitoring "
            "or surgical removal may be recommended."
        ),
        "urgency": "Moderate — schedule a follow-up with a neurosurgeon."
    },
    "Pituitary": {
        "icon": "⚗️",
        "color": "#8e44ad",
        "description": (
            "Pituitary tumours develop in the pituitary gland at the base of the brain. They can "
            "affect hormone production. Many are benign adenomas treatable with medication, surgery, "
            "or radiation."
        ),
        "urgency": "Moderate — endocrinology and neurosurgery consultation advised."
    },
    "Normal": {
        "icon": "✅",
        "color": "#27ae60",
        "description": (
            "No tumour or significant abnormality was detected in this scan. The brain tissue "
            "appears within normal limits based on the model's analysis."
        ),
        "urgency": "No immediate action required. Continue routine check-ups."
    },
    "MildDemented": {
        "icon": "🧩",
        "color": "#f39c12",
        "description": (
            "Mild dementia indicates early-stage cognitive decline that affects daily activities "
            "to a noticeable degree. Early intervention with cognitive therapy and medication "
            "can slow progression."
        ),
        "urgency": "Moderate — consult a neurologist for cognitive assessment."
    },
    "ModerateDemented": {
        "icon": "⚠️",
        "color": "#e74c3c",
        "description": (
            "Moderate dementia involves significant memory loss and confusion that requires "
            "caregiver support. A comprehensive care plan including medication and structured "
            "daily routines is essential."
        ),
        "urgency": "High — immediate neurological and geriatric care recommended."
    },
    "NonDemented": {
        "icon": "✅",
        "color": "#27ae60",
        "description": (
            "No signs of dementia-related cognitive decline were detected. The scan is consistent "
            "with normal cognitive ageing patterns."
        ),
        "urgency": "No immediate action required. Maintain a healthy lifestyle."
    },
    "VeryMildDemented": {
        "icon": "🔍",
        "color": "#3498db",
        "description": (
            "Very mild dementia shows the earliest signs of cognitive decline, often only noticeable "
            "on detailed testing. Lifestyle modifications and monitoring are the first steps."
        ),
        "urgency": "Low-Moderate — periodic neurological follow-up advised."
    },
    "MS_Axial": {
        "icon": "🔗",
        "color": "#e67e22",
        "description": (
            "Multiple Sclerosis lesions detected in the axial plane. MS is a chronic inflammatory "
            "disease of the central nervous system. Disease-modifying therapies can significantly "
            "reduce relapse rates."
        ),
        "urgency": "High — neurology referral for MS workup recommended."
    },
    "MS_Saggital": {
        "icon": "🔗",
        "color": "#e67e22",
        "description": (
            "Multiple Sclerosis lesions detected in the sagittal plane. Early diagnosis and treatment "
            "with disease-modifying drugs can help manage progression."
        ),
        "urgency": "High — neurology referral for MS workup recommended."
    },
    "Control_Axial": {
        "icon": "✅",
        "color": "#27ae60",
        "description": (
            "The axial scan appears consistent with a healthy control — no MS lesions detected. "
        ),
        "urgency": "No immediate action required."
    },
    "Control_Saggital": {
        "icon": "✅",
        "color": "#27ae60",
        "description": (
            "The sagittal scan appears consistent with a healthy control — no MS lesions detected."
        ),
        "urgency": "No immediate action required."
    },
}


def get_disease_info(label: str) -> dict:
    """Return display metadata for a predicted label, with a safe fallback."""
    return DISEASE_INFO.get(label, {
        "icon": "🧠",
        "color": "#555",
        "description": "Detailed information for this class is not yet available.",
        "urgency": "Please consult a medical professional for interpretation."
    })


def create_gradcam_explanation(predicted_label: str, confidence: float) -> str:
    conf_pct = confidence * 100

    if conf_pct >= 90:
        strength = "very strong"
    elif conf_pct >= 75:
        strength = "strong"
    elif conf_pct >= 60:
        strength = "moderate"
    else:
        strength = "limited"

    return (
        f"The Grad-CAM heatmap highlights regions within the brain that most influenced "
        f"the model's prediction of <strong>{predicted_label}</strong>. "
        f"Warmer colours (red/orange) indicate higher activation — areas the model relied on most. "
        f"Cooler colours (blue/green) show regions of lower influence. "
        f"The heatmap is constrained to the brain foreground to eliminate background noise. "
        f"At {conf_pct:.1f}% confidence the network shows a <em>{strength}</em> response. "
        f"If the highlighted regions align with known anatomical markers for {predicted_label}, "
        f"this increases confidence in the prediction. Misaligned highlights warrant clinical review."
    )
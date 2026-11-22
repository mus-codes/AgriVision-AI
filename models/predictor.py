import torch
import json
from pathlib import Path
from functools import lru_cache

from models.model_loader import load_all_assets
from models.image_processor import preprocess_image


BASE_DIR = Path(__file__).resolve().parent.parent

# Crops supported by the specialist model
SPECIALIST_CROPS = {
    "Apple",
    "Corn_(maize)",
    "Grape",
    "Pepper,_bell_",
    "Potato",
    "Squash",
    "Tomato"
}


# Load models and class mappings once
ASSETS = load_all_assets()

PRIMARY_MODEL = ASSETS["primary_model"]
SPECIALIST_MODEL = ASSETS["specialist_model"]

DEVICE = ASSETS["device"]

PRIMARY_CLASS_MAPPING = ASSETS["primary_classes"]


@lru_cache(maxsize=1)
def get_specialist_mapping():
    checkpoint = torch.load(
        BASE_DIR / "models" / "efficientnet_b0_specialist_best.pth",
        map_location="cpu",
        weights_only=False
    )

    return checkpoint["specialist_class_to_idx"]


def format_class_name(class_name):
    crop, disease = class_name.split("___", 1)

    crop_names = {
        "Apple": "Apple",
        "Blueberry": "Blueberry",
        "Cherry_(including_sour)": "Cherry",
        "Corn_(maize)": "Corn",
        "Grape": "Grape",
        "Orange": "Orange",
        "Peach": "Peach",
        "Pepper,_bell_": "Bell Pepper",
        "Potato": "Potato",
        "Raspberry": "Raspberry",
        "Soybean": "Soybean",
        "Squash": "Squash",
        "Strawberry": "Strawberry",
        "Tomato": "Tomato"
    }

    crop_display = crop_names.get(crop, crop.replace("_", " "))

    disease_display = disease.replace("_", " ").strip()

    return {
        "crop": crop_display,
        "disease": disease_display
    }


def run_prediction(model, image_tensor, class_mapping):
    with torch.inference_mode():
        image_tensor = image_tensor.to(DEVICE)

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    reverse_mapping = {
        index: class_name
        for class_name, index in class_mapping.items()
    }

    predicted_class = reverse_mapping[predicted_index.item()]

    result = format_class_name(predicted_class)

    result["class_name"] = predicted_class
    result["confidence"] = round(confidence.item() * 100, 2)

    return result


def predict_image(image):

    # Preprocess uploaded image
    image_tensor = preprocess_image(image)

    # Primary 38-class prediction
    primary_result = run_prediction(
        PRIMARY_MODEL,
        image_tensor,
        PRIMARY_CLASS_MAPPING
    )

    # Specialist prediction when the primary model identifies
    # a crop supported by the specialist.
    specialist_result = None

    primary_raw_crop = primary_result["class_name"].split("___")[0]

    if primary_raw_crop in SPECIALIST_CROPS:

        specialist_mapping = get_specialist_mapping()

        specialist_result = run_prediction(
            SPECIALIST_MODEL,
            image_tensor,
            specialist_mapping
        )

        specialist_result["crop_matches_primary"] = (
            specialist_result["crop"] == primary_result["crop"]
        )

    return {
        "primary_prediction": primary_result,
        "specialist_prediction": specialist_result,
        "specialist_available": specialist_result is not None,
        "device": str(DEVICE)
    }
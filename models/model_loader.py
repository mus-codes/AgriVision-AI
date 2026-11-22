import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PRIMARY_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_best.pth"
SPECIALIST_MODEL_PATH = BASE_DIR / "models" / "efficientnet_b0_specialist_best.pth"

PRIMARY_MAPPING_PATH = BASE_DIR / "metadata" / "class_mapping.json"
SPECIALIST_MAPPING_PATH = BASE_DIR / "metadata" / "plantdoc_class_mapping.json"

DEVICE = torch.device("cpu")


def load_class_mapping(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        in_features=1280,
        out_features=num_classes
    )

    return model


def load_model(model_path, num_classes):
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = create_efficientnet(num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    return model


def load_all_assets():
    print("Loading AgriVision models...")

    primary_classes = load_class_mapping(PRIMARY_MAPPING_PATH)
    specialist_classes = load_class_mapping(SPECIALIST_MAPPING_PATH)

    primary_model = load_model(
        PRIMARY_MODEL_PATH,
        len(primary_classes)
    )

    specialist_model = load_model(
        SPECIALIST_MODEL_PATH,
        len(specialist_classes)
    )

    print("Primary model loaded:", len(primary_classes), "classes")
    print("Specialist model loaded:", len(specialist_classes), "classes")
    print("Device:", DEVICE)

    return {
        "primary_model": primary_model,
        "specialist_model": specialist_model,
        "primary_classes": primary_classes,
        "specialist_classes": specialist_classes,
        "device": DEVICE
    }


if __name__ == "__main__":
    assets = load_all_assets()
    print("\nAll AgriVision models loaded successfully!")
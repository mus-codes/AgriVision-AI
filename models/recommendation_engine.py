import json
import copy
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    BASE_DIR / "metadata" / "disease_information.json"
)


# Load disease information
with open(DATABASE_PATH, "r", encoding="utf-8") as file:
    DISEASE_DATABASE = json.load(file)


DISCLAIMER = DISEASE_DATABASE["disclaimer"]

DISEASE_CLASSES = DISEASE_DATABASE["classes"]


def get_recommendations(prediction):
    """
    Retrieve disease information for a model prediction.

    Parameters:
        prediction (dict): Prediction dictionary from predictor.py.

    Returns:
        dict: Disease information and recommendations.
    """

    if not isinstance(prediction, dict):
        raise TypeError("Prediction must be a dictionary.")

    class_name = prediction.get("class_name")

    if not class_name:
        raise ValueError("Prediction does not contain a class name.")

    if class_name not in DISEASE_CLASSES:
        raise ValueError(
            f"Disease information not found for: {class_name}"
        )

    # Retrieve disease information
    disease_info = copy.deepcopy(
        DISEASE_CLASSES[class_name]
    )

    # Get model score
    raw_score = float(prediction.get("confidence", 0.0))

    # Add prediction details
    disease_info["class_name"] = class_name

    disease_info["crop"] = prediction.get(
        "crop", "Unknown"
    )

    disease_info["predicted_condition"] = prediction.get(
        "disease", "Unknown"
    )

    disease_info["raw_softmax_score"] = round(raw_score, 2)

    # Explain score limitations
    disease_info["score_note"] = (
        "This is the model's raw softmax score, not a "
        "calibrated probability of disease. It should not "
        "be interpreted as diagnostic certainty."
    )

    # Add uncertainty warning
    disease_info["uncertainty_warning"] = (
        "AI predictions can be incorrect, especially on "
        "field images that differ from training images. "
        "Check the symptoms carefully and consult an "
        "agricultural expert before taking treatment decisions."
    )

    # Add educational disclaimer
    disease_info["disclaimer"] = DISCLAIMER

    return disease_info
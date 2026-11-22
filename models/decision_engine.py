from models.recommendation_engine import get_recommendations


def assess_prediction(prediction_result):
    """
    Combine primary and specialist predictions into
    an explainable screening assessment.
    """

    if not isinstance(prediction_result, dict):
        raise TypeError("Prediction result must be a dictionary.")

    primary = prediction_result.get("primary_prediction")

    if not primary:
        raise ValueError("Primary prediction is missing.")

    # Get disease recommendations for primary prediction
    recommendations = get_recommendations(primary)

    specialist = prediction_result.get("specialist_prediction")

    # Prepare the assessment
    if specialist is None:
        agreement_status = "primary_only"
        agreement_message = (
            "A specialist prediction was not available. "
            "The result is based on the primary model only."
        )

    elif specialist.get("class_name") == primary.get("class_name"):
        agreement_status = "models_agree"
        agreement_message = (
            "The primary and specialist models predicted "
            "the same class. This agreement is supportive, "
            "but does not confirm the diagnosis."
        )

    else:
        agreement_status = "models_disagree"
        agreement_message = (
            "The primary and specialist models predicted "
            "different classes. Treat this result with "
            "additional caution and seek expert assessment."
        )

    # Prepare specialist information
    specialist_information = None

    if specialist is not None:
        specialist_information = {
            "class_name": specialist.get("class_name"),
            "crop": specialist.get("crop"),
            "disease": specialist.get("disease"),
            "raw_softmax_score": specialist.get("confidence"),
            "crop_matches_primary": specialist.get(
                "crop_matches_primary"
            )
        }

    # Final screening assessment
    assessment = {
        "primary_prediction": primary,
        "specialist_prediction": specialist_information,
        "specialist_available": specialist is not None,
        "agreement_status": agreement_status,
        "agreement_message": agreement_message,
        "screening_result": {
            "crop": primary.get("crop"),
            "predicted_condition": primary.get("disease"),
            "status": recommendations.get("status")
        },
        "recommendations": recommendations,
        "device": prediction_result.get("device", "unknown"),
        "disclaimer": recommendations["disclaimer"]
    }

    return assessment
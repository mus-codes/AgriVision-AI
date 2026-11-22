# AgriVision AI

An AI-powered plant disease screening application that analyzes leaf images to identify potential plant diseases.

## Features

- Multi-crop plant disease detection.
- EfficientNet-B0 primary classification model.
- Specialist model for additional prediction analysis.
- Grad-CAM visual explanations.
- Disease information and management recommendations.
- Interactive Streamlit interface.
- CPU-based inference support.

## Supported Crops

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato.

## Technology Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone this repository.
2. Open the project folder in your terminal.
3. Create a virtual environment:

   `python -m venv .venv`

4. Activate it:

   `.venv\Scripts\activate`

5. Install dependencies:

   `pip install -r requirements.txt`

6. Start AgriVision:

   `streamlit run app.py`

## Project Structure

- `app.py` — Main Streamlit application.
- `models/` — Model architecture, prediction, Grad-CAM and decision modules, and trained weights.
- `metadata/` — Class mappings and disease information.
- `requirements.txt` — Python dependencies.

## Model Assets

The trained model checkpoints and required metadata are included in this repository. No separate Hugging Face download is required.

## Important Disclaimer

AgriVision AI is an educational plant disease screening tool. Its predictions are not a substitute for professional agricultural diagnosis.

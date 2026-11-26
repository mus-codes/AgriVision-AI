# 🌿 AgriVision AI
### An Explainable Deep Learning-Based Plant Disease Detection and Screening System

<p align="center">
  <strong>AI-Powered Plant Disease Screening | Deep Learning | Computer Vision | Explainable AI</strong>
</p>

<p align="center">
  <a href="https://agrivision--ai.streamlit.app/"><strong>🚀 Live Application</strong></a> •
</p>

---

## 1. Project Overview

**AgriVision AI** is an AI-powered plant disease screening system developed using Deep Learning and Computer Vision. It analyzes uploaded leaf images to identify potential plant diseases and provides relevant disease information and recommendations.

The system uses Convolutional Neural Networks (CNNs), particularly the EfficientNet-B0 architecture, to extract visual features from plant leaf images and classify them into different healthy and diseased categories.

In addition to disease classification, AgriVision AI incorporates Explainable Artificial Intelligence (XAI) through Grad-CAM (Gradient-weighted Class Activation Mapping). This helps users visualize the image regions that influenced the model's prediction.

The system also incorporates a secondary disease-specialist model and a decision engine to provide an additional perspective on compatible disease categories.

The complete application is integrated into an interactive Streamlit interface, allowing users to upload leaf images and receive predictions, visual explanations, and disease-related information.

## 2. Problem Statement

Plant diseases can negatively affect agricultural productivity, crop quality, and food security. Early identification of plant diseases is important because delayed recognition may allow infections to spread and cause further damage.

Traditional disease identification often depends on manual observation and expert knowledge. However, identifying diseases through visual symptoms can be difficult, especially when different diseases produce similar symptoms.

AgriVision AI addresses this challenge by applying deep learning and computer vision techniques to automate the preliminary screening of plant leaf diseases.

The system aims to make image-based disease screening more accessible while providing visual explanations to help users understand the model's predictions.

## 3. Project Objectives

The major objectives of AgriVision AI are:

- Develop an AI-based system for plant disease classification using leaf images.
- Apply transfer learning to build an effective image classification model.
- Support multiple crops and disease categories through a unified classification system.
- Implement a secondary disease-specialist model for additional classification analysis.
- Integrate Explainable AI using Grad-CAM.
- Develop a decision mechanism to compare predictions from multiple models.
- Provide disease-related information and recommendations.
- Build an interactive and accessible web application using Streamlit.

## 4. Key Features

| Feature | Description |
|---|---|
| Multi-Crop Classification | Supports 14 crop categories through the primary model. |
| Deep Learning Detection | Uses EfficientNet-B0 for image classification. |
| Secondary Specialist Model | Provides an additional disease-focused prediction. |
| Explainable AI | Uses Grad-CAM to visualize influential image regions. |
| Model Comparison | Compares compatible predictions from both models. |
| Decision Engine | Identifies model agreement, disagreement, or primary-only results. |
| Disease Information | Provides disease descriptions and related information. |
| Recommendations | Displays relevant disease management suggestions. |
| Interactive Interface | Provides image upload and prediction through Streamlit. |

## 5. System Architecture

AgriVision AI follows a modular architecture in which image processing, model prediction, explainability, decision-making, and recommendation generation are handled by separate components.

### Overall Workflow

```text
             USER
               |
               v
       Upload Leaf Image
               |
               v
       STREAMLIT INTERFACE
               |
               v
       IMAGE PREPROCESSING
               |
               v
       PROCESSED IMAGE TENSOR
               |
          +----+----+
          |         |
          v         v
     PRIMARY     SPECIALIST
      MODEL        MODEL
          |         |
          v         v
      38 CLASS    16 CLASS
     PREDICTION  PREDICTION
          |         |
          +----+----+
               |
               v
        DECISION ENGINE
               |
          +----+----+
          |         |
          v         v
        GRAD-CAM  RECOMMENDATION
          |         ENGINE
          +----+----+
               |
               v
         FINAL RESULTS
               |
               v
       STREAMLIT DISPLAY
```

### Workflow Explanation

1. The user uploads a plant leaf image through the Streamlit interface.
2. The image is passed to the preprocessing module.
3. The processed image is converted into a tensor suitable for deep learning inference.
4. The primary EfficientNet-B0 model generates a prediction across 38 classes.
5. The secondary specialist model generates an additional prediction across its 16 disease classes when applicable.
6. The decision engine compares compatible model outputs.
7. Grad-CAM generates a visual explanation for the primary prediction.
8. The recommendation engine retrieves relevant disease information.
9. The application displays the prediction, explanation, model comparison, and recommendations.

## 6. Dataset Description

The project uses the **PlantVillage dataset** as its primary training and validation dataset.

PlantVillage is a publicly available plant leaf image dataset containing healthy and diseased leaf images across multiple crop categories.

### Dataset Statistics

The original PlantVillage dataset contains approximately 54,305 images.

After cleaning and preparing the dataset, the project used the following image distribution:

| Dataset Split | Number of Images |
|---|---:|
| Training Dataset | 43,433 |
| Validation Dataset | 10,851 |
| Total Cleaned Images | 54,284 |

### Dataset Purpose

- Training images are used to teach the model visual patterns associated with different plant conditions.
- Validation images are used to monitor model performance during training and support model selection.

### Crop Categories

The primary classification system supports the following 14 crop categories:

1. Apple
2. Blueberry
3. Cherry
4. Corn
5. Grape
6. Orange
7. Peach
8. Pepper
9. Potato
10. Raspberry
11. Soybean
12. Squash
13. Strawberry
14. Tomato

### PlantDoc Dataset

The project also uses PlantDoc for additional evaluation in natural-scene image conditions.

PlantDoc contains plant images captured in more varied visual environments. It provides an additional way to examine how image classification models behave beyond the relatively controlled images of PlantVillage.

A compatible subset of PlantDoc disease categories is used for specialist-model evaluation.

## 7. Deep Learning Models

AgriVision AI incorporates two EfficientNet-B0 models, each serving a different classification purpose.

### 7.1 Primary Model — EfficientNet-B0

The primary model is responsible for the main plant disease classification task.

It supports **38 output classes**, covering healthy and diseased conditions across the supported crops.

EfficientNet-B0 is a Convolutional Neural Network (CNN) architecture designed to achieve a balance between computational efficiency and model performance.

Important architectural components include:

- Convolutional Layers: Extract visual features such as edges, shapes, and textures.
- Batch Normalization: Helps stabilize activations during training.
- SiLU Activation: Introduces non-linearity into the network.
- MBConv Blocks: Perform efficient feature extraction using inverted bottleneck convolutions.
- Depthwise Convolution: Performs spatial filtering efficiently across feature channels.
- Squeeze-and-Excitation: Learns the relative importance of feature channels.
- Global Average Pooling: Converts spatial feature maps into compact feature representations.
- Dropout: Helps reduce overfitting.
- Fully Connected Layer: Maps extracted features to the target classification categories.
- Softmax: Converts output scores into normalized class scores.

### 7.2 Secondary Model — Disease Specialist

The secondary model is another EfficientNet-B0 model designed for disease-focused classification.

It supports 16 disease classes and provides an additional prediction for compatible disease categories.

Unlike the primary model, the specialist model does not contain healthy classes.

Its purpose is to provide a second disease-focused perspective rather than replace the primary classifier.

### 7.3 Transfer Learning

The project uses transfer learning to adapt pretrained deep learning models to plant disease classification.

Transfer learning allows a model to reuse visual features learned from a large image dataset rather than learning every feature from the beginning.

The general process involves:

1. Loading pretrained EfficientNet-B0 weights.
2. Adapting the classification layer to the target classes.
3. Training the model using plant disease images.
4. Evaluating model behaviour using validation data.
5. Saving the trained model weights for later inference.

## 8. Image Preprocessing

Image preprocessing is an important stage because deep learning models require input images in a consistent numerical format.

AgriVision AI performs the following operations:

### Step 1: Image Upload

The user uploads a leaf image through the application.

### Step 2: EXIF Orientation Correction

The image orientation is corrected using its EXIF metadata, helping prevent incorrectly rotated images from entering the model.

### Step 3: RGB Conversion

The image is converted into RGB format, ensuring three consistent colour channels.

### Step 4: Resizing

The image is resized to 256 × 256 pixels.

### Step 5: Center Cropping

A central region of 224 × 224 pixels is extracted.

### Step 6: Normalization

The pixel values are normalized using ImageNet mean and standard deviation values.

This ensures that the input follows the normalization convention expected by the pretrained architecture.

### Step 7: Tensor Conversion

The processed image is converted into a PyTorch tensor.

**Final Input Tensor Shape:**

```text
[1, 3, 224, 224]
```

Where:

- 1 represents batch size.
- 3 represents RGB channels.
- 224 represents image height.
- 224 represents image width.

The final tensor is passed to the trained classification models.

## 9. Model Prediction Process

After preprocessing, the image is passed through the primary classification model.

The model performs the following operations:

1. Extracts visual features through convolutional layers.
2. Processes these features through EfficientNet's MBConv blocks.
3. Generates class scores through the classification head.
4. Applies Softmax to obtain normalized class scores.
5. Selects the highest-scoring class as the primary prediction.

The specialist model follows a similar process but predicts among its 16 disease-specific classes.

## 10. Decision Engine

AgriVision AI includes a decision engine to compare the predictions generated by the primary and specialist models.

The decision engine considers the compatibility of the predicted classes before comparing their outputs.

It supports three decision states:

| Decision State | Meaning |
|---|---|
| `models_agree` | Both models produce compatible matching disease predictions. |
| `models_disagree` | The models produce different compatible disease predictions. |
| `primary_only` | Only the primary prediction is applicable or available for comparison. |

This component helps organize the outputs from both classification models into an understandable result.

## 11. Explainable AI — Grad-CAM

A major component of AgriVision AI is Explainable Artificial Intelligence (XAI).

Deep learning models can produce predictions without clearly showing which image regions influenced their decisions.

To improve interpretability, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

### How Grad-CAM Works

1. The trained model generates a class prediction.
2. Gradients are calculated with respect to the selected class.
3. These gradients are used to determine the importance of feature maps from the final convolutional feature block.
4. A class activation heatmap is generated.
5. The heatmap is resized and overlaid on the original leaf image.

### Purpose of Grad-CAM

Grad-CAM provides a visual representation of image regions that influenced the model's prediction.

It helps users inspect whether the model focused on relevant visible patterns when making its classification.

**Important:** Grad-CAM provides an approximate visual explanation. It does not prove that a highlighted region contains a particular disease.

## 12. Disease Information and Recommendation Engine

After generating the prediction, the application retrieves relevant disease information from its metadata database.

The recommendation engine uses:

`metadata/disease_information.json`

This file contains structured disease-related information used by the application.

The recommendation process includes:

1. Receiving the predicted disease class.
2. Looking up the corresponding disease entry.
3. Retrieving the available description and management information.
4. Displaying the relevant information through the Streamlit interface.

This component connects the model's classification output with useful information for the user.

## 13. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Pretrained computer vision architectures and image transformations |
| EfficientNet-B0 | Primary and specialist classification architecture |
| CNN | Image feature extraction and classification |
| Grad-CAM | Explainable AI |
| NumPy | Numerical operations |
| Pillow | Image loading and processing |
| OpenCV | Image processing support |
| Matplotlib | Visualization |
| Streamlit | Interactive web application |
| JSON | Class mappings and disease metadata |
| Git | Version control |
| GitHub | Source code hosting and collaboration |
| Streamlit Community Cloud | Application deployment |

## 14. Project Folder Structure

```text
AgriVision/
│
├── app.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── metadata/
│   ├── class_mapping.json
│   ├── plantdoc_class_mapping.json
│   └── disease_information.json
│
└── models/
    ├── efficientnet_b0_best.pth
    ├── efficientnet_b0_specialist_best.pth
    │
    ├── model_loader.py
    ├── image_processor.py
    ├── predictor.py
    ├── gradcam.py
    ├── decision_engine.py
    └── recommendation_engine.py
```

### Description of Important Files

| File | Responsibility |
|---|---|
| `app.py` | Main Streamlit application and user interface |
| `model_loader.py` | Loads trained models, model weights, and class mappings |
| `image_processor.py` | Handles image preprocessing and tensor conversion |
| `predictor.py` | Performs primary and specialist model predictions |
| `gradcam.py` | Generates Grad-CAM visual explanations |
| `decision_engine.py` | Compares model predictions |
| `recommendation_engine.py` | Retrieves disease information and recommendations |
| `class_mapping.json` | Stores primary model class mappings |
| `plantdoc_class_mapping.json` | Stores specialist model class mappings |
| `disease_information.json` | Stores disease descriptions and recommendation information |
| `requirements.txt` | Contains the Python dependencies required by the project |

## 15. Installation and Setup

Follow these steps to run AgriVision AI locally.

### Prerequisites

- Python
- Git
- Visual Studio Code or another Python-compatible editor

### Step 1: Clone the Repository

```bash
git clone https://github.com/mus-codes/AgriVision-AI.git
```

### Step 2: Open the Project Directory

```bash
cd AgriVision-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will start locally and Streamlit will provide a local browser address.

## 16. Live Application

AgriVision AI is deployed using Streamlit Community Cloud.

**Live Application:**

https://agrivision--ai.streamlit.app/

Users can open the application directly in a browser, upload a plant leaf image, and explore the available screening features.

**GitHub Repository:**

https://github.com/mus-codes/AgriVision-AI

The GitHub repository provides access to the project source code, model weights, metadata, and setup instructions.

## 17. Applications and Potential Impact

AgriVision AI demonstrates how artificial intelligence and computer vision can be applied to agricultural image analysis.

Potential applications include:

- Preliminary screening of visible plant leaf diseases.
- Educational demonstrations of deep learning in agriculture.
- Understanding the use of CNN architectures in image classification.
- Exploring Explainable AI in agricultural applications.
- Supporting further research into automated plant disease recognition.

The project provides a foundation for future development in AI-assisted agricultural analysis.

## 18. Limitations and Responsible Use

AgriVision AI is an educational and research-oriented plant disease screening system.

Its predictions are based on patterns learned from the available image datasets. Real-world disease identification may also depend on environmental conditions, plant variety, growth stage, and other factors that may not be visible in a photograph.

The displayed model scores should not be interpreted as guaranteed diagnostic probabilities.

Grad-CAM visualizations are approximate explanations and should not be treated as proof of disease.

The application is intended to support preliminary screening and learning, not replace agricultural experts or laboratory diagnosis.

## 19. Future Scope

The project can be extended through:

- Adding more crop species and disease categories.
- Incorporating larger and more diverse agricultural image datasets.
- Improving performance on natural-scene images.
- Developing a mobile-friendly application.
- Adding multilingual support for farmers.
- Integrating weather and environmental information.
- Exploring real-time camera-based plant image analysis.
- Extending the recommendation system with additional agricultural knowledge.

## 20. Conclusion

AgriVision AI combines Deep Learning, Computer Vision, and Explainable Artificial Intelligence to develop an interactive plant disease screening application.

By integrating EfficientNet-B0 classification models, image preprocessing, Grad-CAM visualization, a model decision engine, and a recommendation system, the project demonstrates a complete AI-based image analysis pipeline.

The modular architecture makes it possible to understand, maintain, and extend individual components independently.

Overall, AgriVision AI demonstrates the practical application of modern deep learning techniques to an agricultural problem while emphasizing model interpretability and accessibility.

---

<p align="center">
  <strong>🌱 AgriVision AI — Making Plant Disease Screening Smarter Through Artificial Intelligence.</strong>
</p>

**Developed as an academic AI project.**

# 🌿 AgroVision AI – Plant Disease Detection System

An AI-powered plant disease detection application built using **Deep Learning, TensorFlow, and MobileNetV2**. The system classifies plant leaf images into 38 disease categories and provides visual explanations using Grad-CAM.

🚀 **Live Demo:** [AgroVision AI – Streamlit App](YOUR_STREAMLIT_APP_URL)

💻 **GitHub Repository:** [AgroVision AI](https://github.com/Nimish-0070/AgroVision-AI)

---

## 📌 Project Overview

AgroVision AI is a deep learning-based application designed to identify plant diseases from leaf images.

The project uses transfer learning with MobileNetV2, pretrained on ImageNet, followed by fine-tuning on the PlantVillage dataset.

Users can upload a plant leaf image and receive:
- Predicted plant disease.
- Model softmax score.
- Top 3 predicted categories.
- Grad-CAM visualization to understand which image regions influenced the prediction.
- Image validation and preliminary image screening warnings.

## 🎯 Key Features

- **38-Class Classification:** Identifies 38 PlantVillage plant disease and healthy-leaf categories.
- **Transfer Learning:** Uses pretrained MobileNetV2.
- **Fine-Tuning:** Fine-tunes selected layers to improve classification performance.
- **Top-3 Predictions:** Displays the three highest-scoring categories.
- **Grad-CAM Explainability:** Visualizes regions influencing the model's prediction.
- **Image Validation:** Checks image format, dimensions, file size, and potential blur.
- **Image Screening:** Uses ImageNet predictions as a preliminary safeguard against unrelated images.
- **Interactive Web Application:** Built using Streamlit.

## 🧠 Model Architecture

The project uses MobileNetV2 as its feature extraction backbone.

**Architecture:**

1. Input Image (224 × 224 × 3)
2. MobileNetV2 Feature Extractor
3. Global Average Pooling
4. Dropout (0.3)
5. Dense Layer (38 neurons)
6. Softmax Activation

### Training Strategy

**Phase 1 – Feature Extraction**
- Loaded ImageNet pretrained MobileNetV2 weights.
- Froze the pretrained base model.
- Trained the classification head.

**Phase 2 – Fine-Tuning**
- Unfroze selected final layers.
- Kept Batch Normalization layers frozen.
- Used a lower learning rate for fine-tuning.

## 📊 Dataset

**PlantVillage Dataset**

- Total images: 54,305
- Number of categories: 38
- Image type: RGB leaf images
- Dataset source: [PlantVillage on Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)

### Dataset Split

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 38,013 | 70% |
| Validation | 8,146 | 15% |
| Testing | 8,146 | 15% |
| **Total** | **54,305** | **100%** |

The dataset was divided using stratified splitting to preserve class representation.

## 📈 Model Performance

The trained model achieved the following results on the held-out PlantVillage test set:

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | **97.81%** |
| Test Loss | 0.0859 |
| Macro Precision | 97.33% |
| Macro Recall | 96.80% |
| Macro F1-Score | 97.03% |
| Weighted F1-Score | 97.81% |

- Test images: 8,146
- Correct predictions: 7,968
- Incorrect predictions: 178

**Note:** These results represent performance on the held-out PlantVillage test split. They should not be interpreted as guaranteed accuracy on real-world farm photographs.

## 🔍 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize the image regions that influence the model's prediction.

The application displays:

1. Original Image
2. Grad-CAM Heatmap
3. Heatmap Overlay

This provides a visual interpretation of the model's decision-making process.

Grad-CAM highlights influential regions but does not establish the biological location of a disease.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | MobileNetV2 |
| Data Processing | NumPy |
| Image Processing | OpenCV, Pillow |
| Visualization | Matplotlib, Grad-CAM |
| Web Framework | Streamlit |
| Dataset | PlantVillage |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 📂 Project Structure

```text
AgroVision-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── agrovision_final.weights.h5
    └── class_names.json
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Nimish-0070/AgroVision-AI.git
```

### 2. Navigate to the Project Directory

```bash
cd AgroVision-AI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## ⚠️ Limitations

- The classifier is trained on the PlantVillage dataset, which contains relatively controlled-background images.
- Real-world images may differ in lighting, background, camera quality, and leaf orientation.
- The disease classifier is limited to its 38 trained categories.
- ImageNet screening is preliminary and may incorrectly classify unfamiliar plant images.
- Softmax scores are not calibrated probabilities.
- Grad-CAM provides visual explanations, not proof of disease localization.

This application is an educational and research prototype, not a replacement for professional agricultural diagnosis.

## 🔮 Future Improvements

- Improve generalization using diverse real-world plant images.
- Add support for more plant species and diseases.
- Improve out-of-distribution image detection.
- Expand model evaluation using external datasets.
- Explore mobile application integration.

## 👨‍💻 Author

**Nimish Sawant**

B.Tech – Computer Science and Engineering

GitHub: [Nimish-0070](https://github.com/Nimish-0070)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, TensorFlow, MobileNetV2, and Streamlit.**


🚀 **Live Demo:** [AgroVision AI – Plant Disease Detection](https://agrovision-ai-dfvazhyapbqocokzeqrfzj.streamlit.app/)

💻 **GitHub Repository:** [AgroVision AI](https://github.com/Nimish-0070/AgroVision-AI)
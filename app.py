# ==============================================
# AGROVISION AI - PLANT DISEASE DETECTION SYSTEM
# Deep Learning | MobileNetV2 | Grad-CAM
# Image validation + unsupported-image screening
# ==============================================

import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as plant_preprocess_input,
)
from tensorflow.keras.applications.mobilenet_v2 import (
    decode_predictions,
)


# ==============================================
# 1. APPLICATION CONFIGURATION
# ==============================================

st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🌿",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

IMG_SIZE = 224
WEIGHTS_PATH = MODEL_DIR / "agrovision_final.weights.h5"
CLASS_PATH = MODEL_DIR / "class_names.json"

# Basic upload safeguards. These are input-quality checks, not disease checks.
MAX_FILE_SIZE_MB = 20
MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 6000
MAX_ASPECT_RATIO = 5.0

# A conservative warning threshold for the plant-vs-other screening stage.
# It is a heuristic and must be validated against a representative image set.
NON_PLANT_TOP1_THRESHOLD = 0.35

# Confidence below this level triggers a caution message.
# This is NOT a calibrated probability threshold.
LOW_CONFIDENCE_THRESHOLD = 0.70

PLANT_LABEL_KEYWORDS = {
    "leaf", "plant", "flower", "corn", "maize", "cucumber",
    "strawberry", "orange", "lemon", "lime", "banana",
    "pineapple", "pomegranate", "fig", "jackfruit", "custard",
    "bell_pepper", "pepper", "broccoli", "cauliflower",
    "cabbage", "artichoke", "cardoon", "mushroom", "acorn",
    "ear", "pot", "greenhouse", "daisy", "yellow_lady",
    "rapeseed", "head_cabbage", "Granny_smith",
}


# ==============================================
# 2. LOAD THE TRAINED PLANT DISEASE MODEL
# ==============================================

@st.cache_resource
def load_model_and_classes():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}\n"
            "Place agrovision_final.weights.h5 inside the models folder."
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_PATH}\n"
            "Place class_names.json inside the models folder."
        )

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="mobilenetv2_1.00_224",
    )

    # Recreate the same fine-tuning trainability configuration used in training.
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(
        38,
        activation="softmax",
        name="dense",
    )(x)

    reconstructed_model = Model(
        inputs=inputs,
        outputs=outputs,
        name="AgroVision_MobileNetV2",
    )

    reconstructed_model.load_weights(str(WEIGHTS_PATH))

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != 38:
        raise ValueError(
            f"Expected 38 class names, found {len(class_names)}."
        )

    return reconstructed_model, class_names


# ==============================================
# 3. LOAD A SECONDARY IMAGE SCREENING MODEL
# ==============================================

@st.cache_resource
def load_image_screening_model():
    """
    ImageNet MobileNetV2 is used only as a broad object-screening signal.
    It is not trained to diagnose plant diseases and is not a perfect
    plant/non-plant detector.
    """
    return MobileNetV2(
        weights="imagenet",
        include_top=True,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )


try:
    model, class_names = load_model_and_classes()
except Exception as exc:
    st.error("AgroVision AI could not load its trained model.")
    st.exception(exc)
    st.stop()


# ==============================================
# 4. VALIDATE AND PREPROCESS UPLOADED IMAGE
# ==============================================

def validate_uploaded_image(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ValueError(
            f"File is too large. Maximum allowed size is "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    try:
        # Verify file integrity before fully decoding it.
        with Image.open(uploaded_file) as check_image:
            detected_format = check_image.format
            check_image.verify()

        uploaded_file.seek(0)

        with Image.open(uploaded_file) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "This file is not a valid or readable image. "
            "Please upload a JPG or PNG image."
        ) from exc

    width, height = image.size

    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(
            f"Image is too small ({width} × {height}). "
            f"Both sides must be at least {MIN_IMAGE_SIDE} pixels."
        )

    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"Image dimensions are too large ({width} × {height}). "
            f"The maximum side length is {MAX_IMAGE_SIDE} pixels."
        )

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        raise ValueError(
            "The image has an unusually wide or tall aspect ratio. "
            "Please upload a photo focused on one leaf."
        )

    image_array = np.asarray(image)

    # A simple blur warning; this does not determine whether the image is a leaf.
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return image, image_array, detected_format, blur_score


def preprocess_plant_image(image):
    image_array = np.asarray(image, dtype=np.float32)

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )
    image_tensor = tf.image.resize(
        image_tensor,
        [IMG_SIZE, IMG_SIZE],
    )
    image_tensor = plant_preprocess_input(image_tensor)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    return image_tensor


# ==============================================
# 5. SCREEN FOR CLEARLY UNRELATED OBJECT IMAGES
# ==============================================

def label_looks_plant_related(label):
    normalized = label.lower().replace(" ", "_")
    return any(
        keyword.lower().replace(" ", "_") in normalized
        for keyword in PLANT_LABEL_KEYWORDS
    )


def screen_image_content(image):
    """
    Returns one of:
      likely_plant: at least one plant-related ImageNet label appears in top 5
      likely_non_plant: a confident non-plant top-1 label with no plant cue
      uncertain: ImageNet does not provide a clear signal

    This is a heuristic screening layer, not a certified OOD detector.
    """
    screening_model = load_image_screening_model()

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    scores = screening_model(batch, training=False).numpy()

    decoded = decode_predictions(scores, top=5)[0]

    # DEBUG: Print ImageNet screening predictions in the VS Code terminal.
    print("\n========== IMAGE SCREENING DEBUG ==========")
    for item in decoded:
        print(item[1], f"{item[2] * 100:.2f}%")
    print("===========================================\n")
    top5 = [
        {
            "label": label,
            "description": description.replace("_", " "),
            "score": float(score),
        }
        for label, description, score in decoded
    ]

    has_plant_cue = any(
        label_looks_plant_related(item["description"])
        for item in top5
    )

    top1 = top5[0]

    if has_plant_cue:
        status = "likely_plant"
        message = (
            "The screening model found a plant-related visual category. "
            "Continue with the plant disease classifier."
        )
    elif top1["score"] >= NON_PLANT_TOP1_THRESHOLD:
        status = "likely_non_plant"
        message = (
            "This image appears more consistent with a non-plant object "
            f"({top1['description']}, screening score "
            f"{top1['score']:.1%}). Please verify the image before continuing."
        )
    else:
        status = "uncertain"
        message = (
            "The image screening model could not confidently determine "
            "whether this is a plant image. Review the image before continuing."
        )

    return status, message, top5


# ==============================================
# 6. PREDICT PLANT DISEASE
# ==============================================

def predict_disease(image_tensor):
    predictions = model(
        image_tensor,
        training=False,
    ).numpy()[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    top_indices = np.argsort(predictions)[-3:][::-1]

    return predicted_index, confidence, top_indices, predictions


# ==============================================
# 7. GRAD-CAM EXPLAINABILITY
# ==============================================

@st.cache_resource
def create_gradcam_components():
    base_model = model.get_layer("mobilenetv2_1.00_224")

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("out_relu").output,
    )

    gap_layer = model.layers[-3]
    dropout_layer = model.layers[-2]
    dense_layer = model.layers[-1]

    return (
        feature_extractor,
        gap_layer,
        dropout_layer,
        dense_layer,
    )


def generate_gradcam(image_tensor, class_index):
    (
        feature_extractor,
        gap_layer,
        dropout_layer,
        dense_layer,
    ) = create_gradcam_components()

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(
            image_tensor,
            training=False,
        )
        tape.watch(conv_outputs)

        x = gap_layer(conv_outputs)
        x = dropout_layer(x, training=False)
        predictions = dense_layer(x)
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise ValueError("Could not calculate Grad-CAM gradients.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0],
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ==============================================
# 8. FORMAT DISEASE LABELS
# ==============================================

def format_label(label):
    return label.replace("___", " - ").replace("_", " ")


# ==============================================
# 9. APPLICATION HEADER
# ==============================================

st.title("🌿 AgroVision AI")
st.subheader("Deep Learning Based Plant Disease Detection")

st.markdown(
    """
    Upload a plant leaf image to classify it into one of 38
    PlantVillage categories using MobileNetV2.
    """
)

st.caption(
    "Educational research prototype. Predictions are not a substitute "
    "for diagnosis by an agricultural expert."
)

st.divider()


# ==============================================
# 10. IMAGE UPLOAD AND BASIC VALIDATION
# ==============================================

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        original_image, original_array, image_format, blur_score = (
            validate_uploaded_image(uploaded_file)
        )

        image_tensor = preprocess_plant_image(original_image)

        st.subheader("Uploaded Image")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(
                original_image,
                caption="Original Leaf Image",
                use_container_width=True,
            )

        with col2:
            st.write(
                f"**Image dimensions:** "
                f"{original_image.width} × {original_image.height}"
            )
            st.write(f"**Image format:** {image_format}")
            st.write(f"**Supported categories:** {len(class_names)}")
            st.write(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

            if blur_score < 20:
                st.warning(
                    "This image may be blurry or out of focus. "
                    "A clearer leaf photograph may improve the result."
                )

        st.divider()

        # ==========================================
        # 11. CONTENT SCREENING AND PREDICTION
        # ==========================================

        # Clear a previous screening result when a different file is uploaded.
        current_upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("agrovision_upload_key") != current_upload_key:
            st.session_state["agrovision_upload_key"] = current_upload_key
            st.session_state.pop("agrovision_screening_result", None)

        if st.button(
            "🔍 Detect Plant Disease",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Checking image content..."):
                try:
                    screening_status, screening_message, screening_top5 = (
                        screen_image_content(original_image)
                    )
                except Exception as screen_error:
                    screening_status = "uncertain"
                    screening_message = (
                        "The image screening model could not run. "
                        "Please inspect the uploaded image before continuing."
                    )
                    screening_top5 = []
                    st.warning(f"Screening detail: {screen_error}")

            # Save the result so the confirmation form survives Streamlit reruns.
            st.session_state["agrovision_screening_result"] = {
                "status": screening_status,
                "message": screening_message,
                "top5": screening_top5,
            }

        screening_result = st.session_state.get("agrovision_screening_result")

        if screening_result is not None:
            screening_status = screening_result["status"]
            screening_message = screening_result["message"]
            screening_top5 = screening_result["top5"]
            proceed = False

            if screening_status == "likely_non_plant":
                st.error("Image screening warning: possible non-plant image")
                st.warning(
                    screening_message.replace(
                        "Disease prediction has been blocked.",
                        "The screening result may be incorrect for unfamiliar plant images.",
                    )
                )
                st.info(
                    "ImageNet screening can confuse unfamiliar plant photos with "
                    "unrelated categories. Check the preview carefully. If this "
                    "is genuinely a plant leaf, you can choose to continue."
                )

            elif screening_status == "uncertain":
                st.warning("Image screening is inconclusive")
                st.write(screening_message)

            else:
                st.success("Plant-related image signal detected.")
                st.caption(screening_message)
                proceed = True

            if screening_top5 and screening_status != "likely_plant":
                st.caption(
                    "ImageNet screening labels: "
                    + ", ".join(
                        f"{item['description']} ({item['score']:.1%})"
                        for item in screening_top5[:5]
                    )
                )

            # Explicit confirmation is required for a non-plant or uncertain signal.
            if screening_status in {"likely_non_plant", "uncertain"}:
                with st.form("confirm_plant_image_form"):
                    user_confirms_plant = st.checkbox(
                        "I have checked the uploaded image. It is a plant leaf, "
                        "and I want to continue with an experimental prediction."
                    )
                    submitted = st.form_submit_button(
                        "Continue to Disease Prediction",
                        type="primary",
                        use_container_width=True,
                    )

                proceed = bool(submitted and user_confirms_plant)

                if submitted and not user_confirms_plant:
                    st.warning(
                        "Please confirm that the uploaded image is a plant leaf "
                        "before continuing."
                    )

                if proceed:
                    st.warning(
                        "Proceeding despite the screening warning. The disease "
                        "classifier may produce an unreliable result for this image."
                    )

            if proceed:
                try:
                    with st.spinner("Analyzing leaf image..."):
                        (
                            predicted_index,
                            confidence,
                            top_indices,
                            predictions,
                        ) = predict_disease(image_tensor)

                        predicted_disease = class_names[predicted_index]

                    st.subheader("Disease Detection Result")

                    if confidence < LOW_CONFIDENCE_THRESHOLD:
                        st.warning(
                            "Low model score: the classifier is not strongly "
                            "favoring one category. Please treat this result "
                            "with caution."
                        )
                    else:
                        st.success("The classifier produced a strong top score.")

                    st.markdown(f"### 🌱 {format_label(predicted_disease)}")
                    st.metric(
                        "Model softmax score",
                        f"{confidence * 100:.2f}%",
                    )
                    st.progress(float(confidence))
                    st.info(
                        "The softmax score reflects the model's relative output "
                        "for this image. It is not a calibrated probability or "
                        "a guarantee that the prediction is correct."
                    )

                    st.subheader("Top 3 Predictions")
                    for index in top_indices:
                        st.write(
                            f"**{format_label(class_names[index])}**"
                            f" — {predictions[index] * 100:.2f}%"
                        )
                        st.progress(float(predictions[index]))

                    st.divider()

                    with st.spinner("Generating Grad-CAM explanation..."):
                        heatmap = generate_gradcam(
                            image_tensor,
                            predicted_index,
                        )

                    st.subheader("AI Explainability - Grad-CAM")
                    heatmap_resized = cv2.resize(
                        heatmap,
                        (original_array.shape[1], original_array.shape[0]),
                    )
                    heatmap_uint8 = np.uint8(255 * heatmap_resized)
                    colored_heatmap = cv2.applyColorMap(
                        heatmap_uint8,
                        cv2.COLORMAP_JET,
                    )
                    colored_heatmap = cv2.cvtColor(
                        colored_heatmap,
                        cv2.COLOR_BGR2RGB,
                    )
                    overlay = cv2.addWeighted(
                        original_array,
                        0.6,
                        colored_heatmap,
                        0.4,
                        0,
                    )

                    grad_col1, grad_col2, grad_col3 = st.columns(3)
                    with grad_col1:
                        st.image(
                            original_array,
                            caption="Original Image",
                            use_container_width=True,
                        )
                    with grad_col2:
                        st.image(
                            heatmap_resized,
                            caption="Grad-CAM Heatmap",
                            clamp=True,
                            use_container_width=True,
                        )
                    with grad_col3:
                        st.image(
                            overlay,
                            caption="Heatmap Overlay",
                            use_container_width=True,
                        )

                    st.caption(
                        "Grad-CAM highlights image regions that influenced "
                        "the selected prediction. It does not prove biological "
                        "disease localization."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")


    except Exception as image_error:
        st.error(f"Image validation failed: {image_error}")


# ==============================================
# 12. FOOTER
# ==============================================

st.divider()

st.caption(
    "AgroVision AI | TensorFlow • MobileNetV2 • Transfer Learning • Grad-CAM"
)

st.caption(
    "Image screening is a preliminary safeguard, not a guaranteed "
    "plant detector. Verify results with an agricultural expert."
)

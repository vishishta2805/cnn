import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "plant_disease_model.h5"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿"
)

# =========================
# TITLE
# =========================

st.title("🌿 Plant Disease Detection")

st.write("Upload a plant leaf image.")

# =========================
# CLASS LABELS
# =========================

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image",
             use_container_width=True)

    # Resize image
    image = image.resize((128, 128))

    # Convert to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.subheader("Prediction")

    st.success(
        f"{class_names[predicted_class]}"
    )

    st.write(
        f"Confidence: {confidence:.2f}"
    )

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Solar Panel Defect Detector", layout="centered")
st.title("🔍 Solar Panel Defect Detector")

# Load trained model
model = tf.keras.models.load_model("solar_panel_defect_classifier.h5")

# Class names in the same order as training
class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

# Upload image
uploaded_file = st.file_uploader("Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    st.success(f"Prediction: **{class_names[class_idx]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

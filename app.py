import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Solar Panel Defect Detection", layout="centered")

CLASSES  = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
IMG_SIZE = (224, 224)

ACTIONS = {
    "Bird-drop":         ("5 - 15%",   "Clean the panel within the next 2 to 3 days."),
    "Clean":             ("0%",        "No action needed. Panel is in good condition."),
    "Dusty":             ("10 - 25%",  "Clean the panel within the next 1 to 2 weeks."),
    "Electrical-damage": ("30 - 100%", "Disconnect the panel immediately and contact a technician."),
    "Physical-Damage":   ("20 - 80%",  "Arrange for repair or replacement within 48 hours."),
    "Snow-Covered":      ("50 - 100%", "Remove the snow using a soft brush. Do not use metal tools."),
}

@st.cache_resource
def load_model():
    if os.path.exists("solar_panel_model.h5"):
        return tf.keras.models.load_model("solar_panel_model.h5")
    return None

model = load_model()

st.title("Solar Panel Defect Detection")
st.write("Upload a photo of a solar panel to find out its condition and what action needs to be taken.")
st.write("---")

uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model is None:
        st.error("The model file solar_panel_model.h5 was not found. Please run the notebook first to train and save the model.")
    else:
        img_arr = np.array(image)
        img_arr = cv2.resize(img_arr, IMG_SIZE).astype(np.float32)
        inp     = np.expand_dims(img_arr, axis=0)

        preds = model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(preds))
        label = CLASSES[idx]
        conf  = preds[idx]
        loss_pct, action = ACTIONS[label]

        st.write("---")
        st.subheader("Result")
        st.write(f"**Condition detected:** {label}")

        col1, col2 = st.columns(2)
        col1.metric("Confidence", f"{conf * 100:.1f}%")
        col2.metric("Estimated efficiency loss", loss_pct)

        st.write("**Recommended action:**")
        st.info(action)

        st.write("---")
        st.write("**Confidence for all classes:**")
        for i, cls in enumerate(CLASSES):
            col_a, col_b = st.columns([4, 1])
            col_a.progress(float(preds[i]), text=cls)
            col_b.write(f"{preds[i] * 100:.1f}%")

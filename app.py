import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image

# ---------------------- Settings ----------------------
TARGET_SIZE = (128, 128)

# ---------------------- Helper Functions ----------------------

def convert_image_to_array(image, target_size=TARGET_SIZE):
    try:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, target_size)
        image = img_to_array(image) / 255.0  # normalize
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# ---------------------- Load Model and Labels ----------------------

@st.cache_resource
def load_model_and_labels():
    model = load_model("crop_health_model.h5")
    with open("label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)
    return model, label_binarizer

# ---------------------- Streamlit UI ----------------------

st.title("🌾 Crop Health Classifier")
st.write("Upload a crop leaf image to predict its health status.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing..."):
        model, label_binarizer = load_model_and_labels()
        processed_image = convert_image_to_array(image)

        if processed_image is not None:
            processed_image = np.expand_dims(processed_image, axis=0)
            predictions = model.predict(processed_image)
            predicted_index = np.argmax(predictions)
            predicted_label = label_binarizer.classes_[predicted_index]

            st.success(f"✅ Predicted Class: **{predicted_label}**")
        else:
            st.error("Image could not be processed correctly.")

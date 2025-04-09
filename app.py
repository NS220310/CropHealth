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

@st.cache_resource
def load_crop_model_and_labels():
    model = load_model("crop_health_model.h5")
    with open("label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)
    return model, label_binarizer

@st.cache_resource
def load_irrigation_model():
    with open("irrigation_model.pkl", "rb") as f:
        model = pickle.load(f)
    # If it's a tuple, unpack it
    if isinstance(model, tuple):
        model = model[0]
    return model


# ---------------------- Streamlit UI ----------------------

st.title("🌾 Smart Agriculture Assistant")

st.sidebar.title("Select a Feature")
feature = st.sidebar.radio("Choose what you want to do:", ("Crop Health Detection", "Irrigation Scheduling"))

# ---------- Crop Health Detection ----------
if feature == "Crop Health Detection":
    st.header("🌿 Crop Health Classifier")
    st.write("Upload a crop leaf image to predict its health status.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing..."):
            model, label_binarizer = load_crop_model_and_labels()
            processed_image = convert_image_to_array(image)

            if processed_image is not None:
                processed_image = np.expand_dims(processed_image, axis=0)
                predictions = model.predict(processed_image)
                predicted_index = np.argmax(predictions)
                predicted_label = label_binarizer.classes_[predicted_index]

                st.success(f"✅ Predicted Class: **{predicted_label}**")
            else:
                st.error("Image could not be processed correctly.")

# ---------- Irrigation Scheduling ----------
elif feature == "Irrigation Scheduling":
    st.header("💧 Irrigation Scheduling Predictor")
    st.write("Enter environmental parameters to predict irrigation level.")

    temp = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)
    pressure = st.number_input("⏲️ Pressure (hPa)", min_value=800.0, max_value=1100.0, step=0.1)
    altitude = st.number_input("⛰️ Altitude (meters)", min_value=0.0, max_value=5000.0, step=1.0)
    soil_moisture = st.number_input("💧 Soil Moisture (%)", min_value=0.0, max_value=100.0, step=0.1)

    if st.button("Predict Irrigation Level"):
        irrigation_model = load_irrigation_model()
        input_features = np.array([[temp, pressure, altitude, soil_moisture]])
        prediction = irrigation_model.predict(input_features)
        classes = ['Dry', 'Very Dry', 'Very Wet', 'Wet']
        predicted_label = classes[int(prediction[0])]
        st.success(f"🪴 Predicted Irrigation Category: **{predicted_label}**")

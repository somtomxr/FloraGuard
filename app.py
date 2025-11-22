import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Set page config
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Classifier")
st.markdown("Upload an image of a plant leaf to detect diseases.")

# Load model and class indices
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    return model

@st.cache_data
def load_class_indices():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Ensure keys are integers
    return {int(k): v for k, v in class_indices.items()}

try:
    model = load_model()
    class_names = load_class_indices()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            # Preprocess the image
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class_name = class_names[predicted_class_index]

            st.write(f"### Prediction: {predicted_class_name}")
            st.write(f"**Confidence:** {confidence:.2f}")

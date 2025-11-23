import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FloraGuard AI", layout="centered", page_icon="🌿")

# --- UI & CSS INJECTION ---
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* General Body Styling */
.stApp {
    font-family: 'Outfit', sans-serif;
    background-image: 
        radial-gradient(at 0% 0%, hsla(152,100%,93%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(135,100%,96%,1) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(152,100%,93%,1) 0, transparent 50%);
    background-attachment: fixed;
    color: #1e293b;
}

/* TOP MARGIN FIX */
.block-container {
    padding-top: 5rem !important; 
    padding-bottom: 2rem !important;
}

/* Navbar Styling */
.nav-container {
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    margin-bottom: 2rem;
}

.nav-links-group {
    display: flex; 
    align-items: center; 
    gap: 2rem;
}

.nav-link {
    color: #000000 !important; /* Pure Black */
    text-decoration: none !important;
    font-weight: 500;
    font-size: 0.95rem;
    transition: color 0.2s;
    cursor: pointer;
}
.nav-link:hover {
    color: #16a34a !important;
    text-decoration: none;
}

.nav-btn {
    background-color: #0f172a;
    color: white !important;
    padding: 0.6rem 1.4rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}
.nav-btn:hover {
    transform: translateY(-2px);
    color: white !important;
    text-decoration: none;
}

/* --- CUSTOM FILE UPLOADER STYLING --- */
/* Add margin to push it down from the hero text */
[data-testid='stFileUploader'] {
    margin-top: 3rem !important;
}

/* Hide the default label */
[data-testid='stFileUploader'] label {
    display: none;
}

/* Target the main dropzone container */
[data-testid='stFileUploader'] section {
    background-color: white;
    border: 2px dashed #bbf7d0;
    border-radius: 1.5rem;
    padding: 3rem 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    transition: border-color 0.3s, box-shadow 0.3s;
    
    /* Force Flex Column for vertical stacking */
    display: flex;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Target internal Streamlit containers */
[data-testid='stFileUploader'] section > div {
    display: flex;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100%;
}

/* Hover effects */
[data-testid='stFileUploader'] section:hover {
    border-color: #22c55e;
    background-color: #f0fdf4;
}

/* Icon styling */
[data-testid='stFileUploader'] svg {
    width: 50px !important;
    height: 50px !important;
    fill: #16a34a !important;
    color: #16a34a !important;
    margin-bottom: 1rem !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Text styling ("Drag and drop file here") */
[data-testid='stFileUploader'] div[data-testid="stMarkdownContainer"] p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    margin-bottom: 0.5rem !important;
    text-align: center !important;
}

/* Small text styling ("Limit 200MB...") */
[data-testid='stFileUploader'] small {
    color: #94a3b8;
    font-size: 0.85rem;
    text-align: center;
    margin-bottom: 1.5rem !important;
    display: block;
}

/* Button styling */
[data-testid='stFileUploader'] button {
    background-color: #16a34a;
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: background-color 0.3s;
    margin-top: 1rem !important;
    display: block !important;
    margin-left: auto !important;
    margin-right: auto !important;
    position: relative !important;
    z-index: 999 !important;
    cursor: pointer !important;
}
[data-testid='stFileUploader'] button:hover {
    background-color: #15803d;
    color: white;
    border: none;
}

/* Result Card Styling */
.result-card {
    animation: slideUp 0.6s ease-out;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Custom Predict Button */
div.stButton > button {
    background-color: #16a34a;
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 0.75rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px -1px rgba(22, 163, 74, 0.2);
    width: 100%;
    margin-top: 1rem;
}
div.stButton > button:hover {
    background-color: #15803d;
    box-shadow: 0 10px 15px -3px rgba(22, 163, 74, 0.3);
    transform: translateY(-2px);
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --- BACKEND LOGIC START (From app2.py) ---

HF_MODEL_URL = "https://huggingface.co/somtomxr/FloraGuard/resolve/main/plant_disease_model.h5"
LOCAL_PATH = "plant_disease_model.h5"

# Download once
if not os.path.exists(LOCAL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(HF_MODEL_URL)
        with open(LOCAL_PATH, "wb") as f:
            f.write(r.content)

# Load model and class indices
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(LOCAL_PATH)
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
    # st.success("Model loaded successfully!") # Commented out to prevent UI clutter
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- BACKEND LOGIC END ---


# --- UI HEADER SECTION (HTML) ---
st.markdown("""
<!-- Navigation -->
<div class="nav-container">
<div style="display: flex; align-items: center; gap: 0.75rem;">
<div style="background-color: #16a34a; padding: 0.5rem; border-radius: 0.75rem; color: white; box-shadow: 0 4px 6px -1px rgba(22, 197, 94, 0.3); display: flex; align-items: center; justify-content: center;">
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>
</div>
<span style="font-size: 1.5rem; font-weight: 700; color: #0f172a; tracking-tight: -0.025em; line-height: 1;">Flora<span style="color: #16a34a;">Guard</span></span>
</div>

<div class="nav-links-group">
<a href="#" class="nav-link">About Us</a>
<a href="#" class="nav-link">How it Works</a>
<a href="#" class="nav-btn">API Access</a>
</div>
</div>

<!-- Hero Section -->
<div style="text-align: center; margin-bottom: 2rem; animation: slideUp 0.8s ease-out;">
<div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.75rem; background-color: #dcfce7; color: #15803d; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1.5rem; border: 1px solid #bbf7d0;">
<span style="width: 8px; height: 8px; background-color: #22c55e; border-radius: 50%; display: inline-block;"></span>
System Online
</div>
<h1 style="font-size: 3.5rem; font-weight: 800; color: #0f172a; margin-bottom: 1rem; line-height: 1.1; letter-spacing: -0.025em;">
Plant Health Check <br/>
<span style="background: linear-gradient(to right, #16a34a, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Instantly.</span>
</h1>
<p style="font-size: 1.125rem; color: #64748b; max-width: 600px; margin: 0 auto; line-height: 1.6;">
Upload a photo of your plant's leaf. Our advanced AI will identify diseases.
</p>
</div>
""", unsafe_allow_html=True)


# --- MAIN INTERACTION AREA ---

# File uploader (Styled by CSS above)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Centered Image Preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a spacer
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centered Predict Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button('Predict')

    if predict_btn:
        with st.spinner('Analyzing...'):
            # --- LOGIC START (Prediction) ---
            # Preprocess the image
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class_name = class_names[predicted_class_index].replace('_', ' ')
            # --- LOGIC END ---

            # Calculate confidence percentage for the bar
            confidence_percent = int(confidence * 100)
            
            # Determine color based on health (simple logic)
            is_healthy = "healthy" in predicted_class_name.lower()
            color_theme = "green" if is_healthy else "orange" if confidence_percent < 80 else "red"
            
            # --- RESULT CARD (HTML Injection) ---
            result_html = f"""
<div class="result-card" style="margin-top: 2rem; background: white; border-radius: 1.5rem; padding: 2rem; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 2rem;">
<div>
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
<span style="background-color: #dcfce7; color: #15803d; padding: 0.25rem 0.75rem; border-radius: 0.5rem; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Diagnosis</span>
<span style="color: #94a3b8; font-size: 0.875rem;">AI Analysis</span>
</div>
<h2 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin: 0;">{predicted_class_name}</h2>
<p style="color: #64748b; margin-top: 0.5rem;">The model has identified this pattern with high accuracy.</p>
</div>
<div style="background-color: #f8fafc; border-radius: 1rem; padding: 1rem; border: 1px solid #e2e8f0; min-width: 180px;">
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 0.5rem;">
<span style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Confidence</span>
<span style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(90deg, #2ECC71, #27AE60); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent;">{confidence_percent}%</span>
</div>
<div style="width: 100%; height: 8px; background-color: #e2e8f0; border-radius: 9999px; overflow: hidden;">
<div style="width: {confidence_percent}%; height: 100%; background: linear-gradient(90deg, #2ECC71, #27AE60); border-radius: 9999px;"></div>
</div>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
<div style="background-color: #fff7ed; padding: 1.5rem; border-radius: 1rem; border: 1px solid #ffedd5;">
<h3 style="color: #9a3412; font-weight: 700; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
Analysis
</h3>
<p style="color: #431407; font-size: 0.9rem; line-height: 1.6;">
The uploaded leaf shows visual characteristics consistent with <strong>{predicted_class_name}</strong>.
</p>
</div>
<div style="background-color: #eff6ff; padding: 1.5rem; border-radius: 1rem; border: 1px solid #dbeafe;">
<h3 style="color: #1e40af; font-weight: 700; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
Recommendation
</h3>
<p style="color: #1e3a8a; font-size: 0.9rem; line-height: 1.6;">
{'Maintain current care routine.' if is_healthy else 'Isolate the plant and consult a local agricultural extension for specific fungicide or treatment options.'}
</p>
</div>
</div>
</div>
"""
            st.markdown(result_html, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.875rem; margin-top: 5rem; padding-bottom: 3rem;">
&copy; 2025 FloraGuard AI. Powered by TensorFlow.
</div>
""", unsafe_allow_html=True)
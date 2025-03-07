import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

import gdown


import subprocess

# 🔹 Raw GitHub link to your model
MODEL_URL = "https://raw.githubusercontent.com/chennamsettisravya/LDD/main/mobile_net_new.h5"
MODEL_PATH = "mobile_net_new.h5"

# 🔹 Function to download model using wget
@st.cache_resource  # Cache to avoid re-downloading
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model... Please wait.")
        try:
            subprocess.run(["wget", MODEL_URL, "-O", MODEL_PATH], check=True)
            st.success("✅ Model downloaded successfully!")
        except subprocess.CalledProcessError:
            st.error("❌ Failed to download model.")
            return None
    return tf.keras.models.load_model(MODEL_PATH)


# 🔹 Load the model
model = download_model()



import json
import requests

# 🔹 GitHub Raw URL for class_indices.json
JSON_URL = "https://raw.githubusercontent.com/chennamsettisravya/LDD/main/class_indices.json"

# 🔹 Download and Load JSON Directly
try:
    response = requests.get(JSON_URL)
    response.raise_for_status()  # Check if the request was successful
    class_indices = response.json()  # Load JSON content
    print("✅ JSON loaded successfully!")
except requests.exceptions.RequestException as e:
    print(f"❌ Failed to fetch JSON: {e}")
except json.JSONDecodeError:
    print("❌ Error decoding JSON")




# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(to right, #4CAF50, #2E7D32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .results-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .stAlert {
        background-color: #e8f5e9;
        border-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="upload-header"><h1>🌿 Plant Disease Classifier</h1></div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.write("### Upload a plant leaf image to detect diseases")
st.write("Our model will analyze the image and identify potential diseases.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_image is not None:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Image Preview")
        image = Image.open(uploaded_image)
        resized_img = image.resize((400, 400))
        st.image(resized_img, use_column_width=True)

    with col2:
        st.markdown("### Analysis Results")
        if st.button('Analyze Image'):
            with st.spinner('Processing your image...'):
                # Add a small delay to show the spinner
                
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                
                st.success(f'### Detected Condition:\n# {str(prediction)}')
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666;'>
    <p>Made by Plant Disease Detection Team</p>
</div>
""", unsafe_allow_html=True)

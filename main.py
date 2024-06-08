import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames from pickle files
@st.cache_data
def load_data():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        filenames = pickle.load(open('filenames.pkl', 'rb'))
        return feature_list, filenames
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

feature_list, filenames = load_data()
if feature_list is None or filenames is None:
    st.stop()

# Load the ResNet50 model pre-trained on ImageNet
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    model.trainable = False
    return tf.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

model = load_model()

st.title('Fashion Recommendation System')

# Function to save uploaded files
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error in saving file: {e}")
        return 0

# Function to extract features from an image
def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# Function to recommend similar images based on extracted features
def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommending: {e}")
        return None

# Steps to upload, save, and process the image
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)
        
        # Extract features from the uploaded image
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        if features is not None:
            # Get recommendations based on extracted features
            indices = recommend(features, feature_list)
            
            if indices is not None:
                # Display the recommended images
                st.write("Recommended Images:")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(indices[0]):  # Ensure we do not exceed the number of recommendations
                        with col:
                            st.image(filenames[indices[0][i]], use_column_width=True)
    else:
        st.header("Some error occurred in file upload")

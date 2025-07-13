import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('models/final_model.h5')

def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)[0][0]
    return "Rotten 🍂" if result > 0.5 else "Fresh 🍏"

st.title("🍎 Smart Sorting: Fresh vs Rotten Classifier")
st.write("Upload an image of a fruit or vegetable to check if it's fresh or rotten.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Analyzing...")
    label = predict(img)
    st.success(f"Prediction: {label}")

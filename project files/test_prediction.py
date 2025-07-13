from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("models/final_model.h5")

def predict_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Rotten" if prediction > 0.5 else "Fresh"

# Replace with your actual image path
img_path = "dataset/fresh/freshbanana/b_f001.png" 
print("Prediction:", predict_image(img_path))

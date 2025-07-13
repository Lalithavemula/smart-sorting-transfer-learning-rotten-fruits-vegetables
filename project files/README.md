# 🍎 Smart Sorting: Detecting Rotten Fruits and Vegetables using Transfer Learning

This project detects whether a fruit or vegetable is fresh or rotten using deep learning. It is powered by MobileNetV2, a transfer learning model pre-trained on ImageNet.

---

## 🧠 Problem Solved

Sorting produce manually is time-consuming and error-prone in:

- 🏭 Food factories
- 🏬 Supermarkets
- 🏡 Smart homes

This tool helps identify rotten items using images.

---

## 📁 Folder Structure

```
SmartSortingProject/
├── dataset/           ← fresh/ and rotten/ images
├── models/            ← trained model (final_model.h5)
├── app/               ← Streamlit GUI app
│   └── main.py
├── training_model.py  ← trains the MobileNetV2 model
├── test_prediction.py ← test using sample image
├── requirements.txt   ← Python libraries list
├── README.md          ← this file
```

---

## 🚀 How to Run

1. Install the required Python libraries:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit web app:
```bash
python -m streamlit run app/main.py
```

3. Upload a fruit/vegetable image — the app tells if it’s:
- 🟢 Fresh
- 🔴 Rotten
---

## 🧪 Model Details

- Base model: MobileNetV2 (pre-trained on ImageNet)
- Input image size: 224x224
- Output: Binary classification
- Loss: Binary Crossentropy
- Optimizer: Adam
---

## 👩‍💻 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit (frontend)
- PIL (image processing)
- NumPy

---

## 📸 Example

Upload a fruit image like:

- `freshbanana.jpg` → ✅ Prediction: Fresh 🍏  
- `rottenapple.jpg` → ❌ Prediction: Rotten 🍂

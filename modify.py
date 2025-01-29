import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ========== Model Paths ==========
MODEL_PATH = "resnet50.h5"
SURVIVAL_MODEL_PATH = "random_survival_forest_model.joblib"

# ========== Fix: Google Drive Model Download ==========
if not os.path.exists(MODEL_PATH):
    file_id = "1gsSx_qNKbMnWHTj8NyEH0m1-SdmL59j_"
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ========== Fix: Model Loading with Error Handling ==========
try:
    survival_model = joblib.load(SURVIVAL_MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading survival model: {e}")
    survival_model = None

try:
    all_model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error loading ALL detection model: {e}")
    all_model = None

# ========== Class Labels & Recommendations ==========
class_names = ['Benign', 'Pre', 'Pro', 'Early']
recommendations = {
    "Benign": "No signs of ALL detected. Maintain regular health check-ups.",
    "Pre": "Early indications of ALL. Consult a hematologist for further testing.",
    "Pro": "Progressive stage of ALL. Seek immediate medical attention.",
    "Early": "Early-stage ALL detected. Initiate treatment as advised by an oncologist."
}

# ========== Image Preprocessing ==========
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224, 224))  # ResNet50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# ========== Function to Predict Survival ==========
def predict_survival(data):
    if survival_model:
        survival_function = survival_model.predict_survival_function(data, return_array=True)
        return survival_function[0]
    else:
        st.warning("⚠️ Survival model not loaded.")
        return None

# ========== Streamlit UI ==========
st.title("🩸 Acute Lymphoblastic Leukemia (ALL) Diagnosis & Survival Forecasting")
st.markdown("This app provides **survival forecasting** and **blood smear analysis** for ALL detection.")

# Sidebar navigation
st.sidebar.title("🔍 Choose an Option")
option = st.sidebar.radio("Select:", ["Survival Forecasting", "ALL Prediction"])

# ========== Survival Forecasting ==========
if option == "Survival Forecasting":
    st.sidebar.header("🧑‍⚕️ Patient Information")
    age = st.sidebar.slider("Age", 1, 100, 30)
    wbc_count = st.sidebar.slider("WBC Count (×10³/μL)", 0, 50, 20)
    rbc_count = st.sidebar.slider("RBC Count (×10⁶/μL)", 0, 10, 5)
    hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 0, 20, 12)
    platelet_count = st.sidebar.slider("Platelet Count (×10³/μL)", 50, 500, 150)
    lymphoblast_percentage = st.sidebar.slider("Lymphoblast Percentage", 0, 100, 50)
    chromosomal_abnormalities = st.sidebar.selectbox("Chromosomal Abnormalities", [0, 1])

    input_data = pd.DataFrame({
        'Age': [age],
        'WBC_Count': [wbc_count],
        'RBC_Count': [rbc_count],
        'Hemoglobin': [hemoglobin],
        'Platelet_Count': [platelet_count],
        'Lymphoblast_Percentage': [lymphoblast_percentage],
        'Chromosomal_Abnormalities': [chromosomal_abnormalities]
    })

    if st.button("🔮 Predict Survival"):
        survival_probs = predict_survival(input_data)

        if survival_probs is not None:
            plt.figure(figsize=(10, 6))
            plt.step(np.arange(len(survival_probs)), survival_probs, where="post", color='green', label='Survival Function')
            plt.axhline(y=0.5, color='orange', linestyle='--', label='50% Survival Probability')
            plt.axvline(x=12, color='blue', linestyle='--', label='12 Months Milestone')
            plt.axvline(x=24, color='purple', linestyle='--', label='24 Months Milestone')
            plt.axvline(x=36, color='red', linestyle='--', label='36 Months Milestone')
            plt.xlabel("Time (Months)")
            plt.ylabel("Survival Probability")
            plt.title("🩺 Predicted Survival Function")
            plt.legend()
            st.pyplot(plt)

            survival_df = pd.DataFrame({'Time (Months)': np.arange(len(survival_probs)), 'Survival Probability': survival_probs})
            st.dataframe(survival_df)

# ========== ALL Prediction ==========
elif option == "ALL Prediction":
    st.subheader("📸 Upload a Blood Smear Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image and all_model:
        st.image(uploaded_image, caption="📍 Uploaded Image", use_column_width=True)
        
        preprocessed_image = preprocess_image(uploaded_image)
        probabilities = all_model.predict(preprocessed_image)[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        predicted_probability = probabilities[predicted_index] * 100

        st.subheader("🔬 Prediction Results")
        st.metric("🔍 Likelihood of ALL", f"{predicted_probability:.2f} %")
        st.metric("🩸 Predicted Stage", predicted_class)
        st.write(f"**📌 Recommendations:** {recommendations[predicted_class]}")

        # Fix: Improved bar chart rendering
        probabilities_df = pd.DataFrame({"Class": class_names, "Probability (%)": probabilities * 100})
        st.bar_chart(probabilities_df.set_index("Class"))

        # Pie chart
        st.subheader("📊 Visualization")
        fig, ax = plt.subplots()
        ax.pie([predicted_probability, 100 - predicted_probability], labels=[predicted_class, "Other"], autopct="%1.1f%%", startangle=140, colors=["#76c7b7", "#e0e0e0"])
        ax.axis("equal")
        st.pyplot(fig)

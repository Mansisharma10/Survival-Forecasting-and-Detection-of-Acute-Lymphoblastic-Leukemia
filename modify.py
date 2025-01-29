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

# Download model from Google Drive if not already present
MODEL_PATH = "re.h5"
SURVIVAL_MODEL_PATH = "random_survival_forest_model.joblib"

if not os.path.exists(MODEL_PATH):
    file_id = "1gsSx_qNKbMnWHTj8NyEH0m1-SdmL59j_"
    gdown.download(f"https://drive.google.com/file/d/1gsSx_qNKbMnWHTj8NyEH0m1-SdmL59j_/view?usp=sharing", MODEL_PATH, quiet=False)

# Load models
survival_model = joblib.load(SURVIVAL_MODEL_PATH)
all_model = load_model(MODEL_PATH)

# Class names and recommendations for ALL prediction
class_names = ['Benign', 'Pre', 'Pro', 'Early']
recommendations = {
    "Benign": "No signs of ALL detected. Maintain regular health check-ups and a healthy lifestyle.",
    "Pre": "Early indications of ALL. Consult a hematologist for further testing and monitoring.",
    "Pro": "Progressive stage of ALL. Seek immediate medical attention for diagnosis and treatment planning.",
    "Early": "Early-stage ALL detected. Initiate treatment as advised by an oncologist."
}

# Helper function to preprocess the image
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize((224, 224))  # ResNet50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to predict survival probability
def predict_survival(data):
    survival_function = survival_model.predict_survival_function(data, return_array=True)
    return survival_function[0]

# Streamlit UI
st.title("🩸 Acute Lymphoblastic Leukemia (ALL) Diagnosis & Survival Forecasting")
st.markdown("This integrated application provides both **survival forecasting** for ALL patients and **microscopic blood smear analysis** to predict the likelihood of ALL.")

# Sidebar navigation
st.sidebar.title("🔍 Choose an Option")
option = st.sidebar.radio("Select:", ["Survival Forecasting", "ALL Prediction"])

if option == "Survival Forecasting":
    # Input form for survival forecasting
    st.sidebar.header("🧑‍⚕️ Patient Information")
    age = st.sidebar.slider("Age", 1, 100, 30)
    wbc_count = st.sidebar.slider("WBC Count (×10³/μL)", 0, 50, 20)
    rbc_count = st.sidebar.slider("RBC Count (×10⁶/μL)", 0, 10, 5)
    hemoglobin = st.sidebar.slider("Hemoglobin (g/dL)", 0, 20, 12)
    platelet_count = st.sidebar.slider("Platelet Count (×10³/μL)", 50, 500, 150)
    lymphoblast_percentage = st.sidebar.slider("Lymphoblast Percentage", 0, 100, 50)
    chromosomal_abnormalities = st.sidebar.selectbox("Chromosomal Abnormalities", [0, 1])

    # Create the input data frame
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
        
        # Create a patient-friendly survival graph
        plt.figure(figsize=(10, 6))
        plt.step(np.arange(len(survival_probs)), survival_probs, where="post", color='green', label='Survival Function')
        plt.axhline(y=0.5, color='orange', linestyle='--', label='50% Survival Probability')
        plt.axvline(x=12, color='blue', linestyle='--', label='12 Months Milestone')
        plt.axvline(x=24, color='purple', linestyle='--', label='24 Months Milestone')
        plt.axvline(x=36, color='red', linestyle='--', label='36 Months Milestone')
        plt.xlabel("Time (Months)", fontsize=12)
        plt.ylabel("Survival Probability", fontsize=12)
        plt.title("🩺 Predicted Survival Function", fontsize=16)
        plt.legend()
        st.pyplot(plt)

        # Display survival probabilities as a table
        survival_df = pd.DataFrame({'Time (Months)': np.arange(len(survival_probs)), 'Survival Probability': survival_probs})
        st.dataframe(survival_df)

        # Highlight key survival probabilities
        st.write(f"**Survival Probability at 12 months:** {survival_probs[12] if 12 < len(survival_probs) else 'N/A'}")
        st.write(f"**Survival Probability at 24 months:** {survival_probs[24] if 24 < len(survival_probs) else 'N/A'}")
        st.write(f"**Survival Probability at 36 months:** {survival_probs[36] if 36 < len(survival_probs) else 'N/A'}")

elif option == "ALL Prediction":
    # Input form for ALL prediction
    st.subheader("📸 Upload a Blood Smear Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, caption="📍 Uploaded Image", use_column_width=True)
        
        preprocessed_image = preprocess_image(uploaded_image)
        probabilities = all_model.predict(preprocessed_image)[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        predicted_probability = probabilities[predicted_index] * 100

        # Display prediction results
        st.subheader("🔬 Prediction Results")
        st.metric("🔍 Likelihood of ALL", f"{predicted_probability:.2f} %")
        st.metric("🩸 Predicted Stage", predicted_class)
        st.write(f"**📌 Recommendations:** {recommendations[predicted_class]}")

        # Display probabilities as a bar chart
        probabilities_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": probabilities * 100
        })
        st.bar_chart(probabilities_df.set_index("Class"))

        # Pie chart visualization
        st.subheader("📊 Visualization")
        predicted_percent = predicted_probability
        other_percent = 100 - predicted_percent
        fig, ax = plt.subplots()
        ax.pie([predicted_percent, other_percent], labels=[predicted_class, "Other"], autopct="%1.1f%%", startangle=140, colors=["#76c7b7", "#e0e0e0"])
        ax.axis("equal")
        st.pyplot(fig)

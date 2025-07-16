from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib
import os

app = Flask(__name__)

# Load models
survival_model = joblib.load("models/random_survival_forest_model.joblib")
all_model = load_model("models/resnet50.h5")

class_names = ['Benign', 'Pre', 'Pro', 'Early']
recommendations = {
    "Benign": "No signs of ALL detected. Maintain regular health check-ups and a healthy lifestyle.",
    "Pre": "Early indications of ALL. Consult a hematologist for further testing and monitoring.",
    "Pro": "Progressive stage of ALL. Seek immediate medical attention for diagnosis and treatment planning.",
    "Early": "Early-stage ALL detected. Initiate treatment as advised by an oncologist."
}

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def plot_survival(survival_probs):
    plt.figure(figsize=(10, 6))
    plt.step(np.arange(len(survival_probs)), survival_probs, where="post", color='green', label='Survival Function')
    plt.axhline(y=0.5, color='orange', linestyle='--', label='50% Survival Probability')
    for milestone, color in zip([12, 24, 36], ['blue', 'purple', 'red']):
        if len(survival_probs) > milestone:
            plt.axvline(x=milestone, color=color, linestyle='--', label=f'{milestone} Months Milestone')

    plt.xlabel("Time (Months)")
    plt.ylabel("Survival Probability")
    plt.title("Predicted Survival Function")
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/survival', methods=['POST'])
def survival():
    try:
        data = {
            'Age': int(request.form['age']),
            'WBC_Count': float(request.form['wbc']),
            'RBC_Count': float(request.form['rbc']),
            'Hemoglobin': float(request.form['hgb']),
            'Platelet_Count': float(request.form['platelets']),
            'Lymphoblast_Percentage': float(request.form['lympho']),
            'Chromosomal_Abnormalities': int(request.form['chrom'])
        }

        df = pd.DataFrame([data])
        survival_probs = survival_model.predict_survival_function(df, return_array=True)[0]
        chart = plot_survival(survival_probs)

        return render_template('result.html',
                               survival=True,
                               chart=chart,
                               survival_12=round(survival_probs[12]*100, 2) if len(survival_probs) > 12 else 'N/A',
                               survival_24=round(survival_probs[24]*100, 2) if len(survival_probs) > 24 else 'N/A',
                               survival_36=round(survival_probs[36]*100, 2) if len(survival_probs) > 36 else 'N/A')

    except Exception as e:
        return f"Error during survival prediction: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file:
            img_array = preprocess_image(file)
            probabilities = all_model.predict(img_array)[0]
            index = np.argmax(probabilities)
            predicted_class = class_names[index]
            predicted_prob = round(probabilities[index] * 100, 2)

            return render_template('result.html',
                                   survival=False,
                                   prediction=predicted_class,
                                   probability=predicted_prob,
                                   recommendation=recommendations[predicted_class],
                                   probs=zip(class_names, (probabilities * 100).round(2)))
        return redirect(url_for('index'))
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

# =========================
# LOAD MODEL
# =========================

burnout_model = tf.keras.models.load_model(
    "burnout_model.keras"
)

mental_model = tf.keras.models.load_model(
    "mental_health_model.keras"
)

# =========================
# LOAD SCALER
# =========================

scaler = joblib.load(
    "scaler.save"
)

# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "message": "Mental Health API Running"
    }

# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([[
        data["stress_level"],
        data["anxiety_score"],
        data["depression_score"],
        data["exam_pressure"],
        data["sleep_hours"],
        data["study_hours_per_day"],
        data["financial_stress"],
        data["family_expectation"],
        data["social_support"],
        data["physical_activity"]
    ]])

    scaled_data = scaler.transform(
        input_data
    )

    # Burnout prediction
    burnout_pred = burnout_model.predict(
        scaled_data
    )

    burnout_class = np.argmax(
        burnout_pred
    )

    # Mental prediction
    mental_pred = mental_model.predict(
        scaled_data
    )

    mental_class = np.argmax(
        mental_pred
    )

    burnout_labels = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    mental_labels = {
        0: "Buruk",
        1: "Sedang",
        2: "Baik"
    }

    return {

        "burnout_prediction":
            burnout_labels[burnout_class],

        "mental_health_prediction":
            mental_labels[mental_class]
    }

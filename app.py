import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load model
model = tf.keras.models.load_model("best_model.keras")

# Load scaler
scaler = joblib.load("scaler.save")

# Judul
st.title("Mental Health Prediction App")

st.write("Masukkan data berikut:")

# Input user
feature1 = st.number_input("Study Hours", 0.0, 24.0)
feature2 = st.number_input("Sleep Hours", 0.0, 24.0)
feature3 = st.number_input("Stress Level", 0.0, 10.0)
feature4 = st.number_input("Work Pressure", 0.0, 10.0)

# Tombol prediksi
if st.button("Predict"):

    # Buat array
    data = np.array([[feature1, feature2, feature3, feature4]])

    # Scaling
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)

    # Hasil
    st.subheader("Hasil Prediksi")

    st.write(prediction)

    if prediction[0][0] > 0.5:
        st.error("Burnout Tinggi")
    else:
        st.success("Burnout Rendah")

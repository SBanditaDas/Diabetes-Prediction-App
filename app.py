import streamlit as st
import joblib
import numpy as np

model = joblib.load('diabetes_model.pkl')

st.title("🩺 Diabetes Prediction App")
st.markdown("Enter your health metrics below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🟢 Not Diabetic" if prediction[0] == 0 else "🔴 Diabetic"
    st.subheader(f"Prediction Result: {result}")
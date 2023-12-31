import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.joblib")  # Replace with the path to your model file

# Create a Streamlit app
st.title("Stress Level Prediction App")

# Feature names
feature_names = [
    "anxiety_level",
    "self_esteem 5=มากที่สุด 1=น้อยที่สุด",
    "mental_health_history",
    "depression คุณมีภาวะ depression กี่ครั้งต่อสัปดาห์",
    "headache คุณปวดหัวกี่ครั้งต่อสัปดาห์",
    "blood_pressure คุณมีปัญหาความดันโลหิตกี่ครั้งต่อสัปดาห์",
    "sleep_quality ระดับคุณภาพ : 5 = ดีมาก , 4 = ดี , 3 = ปานกลาง , 2 = พอใช้, 1 = ปรับปรุง",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance คุณจะให้คะแนนผลการเรียนของคุณอย่างไร",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns 5=มากที่สุด 1=น้อยที่สุด",
    "social_support 5=มากที่สุด 1=น้อยที่สุด",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
    "stress_level คุณจะให้คะแนนระดับความเครียดของคุณอย่างไร  5=มากที่สุด 1=น้อยที่สุด"
]

# Create input fields for the user to provide input data
st.write("Enter values for prediction:")
x_new = []
for i in range(20):  # Assuming you have 20 features in your data
    x_new.append(st.number_input(f"{feature_names[i]}", step=1, format="%d"))

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    x_new = np.array(x_new).reshape(1, -1)  # Reshape the input to match the model's expectations
    y_pred_new = model.predict(x_new)
    st.write(f"Predicted Stress Level: {y_pred_new[0]}")

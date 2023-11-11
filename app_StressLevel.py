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
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
    "stress_level"
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

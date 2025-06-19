import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Title of the app
st.title("🚢 Titanic Survival Predictor")

# User input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Passenger Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encode categorical variables
sex = 1 if sex == "Male" else 0
embarked = {"S": 2, "C": 0, "Q": 1}[embarked]

# Prediction
if st.button("Predict Survival"):
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(f"Prediction: {result}")

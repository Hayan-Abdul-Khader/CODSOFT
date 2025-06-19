import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Species Prediction")
st.write("Input the flower measurements below:")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Prediction
if st.button("Predict Species"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)[0]
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = species_map[prediction]
    st.success(f"Predicted Species: 🌼 {predicted_species.capitalize()}")

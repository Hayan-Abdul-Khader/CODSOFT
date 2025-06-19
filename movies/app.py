import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and expected feature names
model, model_features = joblib.load("movie_rating_model.pkl")

# App UI
st.title("🎬 Movie Rating Predictor")
st.write("Enter movie details below to predict its IMDb rating.")

# Inputs
year = st.number_input("Release Year", min_value=1950, max_value=2030, value=2020)
duration = st.number_input("Duration (in minutes)", min_value=30, max_value=300, value=120)
votes = st.number_input("Number of Votes", min_value=0, value=1000)

# Genre options (make sure these include all top genres from training)
genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller', 'Adventure', 'Crime', 'Animation', 'Biography', 'Horror', 'Documentary', 'Musical', 'Other']
selected_genre = st.selectbox("Genre", genres)

# Create genre dummy variables
genre_input = {f"Genre_{g}": 0 for g in genres}
genre_input[f"Genre_{selected_genre}"] = 1

# Base input features
features = {
    "Year": year,
    "Duration": duration,
    "Votes": votes
}
features.update(genre_input)

# Create input DataFrame
input_df = pd.DataFrame([features])

# Ensure input matches model features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model_features]

# Predict
if st.button("Predict Rating"):
    prediction = model.predict(input_df)[0]
    st.success(f"⭐ Predicted IMDb Rating: {prediction:.2f}")

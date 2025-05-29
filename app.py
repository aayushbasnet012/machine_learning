import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_injury_model.pkl")

st.title("⚽ Football Injury Risk Predictor")
st.write("Predict whether a player is likely to get injured in the next match.")

# Input fields
minutes = st.number_input("Minutes Played in Last 30 Days",
                          min_value=0, max_value=3000, value=300)
matches = st.number_input("Matches Played in Last 14 Days",
                          min_value=0, max_value=7, value=2)
sprints = st.number_input("Sprints in Last 5 Matches",
                          min_value=0, max_value=500, value=100)
duels = st.number_input("Duels in Last 5 Matches",
                        min_value=0, max_value=500, value=50)
position = st.selectbox(
    "Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
injuries = st.number_input("Previous Injury Count",
                           min_value=0, max_value=20, value=1)
age = st.number_input("Age", min_value=15, max_value=45, value=25)

# Encode position (you must match the encoding used during training)
position_map = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3
}
position_encoded = position_map[position]

# Predict
if st.button("Predict Injury Risk"):
    input_data = np.array(
        [[minutes, matches, sprints, duels, position_encoded, injuries, age]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Injury (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ Low Risk of Injury (Confidence: {1 - prob:.2%})")

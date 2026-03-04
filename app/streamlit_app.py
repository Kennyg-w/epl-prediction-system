import streamlit as st
import pandas as pd

from src.artifacts import load_artifact

st.set_page_config(page_title="EPL Win Predictor", layout="centered")

st.title("⚽ EPL Match Win Predictor")

model, threshold, features = load_artifact()

st.subheader("Enter match features")

opp_code = st.number_input("Opponent code (opp_code)", min_value=0, max_value=40, value=10)
gf_rolling = st.number_input("Rolling goals for (gf_rolling)", value=1.5)
ga_rolling = st.number_input("Rolling goals against (ga_rolling)", value=1.2)
day_code = st.number_input("Day code (day_code: 0-6)", min_value=0, max_value=6, value=5)
venue_code = st.number_input("Venue code (venue_code: 0 away, 1 home)", min_value=0, max_value=1, value=1)

input_dict = {
    "opp_code": opp_code,
    "gf_rolling": gf_rolling,
    "ga_rolling": ga_rolling,
    "day_code": day_code,
    "venue_code": venue_code
}

# Ensure correct feature order
if features:
    X = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
else:
    X = pd.DataFrame([input_dict])

prob = model.predict_proba(X)[0, 1]
pred = int(prob >= threshold)

st.markdown(f"### ✅ Win Probability: **{prob:.2%}**")
st.markdown(f"### 🎯 Model Threshold: **{threshold:.3f}**")
st.markdown(f"### 🧠 Prediction: **{'WIN' if pred == 1 else 'NOT WIN'}**")
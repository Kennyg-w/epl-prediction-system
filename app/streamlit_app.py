import streamlit as st
import pandas as pd

from src.artifacts import load_artifact

st.set_page_config(page_title="EPL Win Predictor", layout="centered")
st.title("⚽ EPL Match Win Predictor")

# Choose model type
mode = st.selectbox("Model mode", ["raw", "calibrated"], index=0)

model, threshold, features = load_artifact(mode)

st.subheader("Enter match inputs")

opp_code = st.number_input("Opponent code (opp_code)", min_value=0, max_value=40, value=10)
gf_rolling = st.number_input("Rolling goals for (gf_rolling)", value=1.5)
ga_rolling = st.number_input("Rolling goals against (ga_rolling)", value=1.2)
day_code = st.number_input("Day code (day_code: 0-6)", min_value=0, max_value=6, value=5)
venue_code = st.number_input("Venue code (venue_code: 0 away, 1 home)", min_value=0, max_value=1, value=1)

# --- Build required engineered features ---
gd_rolling = gf_rolling - ga_rolling

# We DO NOT have opponent rolling stats from user input, so we default safely.
# (In a real app you'd pick teams and look up opponent stats from a dataset)
opp_gf_rolling = 0.0
opp_ga_rolling = 0.0
opp_gd_rolling = 0.0
form_rolling = 0.0
opp_form_rolling = 0.0

team_avg_points = 0.0
team_avg_gd = 0.0
opp_team_avg_points = 0.0
opp_team_avg_gd = 0.0

rest_days = 7.0
opp_rest_days = 7.0

# diffs
form_diff = form_rolling - opp_form_rolling
gd_diff = gd_rolling - opp_gd_rolling
avg_points_diff = team_avg_points - opp_team_avg_points
avg_gd_diff = team_avg_gd - opp_team_avg_gd
rest_diff = rest_days - opp_rest_days
home_gd_diff = venue_code * gd_diff

# Input dict includes EVERYTHING the model might expect
input_dict = {
    "venue_code": float(venue_code),
    "day_code": float(day_code),
    "opp_code": float(opp_code),

    "gf_rolling": float(gf_rolling),
    "ga_rolling": float(ga_rolling),
    "gd_rolling": float(gd_rolling),
    "form_rolling": float(form_rolling),

    "team_avg_points": float(team_avg_points),
    "team_avg_gd": float(team_avg_gd),
    "rest_days": float(rest_days),

    "opp_gf_rolling": float(opp_gf_rolling),
    "opp_ga_rolling": float(opp_ga_rolling),
    "opp_gd_rolling": float(opp_gd_rolling),
    "opp_form_rolling": float(opp_form_rolling),

    "opp_team_avg_points": float(opp_team_avg_points),
    "opp_team_avg_gd": float(opp_team_avg_gd),
    "opp_rest_days": float(opp_rest_days),

    "form_diff": float(form_diff),
    "gd_diff": float(gd_diff),
    "home_gd_diff": float(home_gd_diff),
    "avg_points_diff": float(avg_points_diff),
    "avg_gd_diff": float(avg_gd_diff),
    "rest_diff": float(rest_diff),
}

# Ensure correct feature order for model
if features:
    missing = [f for f in features if f not in input_dict]
    if missing:
        st.error(f"Missing required features: {missing}")
        st.stop()

    X = pd.DataFrame([[input_dict[f] for f in features]], columns=features)
else:
    X = pd.DataFrame([input_dict])

prob = model.predict_proba(X)[0, 1]
pred = int(prob >= threshold)

st.markdown(f"### ✅ Win Probability: **{prob:.2%}**")
st.markdown(f"### 🎯 Threshold: **{threshold:.3f}**")
st.markdown(f"### 🧠 Prediction: **{'WIN' if pred == 1 else 'NOT WIN'}**")

with st.expander("Show model inputs"):
    st.dataframe(X)
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="AI Career Guidance System", layout="wide")

st.title("🎓 AI-Powered Career Guidance System")
st.write("Get personalized career recommendations based on your skills and interests.")

# ------------------ LOAD MODEL ------------------ #
model = joblib.load("career_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ USER INPUT ------------------ #
st.subheader("📝 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours working per day", 0, 12, 5)
    logical = st.slider("Logical quotient rating (1-5)", 1, 5, 3)
    hackathons = st.slider("Hackathons participated", 0, 10, 1)
    coding = st.slider("Coding skills rating (1-5)", 1, 5, 3)
    public = st.slider("Public speaking points (1-5)", 1, 5, 3)
    self_learning = st.selectbox("Self-learning capability?", [0, 1])

with col2:
    certifications = st.slider("Certifications completed", 0, 10, 1)
    workshops = st.slider("Workshops attended", 0, 10, 1)
    reading = st.slider("Reading & writing skills (1-5)", 1, 5, 3)
    memory = st.slider("Memory capability score (1-5)", 1, 5, 3)
    management = st.selectbox("Management (1) or Technical (0)?", [0, 1])
    teamwork = st.selectbox("Worked in teams ever?", [0, 1])
    introvert = st.selectbox("Introvert?", [0, 1])

# ------------------ PREDICTION ------------------ #
if st.button("🔮 Predict Career"):

    input_dict = {
        'Hours working per day': hours,
        'Logical quotient rating': logical,
        'hackathons': hackathons,
        'coding skills rating': coding,
        'public speaking points': public,
        'self-learning capability?': self_learning,
        'certifications': certifications,
        'workshops': workshops,
        'reading and writing skills': reading,
        'memory capability score': memory,
        'Management or Technical': management,
        'worked in teams ever?': teamwork,
        'Introvert': introvert
    }

    input_data = pd.DataFrame([input_dict])

    # Ensure correct feature order
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Convert encoded prediction back
    prediction = label_encoder.inverse_transform(prediction)

    # Probability dataframe
    prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)

    # Sort Top 3
    top3 = prob_df.T.sort_values(by=0, ascending=False).head(3)

    # ------------------ OUTPUT ------------------ #
    st.success(f"🥇 Top Career Recommendation: {top3.index[0]}")

    st.subheader("🏆 Top 3 Career Predictions")
    for i, (career, prob) in enumerate(top3.itertuples()):
        st.write(f"{i+1}. {career} — {round(prob*100, 2)}%")

    # Probability Chart
    st.subheader("📊 Career Prediction Probability Distribution")
    st.bar_chart(top3)

    # ------------------ Readiness Score ------------------ #
    readiness_score = int((coding + logical + reading + memory) / 20 * 100)

    st.subheader("📈 Career Readiness Score")
    st.progress(readiness_score)
    st.write(f"Overall Readiness: {readiness_score}%")

    # ------------------ Personalized Action Plan ------------------ #
    st.subheader("📌 Personalized Action Plan")

    if coding < 3:
        st.write("• Improve coding skills by building real-world projects.")
    if public < 3:
        st.write("• Practice communication and public speaking.")
    if logical < 3:
        st.write("• Work on logical reasoning and problem solving.")
    if certifications < 2:
        st.write("• Complete relevant certifications in your field.")
    if teamwork == 0:
        st.write("• Participate in team-based activities.")

    st.write("• Stay consistent and keep upgrading your skills 🚀")

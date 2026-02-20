import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("career_prediction_model.pkl")

st.set_page_config(page_title="AI Career Prediction", layout="centered")

st.title("🎓 AI-Based Career Prediction System")
st.write("Enter your skill details below to get career recommendation.")

# -------- USER INPUTS -------- #

hours = st.slider("Hours working per day", 0, 12, 6)
logical = st.slider("Logical quotient rating (0-5)", 0, 5, 3)
hackathons = st.slider("Hackathons participated", 0, 10, 1)
coding = st.slider("Coding skills rating (0-5)", 0, 5, 3)
public = st.slider("Public speaking points (0-5)", 0, 5, 2)
self_learning = st.selectbox("Self-learning capability?", [0, 1])
certifications = st.slider("Certifications count", 0, 10, 1)
workshops = st.slider("Workshops attended", 0, 10, 1)
reading = st.slider("Reading & Writing Skills (0-5)", 0, 5, 3)
memory = st.slider("Memory capability score", 0, 5, 3)
management = st.selectbox("Management or Technical (0=Management, 1=Technical)", [0, 1])
teamwork = st.selectbox("Worked in teams ever?", [0, 1])
introvert = st.selectbox("Introvert?", [0, 1])

# -------- PREDICTION -------- #

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

    # Fix feature mismatch
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    st.success(f"🎯 Recommended Career Path: {prediction[0]}")

    # Probability Graph
    st.subheader("📊 Career Prediction Probability")
    prob_df = pd.DataFrame(probabilities, columns=model.classes_)
    st.bar_chart(prob_df.T)

    # Readiness Score
    readiness_score = int((coding + logical + reading + memory) / 20 * 100)
    st.subheader("📈 Career Readiness Score")
    st.progress(readiness_score)
    st.write(f"Overall Readiness: {readiness_score}%")

    # Action Plan
    st.subheader("📌 Personalized Action Plan")

    if coding < 3:
        st.write("• Improve coding skills by building projects.")
    if public < 3:
        st.write("• Practice communication and public speaking.")
    if logical < 3:
        st.write("• Improve logical thinking and problem solving.")
    if certifications < 2:
        st.write("• Complete relevant certifications.")
    if teamwork == 0:
        st.write("• Participate in team-based activities.")

    st.write("• Stay consistent and keep upgrading your skills.")

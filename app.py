import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Career Navigator", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
.result-box {
    padding: 15px;
    border-radius: 12px;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    margin: 10px 0;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown("<h1>🚀 AI Career Navigator</h1>", unsafe_allow_html=True)
st.markdown("<h3>Discover your ideal career path using AI</h3>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("career_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- INPUT SECTION ---------------- #
st.markdown("## 🧠 Skill Assessment")

col1, col2 = st.columns(2)

with col1:
    hours = st.slider("Hours Working Per Day", 0, 12, 5)
    logical = st.slider("Logical Thinking (1-5)", 1, 5, 3)
    coding = st.slider("Coding Skills (1-5)", 1, 5, 3)
    public = st.slider("Public Speaking (1-5)", 1, 5, 3)
    reading = st.slider("Reading & Writing (1-5)", 1, 5, 3)
    memory = st.slider("Memory Score (1-5)", 1, 5, 3)

with col2:
    hackathons = st.slider("Hackathons Participated", 0, 10, 1)
    certifications = st.slider("Certifications Completed", 0, 10, 1)
    workshops = st.slider("Workshops Attended", 0, 10, 1)
    self_learning = st.selectbox("Self-Learning Capability", ["No", "Yes"])
    teamwork = st.selectbox("Worked in Teams", ["No", "Yes"])
    introvert = st.selectbox("Introvert", ["No", "Yes"])
    management = st.selectbox("Preferred Area", ["Technical", "Management"])

# Convert categorical to numeric
self_learning = 1 if self_learning == "Yes" else 0
teamwork = 1 if teamwork == "Yes" else 0
introvert = 1 if introvert == "Yes" else 0
management = 1 if management == "Management" else 0

# ---------------- PREDICTION ---------------- #
if st.button("🔮 Predict My Career"):

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
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    prediction = label_encoder.inverse_transform(prediction)

    prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)
    top3 = prob_df.T.sort_values(by=0, ascending=False).head(3)

    st.markdown("## 🎯 Your Career Matches")

    for i, (career, prob) in enumerate(top3.itertuples()):
        st.markdown(
            f"<div class='result-box'>{i+1}. {career} — {round(prob*100,2)}%</div>",
            unsafe_allow_html=True
        )

    st.markdown("## 📊 Probability Breakdown")
    st.bar_chart(top3)

    # ---------------- READINESS SCORE ---------------- #
    readiness_score = int((coding + logical + reading + memory) / 20 * 100)

    st.markdown("## 📈 Career Readiness Score")
    st.progress(readiness_score)
    st.write(f"### {readiness_score}% Ready")

    # ---------------- ACTION PLAN ---------------- #
    st.markdown("## 📌 Personalized Growth Plan")

    if coding < 3:
        st.write("• Improve coding by building real projects.")
    if public < 3:
        st.write("• Practice communication and presentation skills.")
    if logical < 3:
        st.write("• Work on problem-solving exercises.")
    if certifications < 2:
        st.write("• Complete certifications in your target field.")
    if teamwork == 0:
        st.write("• Participate in group projects and hackathons.")

    st.write("🚀 Keep learning and stay consistent!")

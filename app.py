import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI Career Advisor", page_icon="🎯", layout="wide")

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .title {
        font-size:40px;
        font-weight:700;
        text-align:center;
        color:#1f4e79;
    }
    .subtitle {
        text-align:center;
        font-size:18px;
        color:gray;
    }
    .card {
        background-color:white;
        padding:20px;
        border-radius:15px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("career_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ---------------- HEADER ---------------- #
st.markdown('<p class="title">🎓 AI-Powered Career Guidance System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Smart Career Prediction + Personalized Action Plan</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.header("📊 Enter Your Skill Details")

hours = st.sidebar.slider("Hours working per day", 0, 12, 6)
logical = st.sidebar.slider("Logical quotient rating", 0, 5, 3)
hackathons = st.sidebar.slider("Hackathons", 0, 10, 1)
coding = st.sidebar.slider("Coding skills rating", 0, 5, 3)
public = st.sidebar.slider("Public speaking", 0, 5, 2)
self_learning = st.sidebar.selectbox("Self-learning capability?", [0, 1])
certifications = st.sidebar.slider("Certifications", 0, 10, 1)
workshops = st.sidebar.slider("Workshops", 0, 10, 1)
reading = st.sidebar.slider("Reading & Writing Skills", 0, 5, 3)
memory = st.sidebar.slider("Memory capability score", 0, 5, 3)
management = st.sidebar.selectbox("Management (0) / Technical (1)", [0, 1])
teamwork = st.sidebar.selectbox("Worked in teams?", [0, 1])
introvert = st.sidebar.selectbox("Introvert?", [0, 1])

st.markdown("## 🚀 Get Career Recommendation")

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
    prediction_label = label_encoder.inverse_transform(prediction)

    probabilities = model.predict_proba(input_data)

    # ---------------- OUTPUT SECTION ---------------- #
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Recommended Career")
        st.success(prediction_label[0])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        readiness_score = int((coding + logical + reading + memory) / 20 * 100)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Career Readiness Score")
        st.progress(readiness_score)
        st.write(f"Overall Readiness: {readiness_score}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PROBABILITY CHART ---------------- #
    st.markdown("### 📊 Career Prediction Probability")
    prob_df = pd.DataFrame(probabilities, columns=label_encoder.classes_)
    st.bar_chart(prob_df.T)

    # ---------------- ACTION PLAN ---------------- #
    st.markdown("### 📌 Personalized Action Plan")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if coding < 3:
        st.write("• Improve coding skills by building real-world projects.")
    if public < 3:
        st.write("• Practice communication and public speaking.")
    if logical < 3:
        st.write("• Improve logical thinking with problem-solving practice.")
    if certifications < 2:
        st.write("• Complete relevant certifications in your field.")
    if teamwork == 0:
        st.write("• Participate in team-based activities or hackathons.")

    st.write("• Stay consistent and keep upgrading your skills.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("© 2026 AI Career Guidance System | Innovation Marathon Project")

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

    # 🔥 IMPORTANT LINE (Fix feature mismatch)
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    st.success(f"🎯 Recommended Career Path: {prediction[0]}")

    # -------- WOW FACTOR: Probability Display -------- #
    st.subheader("📊 Career Prediction Probability")
    prob_df = pd.DataFrame(probabilities, columns=model.classes_)
    st.bar_chart(prob_df.T)

    # -------- WOW FACTOR: Skill Readiness Score -------- #
    readiness_score = int((coding + logical + reading + memory) / 20 * 100)
    st.subheader("📈 Career Readiness Score")
    st.progress(readiness_score)
    st.write(f"Overall Readiness: {readiness_score}%")

    # -------- Personalized Action Plan -------- #
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

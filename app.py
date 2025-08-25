import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
)

# ------------------ CUSTOM STYLING ------------------
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: #333;
            color: white;
            text-align: center;
            padding: 8px;
            font-size: 14px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ APP TITLE ------------------
st.image("assets/logo.png", width=100)  # optional logo
st.title("🎓 Student Result Predictor (AI Model)")
st.write("This simple AI model predicts whether a student will **Pass** or **Fail** based on their marks.")

# ------------------ SAMPLE DATA ------------------
data = {
    "Maths": [40, 55, 65, 75, 30, 90, 50, 85, 45, 60],
    "Physics": [35, 60, 70, 80, 25, 95, 55, 88, 40, 65],
    "Chemistry": [30, 58, 68, 78, 28, 92, 48, 82, 38, 62],
    "Result": ["Fail", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Pass", "Fail", "Pass"]
}
df = pd.DataFrame(data)

# ------------------ TRAIN MODEL ------------------
X = df[["Maths", "Physics", "Chemistry"]]
y = df["Result"]
model = LogisticRegression()
model.fit(X, y)

# ------------------ USER INPUT ------------------
st.subheader("📥 Enter Student Marks")

maths = st.number_input("Maths Marks", min_value=0, max_value=100, value=50, step=1)
physics = st.number_input("Physics Marks", min_value=0, max_value=100, value=50, step=1)
chemistry = st.number_input("Chemistry Marks", min_value=0, max_value=100, value=50, step=1)

# ------------------ PREDICTION ------------------
if st.button("🔮 Predict Result"):
    prediction = model.predict([[maths, physics, chemistry]])[0]
    if prediction == "Pass":
        st.success("✅ Prediction: The student will PASS 🎉")
    else:
        st.error("❌ Prediction: The student will FAIL 📉")

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">Developed by EngHarBot | Powered by AI 🤖</div>', unsafe_allow_html=True)

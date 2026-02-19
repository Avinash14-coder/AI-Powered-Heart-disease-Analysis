import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from google import genai
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
import os
from dotenv import load_dotenv



# --- 1. SETUP & THEME ---
st.set_page_config(page_title="HeartAI | Diagnostic Suite", layout="wide")

# Custom CSS for Red & White Theme
st.markdown("""
    <style>
    :root { --primary-color: #d32f2f; }
    .stApp { background-color: white; }
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .stTextInput>div>div>input { border-color: #d32f2f; }
    h1, h2, h3 { color: #d32f2f !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #d32f2f !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. PDF CLASS ---
class HeartReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 20)
        self.set_text_color(211, 47, 47) 
        self.cell(0, 10, "HEART HEALTH DIAGNOSTIC REPORT", ln=True, align="C")
        self.set_font("helvetica", "I", 10)
        self.set_text_color(100)
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 10, f"Generated on: {date_str}", ln=True, align="C")
        self.line(10, 32, 200, 32)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()} | Clinical AI Analysis", align="C")

# --- 3. CONFIGURATION & UTILITIES ---
 # Replace with your actual key
 # Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=GENAI_API_KEY)

@st.cache_resource
def load_model():
    return joblib.load("best_heart_model.pkl")

def clean_text(text):
    replacements = {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "\u2022": "*", "•": "*"}
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf(patient_data, prediction_label, risk_prob, gemini_text, fig_bar, fig_pie):
    pdf = HeartReport()
    pdf.add_page()
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 12, f"Assessment: {prediction_label}", ln=True, fill=True, border=1, align="C")
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Patient Vital Parameters:", ln=True)
    pdf.set_font("helvetica", "", 10)
    for key, value in patient_data.items():
        pdf.cell(0, 7, f"* {key}: {value}", ln=True)
    
    buf_bar, buf_pie = BytesIO(), BytesIO()
    fig_bar.savefig(buf_bar, format='png', bbox_inches='tight')
    fig_pie.savefig(buf_pie, format='png', bbox_inches='tight')
    buf_bar.seek(0); buf_pie.seek(0)
    
    pdf.image(buf_bar, x=10, y=pdf.get_y()+5, w=90)
    pdf.image(buf_pie, x=110, y=pdf.get_y()+5, w=80)
    
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 12, "AI Clinical Insights:", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 8, txt=clean_text(gemini_text.replace("|", "")))
    return pdf.output()

# --- 4. MAIN INTERFACE ---
st.title("🫀 HeartAI: Clinical Intelligence System")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Diagnostic Form", "💬 Health Chatbot", "ℹ️ About Project", "👨‍💻 Developer"])

# --- TAB 1: FORM ---
with tab1:
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Input Vitals")
        age = st.slider("Age", 1, 100, 39)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain", ["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
        trestbps = st.number_input("Resting BP", value=122)
        chol = st.number_input("Cholesterol", value=190)
        fbs = st.selectbox("FBS > 120", ["False", "True"])
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
        thalch = st.slider("Max Heart Rate", 60, 220, 178)
        exang = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 0.0)
        slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Major Vessels", ["0", "1", "2", "3"])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversable defect"])

    with col_b:
        st.subheader("Analysis Results")
        input_dict = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
                      'restecg': restecg, 'thalch': thalch, 'exang': exang, 'oldpeak': oldpeak,
                      'slope': slope, 'ca': ca, 'thal': thal}
        input_df = pd.DataFrame([input_dict])

        if st.button("Run Diagnostic Analysis"):
            pipeline = load_model()
            rf_model = pipeline.named_steps['model']
            preprocessor = pipeline.named_steps['preprocessor']
            
            prob = pipeline.predict_proba(input_df)[:, 1][0]
            label = "Heart Disease Detected" if prob >= 0.6 else "No Heart Disease Detected"
            
            st.metric("Risk Probability", f"{prob:.2%}")
            
            # --- FIXED SHAP LOGIC ---
            X_tx = preprocessor.transform(input_df)
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(X_tx)
            
            # Check for different SHAP output formats (List vs Array)
            if isinstance(shap_vals, list):
                # Class 1 (Positive) is usually index 1
                current_sv = shap_vals[1].ravel()
            elif len(shap_vals.shape) == 3:
                # Shape is (samples, features, classes)
                current_sv = shap_vals[0, :, 1]
            else:
                current_sv = shap_vals.ravel()

            fn = preprocessor.get_feature_names_out()
            # Safety check for number of features
            num_show = min(10, len(current_sv))
            idx = np.argsort(np.abs(current_sv))[-num_show:]

            c1, c2 = st.columns(2)
            with c1:
                fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                plt.barh(range(len(idx)), current_sv[idx], color='#d32f2f')
                plt.yticks(range(len(idx)), [fn[i] for i in idx])
                plt.title("Top Risk Contributors")
                st.pyplot(fig_bar)
            with c2:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                pos = np.sum(current_sv[current_sv > 0])
                neg = np.abs(np.sum(current_sv[current_sv < 0]))
                ax_pie.pie([pos, neg], labels=['Risk Factors', 'Protective'], 
                           colors=['#d32f2f', '#4caf50'], autopct='%1.1f%%')
                plt.title("Factor Balance")
                st.pyplot(fig_pie)

            # Gemini Analysis
            with st.spinner("AI Analysis in progress..."):
                prompt = f"Analyze heart health: {label} with {prob:.2%} risk. Provide Risk Table, Diet Plan, and Priority Medical Actions."
                res = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                st.markdown(res.text)
            
            pdf_bytes = generate_pdf(input_dict, label, prob, res.text, fig_bar, fig_pie)
            st.download_button("📥 Download Report", data=bytes(pdf_bytes), file_name="Heart_Report.pdf")

# --- TAB 2: CHATBOT ---
with tab2:
    st.subheader("💬 HeartHealth AI Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if chat_input := st.chat_input("Ask about heart health or your results..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)

        with st.chat_message("assistant"):
            chat_res = client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=f"You are a medical assistant specialized in cardiology. Answer: {chat_input}"
            )
            st.markdown(chat_res.text)
            st.session_state.messages.append({"role": "assistant", "content": chat_res.text})

# --- TAB 3: ABOUT ---
with tab3:
    st.header("About HeartAI")
    st.write("""
    **HeartAI** is an advanced diagnostic support system that combines **Machine Learning (Random Forest)** with **Generative AI (Google Gemini)**.
    
    ### Technology Stack:
    - **Random Forest Classifier:** Predicts the probability of heart disease based on clinical data.
    - **SHAP (SHapley Additive exPlanations):** Provides transparency by showing exactly which biomarkers impacted the score.
    - **Google gemini-3-flash-preview:** Generates personalized clinical insights and lifestyle recommendations.
    - **FPDF:** Generates professional-grade medical reports on the fly.
    """)
    st.image("https://images.unsplash.com/photo-1505751172107-573225a94371?auto=format&fit=crop&q=80&w=1000")

# --- TAB 4: DEVELOPER ---
with tab4:
    st.header("Developer Profile")
    dev_col1, dev_col2 = st.columns([1, 2])
    with dev_col1:
        st.image("Portfolio photo.png", width=200) 
    with dev_col2:
        st.subheader("Avinash Pawar")
        st.write("AI Engineer & Data Scientist")
        st.markdown("""
        - **LinkedIn:** [linkedin.com/in/Avinash](https://www.linkedin.com/in/avinash-pawar-0a19b5347?utm_source=share_via&utm_content=profile&utm_medium=member_android)
        - **GitHub:** [github.com/Avinash14-coder](https://github.com/Avinash14-coder)
        
        Passionate about bridging the gap between AI research and clinical application.
        """)
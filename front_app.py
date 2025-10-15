import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from ml_pipeline import load_data, preprocess_data, load_model

# User-friendly column names
DISPLAY_NAMES = {
    'person_age': 'Age',
    'person_home_ownership': 'Home Ownership',
    'loan_intent': 'Loan Intent',
    'loan_amnt': 'Loan Amount',
    'loan_percent_income': 'Percent Income',
    'cb_person_cred_hist_length': 'Credit History Length',
    'person_income': 'Income',
    'person_emp_length': 'Employment Length',
    'loan_grade': 'Loan Grade',
    'loan_int_rate': 'Interest Rate',
    'cb_person_default_on_file': 'Default On File'
}

MODEL_PATH = 'assets/best_model.pkl'

st.set_page_config(page_title="ZAHRA AI – Predicting Bank Client Solvency", layout="wide")
st.sidebar.title("ZAHRA AI Navigation")
section = st.sidebar.radio("Go to", ["📊 Data Overview", "🤖 Prediction"])

df = load_data()
df_clean, label_encoders, scaler = preprocess_data(df)

if section == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.write("### Data Preview")
    df_display = df.copy()
    df_display = df_display.rename(columns=DISPLAY_NAMES)
    st.dataframe(df_display.head(20))
    st.write("### Basic Statistics")
    st.write(df_display.describe())
    st.write("### Missing Values")
    st.write(df_display.isnull().sum())

elif section == "🤖 Prediction":
    st.title("🤖 Prediction")
    st.write("Enter client information to predict solvency.")
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        st.warning("Model not trained yet. Please run ml_pipeline.py first.")
        st.stop()
    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}
        for i, col in enumerate(df_clean.drop('loan_status', axis=1).columns):
            display_name = DISPLAY_NAMES.get(col, col)
            if df[col].dtype == 'object':
                options = df[col].unique().tolist()
                value = cols[i%2].selectbox(display_name, options)
                value = label_encoders[col].transform([value])[0]
            else:
                value = cols[i%2].number_input(display_name, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = value
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_input = pd.DataFrame([input_data])
        num_cols = X_input.select_dtypes(include=[np.number]).columns
        X_input[num_cols] = scaler.transform(X_input[num_cols])
        pred = model.predict(X_input)[0]
        result = "Good payer" if pred == 1 else "Bad payer"
        color = "#27ae60" if pred == 1 else "#c0392b"
        st.markdown(f'<h2 style="color:{color};text-align:center;">{result}</h2>', unsafe_allow_html=True)
        st.write("Prediction complete.")

st.sidebar.info("Project: ZAHRA AI – Predicting Bank Client Solvency\nAuthor: Your Name\nDate: 2025")

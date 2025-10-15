import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pickle
import os

# --- Constants ---
DATA_PATH = 'dataset/credit_data.csv'
MODEL_PATH = 'assets/best_model.pkl'

# --- Helper Functions ---
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    df = df.copy()
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Scale numerical columns
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('target') if 'target' in df.columns else df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, label_encoders, scaler

def split_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else y_pred
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_prob),
            'Confusion Matrix': confusion_matrix(y_test, y_pred),
            'ROC Curve': roc_curve(y_test, y_prob)
        }
    return results

def select_best_model(results):
    best = max(results.items(), key=lambda x: x[1]['ROC AUC'])
    return best[0]

def save_model(model):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="ZAHRA AI – Predicting Bank Client Solvency", layout="wide")
st.sidebar.title("ZAHRA AI Navigation")
section = st.sidebar.radio("Go to", ["📊 Data Overview", "⚙️ Model Training", "🤖 Prediction"])

# --- Load and preprocess data ---
df = load_data()
df_clean, label_encoders, scaler = preprocess_data(df)

if section == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.write("### Data Preview")
    st.dataframe(df.head(20))
    st.write("### Basic Statistics")
    st.write(df.describe())
    st.write("### Missing Values")
    st.write(df.isnull().sum())

elif section == "⚙️ Model Training":
    st.title("⚙️ Model Training")
    st.write("Split data, train models, and compare results.")
    X_train, X_test, y_train, y_test = split_data(df_clean)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    save_model(best_model)
    st.success(f"Best model: {best_model_name}")
    # Metrics Table
    metrics_df = pd.DataFrame({k: {m: v for m, v in res.items() if m not in ['Confusion Matrix', 'ROC Curve']} for k, res in results.items()}).T
    st.write("### Model Metrics")
    st.dataframe(metrics_df.style.background_gradient(cmap='Blues'))
    # Confusion Matrix & ROC Curve
    col1, col2, col3 = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with [col1, col2, col3][i]:
            st.write(f"#### {name}")
            fig, ax = plt.subplots()
            sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            fig2, ax2 = plt.subplots()
            fpr, tpr, _ = res['ROC Curve']
            ax2.plot(fpr, tpr, label=f"ROC (AUC={res['ROC AUC']:.2f})")
            ax2.plot([0,1],[0,1],'--',color='gray')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            st.pyplot(fig2)

elif section == "🤖 Prediction":
    st.title("🤖 Prediction")
    st.write("Enter client information to predict solvency.")
    # Load best model
    if os.path.exists(MODEL_PATH):
        model = load_model()
    else:
        st.warning("Model not trained yet. Please train in 'Model Training' section.")
        st.stop()
    # Build input form
    with st.form("prediction_form"):
        cols = st.columns(2)
        input_data = {}
        for i, col in enumerate(df_clean.drop('target', axis=1).columns):
            if df[col].dtype == 'object':
                options = df[col].unique().tolist()
                value = cols[i%2].selectbox(col, options)
                value = label_encoders[col].transform([value])[0]
            else:
                value = cols[i%2].number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = value
        submitted = st.form_submit_button("Predict")
    if submitted:
        X_input = pd.DataFrame([input_data])
        # Scale numerical columns
        num_cols = X_input.select_dtypes(include=[np.number]).columns
        X_input[num_cols] = scaler.transform(X_input[num_cols])
        pred = model.predict(X_input)[0]
        result = "Good payer" if pred == 1 else "Bad payer"
        color = "#27ae60" if pred == 1 else "#c0392b"
        st.markdown(f'<h2 style="color:{color};text-align:center;">{result}</h2>', unsafe_allow_html=True)
        st.write("Prediction complete.")

st.sidebar.info("Project: ZAHRA AI – Predicting Bank Client Solvency\nAuthor: Your Name\nDate: 2025")

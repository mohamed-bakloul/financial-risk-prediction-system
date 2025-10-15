import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import pickle
import os

DATA_PATH = 'dataset/credit_risk_dataset.csv'
MODEL_PATH = 'assets/best_model.pkl'

# --- ML Pipeline ---
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    df = df.copy()
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    # Scale numerical columns
    target_col = 'loan_status'
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col) if target_col in df.columns else df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, label_encoders, scaler

def split_data(df):
    target_col = 'loan_status'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
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

if __name__ == "__main__":
    df = load_data()
    df_clean, label_encoders, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_model_name = select_best_model(results)
    best_model = models[best_model_name]
    save_model(best_model)
    print(f"Best model: {best_model_name}")
    print(pd.DataFrame({k: {m: v for m, v in res.items() if m not in ['Confusion Matrix', 'ROC Curve']} for k, res in results.items()}).T)

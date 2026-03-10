# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:14:46 2024

@author: zzulk
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Soil Type Predictor", layout="wide")

# =========================
# CONFIG
# =========================
DATA_FILE = "ML_Analysis_Soil_Type_1.csv"

EXPECTED_FEATURES = [
    'TOC',
    'Field conductivity',
    'Lab conductivity',
    'Field resistivity (?)',
    'Lab. Resistivity (?a)',
    'Depth (m)',
    'Clay (%)',
    'Silt (%)',
    'Gravels (%)',
    'D10',
    'D30',
    'D60',
    'CU',
    'CC',
    '1D inverted resistivity',
    'Lab. Resistivity (Oa)',
    'Moisture content (%)',
    'pH',
    'Fine Soil (%)',
    'Sand (%)'
]

crop_recommendations = {
    "Inorganic Clay": ["Rice", "Sugarcane"],
    "Inorganic Silt": ["Wheat", "Barley"],
    "Poorly Graded Sand": ["Carrots", "Potatoes"],
    "Clayey Sand": ["Tomatoes", "Peppers"],
    "Sandy": ["Peanuts", "Cucumbers"],
    "Well-graded Sand": ["Lettuce", "Zucchini"]
}

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(inplace=True)
    return df

# =========================
# TRAIN MODEL ONCE
# =========================
@st.cache_resource
def train_model():
    df = load_data()

    if 'Soil_Type' not in df.columns:
        raise ValueError("Column 'Soil_Type' not found in dataset.")

    features = [f for f in EXPECTED_FEATURES if f in df.columns]
    if not features:
        raise ValueError("No expected feature columns were found in dataset.")

    X = df[features].copy()
    y = df['Soil_Type'].copy()

    # Ensure numeric features
    for col in features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X.dropna(inplace=True)
    y = y.loc[X.index]

    # Encode target labels safely
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, scaler, label_encoder, features, df, acc

# =========================
# PREDICTION FUNCTION
# =========================
def predict_soil_type(input_features_dict, clf, scaler, label_encoder, features):
    input_df = pd.DataFrame([input_features_dict])[features]
    input_scaled = scaler.transform(input_df)
    pred_encoded = clf.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return pred_label

# =========================
# UI
# =========================
st.title("Soil Type Predictor")

st.markdown(
    """
    <style>
    .stSlider > div > div > div > input[type="range"]::-webkit-slider-runnable-track {
        background: lightblue;
    }
    .stSlider > div > div > div > input[type="range"]::-webkit-slider-thumb {
        background: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    clf, scaler, label_encoder, features, data, acc = train_model()

    st.success(f"Model loaded successfully. Accuracy: {acc:.2%}")

    user_inputs = {}
    st.subheader("Enter Soil Parameters")

    for feature in features:
        col_data = pd.to_numeric(data[feature], errors='coerce').dropna()

        min_value = float(col_data.min())
        max_value = float(col_data.max())
        default_value = float(col_data.iloc[0])

        # avoid slider error if min == max
        if min_value == max_value:
            user_inputs[feature] = st.number_input(
                f"{feature}",
                value=default_value,
                key=feature
            )
        else:
            user_inputs[feature] = st.slider(
                f"{feature} (e.g. {default_value})",
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                key=feature
            )

    if st.button("Predict"):
        predicted_soil_type = predict_soil_type(
            user_inputs, clf, scaler, label_encoder, features
        )

        st.subheader("Prediction Result")
        st.write(f"**Predicted Soil Type:** {predicted_soil_type}")

        if predicted_soil_type in crop_recommendations:
            st.write(
                f"**Suggested crops:** {', '.join(crop_recommendations[predicted_soil_type])}"
            )
        else:
            st.write("No crop recommendations available for this soil type.")

except FileNotFoundError:
    st.error(f"Data file '{DATA_FILE}' not found. Please upload/include it in the deployment folder.")
except Exception as e:
    st.error(f"App error: {e}")
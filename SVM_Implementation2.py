# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:14:46 2024

@author: zzulk
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Soil Type Predictor",
    layout="wide"
)

# =========================
# HEADER IMAGE
# =========================
st.image(
    "https://raw.githubusercontent.com/zulianizulkoffli/Soil_Type_Estimator/main/land.jpg",
    caption="",
    use_container_width=True
)

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
    "Inorganic Clay": ["Paddy", "Cabbage", "Sengkuang"],
    "Inorganic Silt": ["French Bean", "Cauliflower", "Radish"],
    "Poorly Graded Sand": ["Rubber", "Cashew"],
    "Clayey Sand": ["Chilli", "Luffa", "Bitter Gourd", "Long Bean", "Cabbage", "Rubber"],
    "Sandy": ["Tomato", "Chilli", "Green Pepper", "Brinjal", "Okra", "Cucumber", "Coconut", "Tobacco", "Palm Oil", "Rubber"],
    "Well-graded Sand": ["Tomato", "Chilli", "Green Pepper", "Brinjal", "Okra", "Cucumber", "Luffa", "Bitter Gourd", "Coconut", "Palm Oil", "Rubber"],
}

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)

    # Drop fully empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Keep only rows without missing values
    df.dropna(inplace=True)

    return df

# =========================
# TRAIN MODELS ONCE ONLY
# =========================
@st.cache_resource
def train_models():
    df = load_data()

    if "Soil_Type" not in df.columns:
        raise ValueError("Column 'Soil_Type' not found in dataset.")

    features = [f for f in EXPECTED_FEATURES if f in df.columns]

    if not features:
        raise ValueError("None of the expected feature columns were found in dataset.")

    X = df[features].copy()
    y = df["Soil_Type"].copy()

    # Convert feature columns to numeric safely
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop rows with invalid numeric values
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X_train, y_train)
    gb_pred = gb_clf.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)

    # Neural Network
    nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    nn_clf.fit(X_train, y_train)
    nn_pred = nn_clf.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)

    return {
        "gb_model": gb_clf,
        "nn_model": nn_clf,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": features,
        "data": df,
        "gb_accuracy": gb_accuracy,
        "nn_accuracy": nn_accuracy
    }

# =========================
# PREDICTION FUNCTION
# =========================
def predict_soil_type(input_features, model_choice, scaler, label_encoder, features, gb_model, nn_model):
    input_df = pd.DataFrame([input_features])[features]
    input_scaled = scaler.transform(input_df)

    if model_choice == "Gradient Boosting":
        prediction_encoded = gb_model.predict(input_scaled)[0]
    else:
        prediction_encoded = nn_model.predict(input_scaled)[0]

    predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
    return predicted_label

# =========================
# UI
# =========================
st.title("Soil Type Predictor Based On Its Features in Peninsular Malaysia")

try:
    model_artifacts = train_models()

    gb_model = model_artifacts["gb_model"]
    nn_model = model_artifacts["nn_model"]
    scaler = model_artifacts["scaler"]
    label_encoder = model_artifacts["label_encoder"]
    features = model_artifacts["features"]
    data = model_artifacts["data"]
    gb_accuracy = model_artifacts["gb_accuracy"]
    nn_accuracy = model_artifacts["nn_accuracy"]

    st.success("Models loaded successfully.")

    model_choice = st.selectbox(
        "Choose the prediction model:",
        ["Gradient Boosting", "Neural Network"]
    )

    user_inputs = {}

    for feature in features:
        col_data = pd.to_numeric(data[feature], errors="coerce").dropna()

        min_value = float(col_data.min())
        max_value = float(col_data.max())
        default_value = float(col_data.iloc[0])

        if min_value == max_value:
            user_inputs[feature] = st.number_input(
                f"{feature}",
                value=default_value,
                key=feature
            )
        else:
            user_inputs[feature] = st.slider(
                f"{feature} (e.g., {default_value})",
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                key=feature
            )

    if st.button("Predict"):
        predicted_soil_type = predict_soil_type(
            input_features=user_inputs,
            model_choice=model_choice,
            scaler=scaler,
            label_encoder=label_encoder,
            features=features,
            gb_model=gb_model,
            nn_model=nn_model
        )

        if model_choice == "Gradient Boosting":
            accuracy = gb_accuracy
        else:
            accuracy = nn_accuracy

        st.subheader("Prediction Result")
        st.write(f"**Predicted soil type:** {predicted_soil_type}")
        st.write(f"**Model accuracy:** {accuracy:.2%}")

        if predicted_soil_type in crop_recommendations:
            recommended_crops = ", ".join(crop_recommendations[predicted_soil_type])
            st.markdown(
                f"**Suggested crops for {predicted_soil_type}:** {recommended_crops}"
            )
        else:
            st.write("No crop recommendations available for this soil type.")

except FileNotFoundError:
    st.error(f"Data file '{DATA_FILE}' not found. Make sure it is in the same folder as app.py.")
except Exception as e:
    st.error(f"App error: {e}")
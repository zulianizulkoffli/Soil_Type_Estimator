# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:14:46 2024

@author: zzulk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib
import streamlit as st

# Load the data
data = pd.read_csv('ML_Analysis_Soil_Type_1.csv')  # Ensure the CSV is in the same directory

# Replace zeros with ones where appropriate
data.replace(0, 1, inplace=True)

# Drop columns that are completely empty or unnecessary
data.dropna(axis=1, how='all', inplace=True)

# Drop rows with any NaN values
data = data.dropna()

# # Display the first few rows of the dataset
# st.write("First few rows of the dataset:")
# st.write(data.head())

# Define the features and target
features = ['TOC', 'Field conductivity', 'Lab conductivity', 'Field resistivity (?)',
            'Lab. Resistivity (?a)', 'Depth (m)', 'Clay (%)', 'Silt (%)', 'Gravels (%)', 
            'D10', 'D30', 'D60', 'CU', 'CC', '1D inverted resistivity', 'Lab. Resistivity (Oa)', 
            'Moisture content (%)', 'pH', 'Fine Soil (%)', 'Sand (%)']

# Ensure all specified features are present in the data
features = [f for f in features if f in data.columns]

# Split the dataset into features (X) and target (y)
X = data[features]
y = data['Soil_Type']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"Model Accuracy: {accuracy}")

# # Print the classification report
# unique_test_labels = np.unique(y_test)
# st.write("Classification Report:")
# st.text(classification_report(y_test, y_pred, labels=unique_test_labels))

# Save the model and scaler
joblib.dump(clf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Dictionary to map predicted class to soil type
soil_type_mapping = {
    1: "Inorganic Clay",
    2: "Inorganic Silt",
    3: "Poorly Graded Sand",
    4: "Clayey Sand",
    5: "Sandy",
    6: "Well-graded Sand"
}

# Dictionary to map soil type to best crops
crop_recommendations = {
    "Inorganic Clay": ["Rice", "Sugarcane"],
    "Inorganic Silt": ["Wheat", "Barley"],
    "Poorly Graded Sand": ["Carrots", "Potatoes"],
    "Clayey Sand": ["Tomatoes", "Peppers"],
    "Sandy": ["Peanuts", "Cucumbers"],
    "Well-graded Sand": ["Lettuce", "Zucchini"]
}

# Function to load the model and scaler and make predictions based on user inputs
def predict_soil_type(input_features):
    # Load the model and scaler
    clf = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Convert input_features to DataFrame to ensure correct format
    input_features_df = pd.DataFrame([input_features])
    
    # Scale the input features
    input_features_scaled = scaler.transform(input_features_df)
    
    # Make predictions
    prediction = clf.predict(input_features_scaled)
    
    # Map prediction to soil type
    predicted_soil_type = soil_type_mapping.get(prediction[0], "unrecognize")
    
    return predicted_soil_type

# Streamlit UI for user input
st.title("Soil Type Predictor")

# Add custom CSS to change slider and pointer color
st.markdown(
    """
    <style>
    .stSlider > div > div > div > input[type=range]::-webkit-slider-runnable-track {
        background: lightblue;
    }
    .stSlider > div > div > div > input[type=range]::-webkit-slider-thumb {
        background: lightblue;
    }
    .stSlider > div > div > div > input[type=range]::-moz-range-track {
        background: lightblue;
    }
    .stSlider > div > div > div > input[type=range]::-moz-range-thumb {
        background: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

user_inputs = {}
for feature in features:
    value = float(data[feature].iloc[0])
    min_value = float(data[feature].min())
    max_value = float(data[feature].max())
    user_inputs[feature] = st.slider(f"{feature} (e.g., {value})", min_value=min_value, max_value=max_value, value=value, key=feature)

if st.button("Predict"):
    predicted_soil_type = predict_soil_type(user_inputs)
    st.write(f"The predicted soil type is: {predicted_soil_type}")
    
    if predicted_soil_type in crop_recommendations:
        st.write(f"Suggested crops for {predicted_soil_type}: {', '.join(crop_recommendations[predicted_soil_type])}")
    else:
        st.write("No crop recommendations available for this soil type.")

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:14:46 2024

@author: zzulk
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import streamlit as st

# Add an image at the top of the app
st.image('https://raw.githubusercontent.com/zulianizulkoffli/Soil_Type_Estimator/main/land.jpg', caption='', use_column_width=True) # Replace 'header_image.jpg' with your image file name or URL

# Load the data
data = pd.read_csv('ML_Analysis_Soil_Type_1.csv')  # Ensure the CSV is in the same directory

# Replace zeros with ones where appropriate
data.replace(0, 1, inplace=True)

# Drop columns that are completely empty or unnecessary
data.dropna(axis=1, how='all', inplace=True)

# Drop rows with any NaN values
data = data.dropna()

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

# Train Gradient Boosting model
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train, y_train)
gb_accuracy = accuracy_score(y_test, gb_clf.predict(X_test))

# Train Neural Network model
nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn_clf.fit(X_train, y_train)
nn_accuracy = accuracy_score(y_test, nn_clf.predict(X_test))

# Save the models and scaler
joblib.dump(gb_clf, 'gradient_boosting_model.pkl')
joblib.dump(nn_clf, 'neural_network_model.pkl')
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
    "Inorganic Clay": ["Paddy", "Cabbage", "Sengkuang"],  # Reference: Joseph et al., 1974; MARDI Research Station, Jalan Kebun (Siti Doya, pers. comm.)
    "Inorganic Silt": ["French Bean", "Cauliflower", "Radish"],  # Reference: Joseph et al., 1974
    "Poorly Graded Sand": ["Rubber", "Cashew"],  # Reference: Chai, 1981; Khairuddin & Kamaruddin, 1980
    "Clayey Sand": ["Chilli", "Luffa", "Bitter Gourd", "Long Bean", "Cabbage", "Rubber"],  # Reference: Joseph et al., 1974
    "Sandy": ["Tomato", "Chilli", "Green Pepper", "Brinjal", "Okra", "Cucumber", "Coconut", "Tobacco", "Palm Oil", "Rubber"],  # Reference: Kho et al., 1979; Zainuddin, 1981
    "Well-graded Sand": ["Tomato", "Chilli", "Green Pepper", "Brinjal", "Okra", "Cucumber", "Luffa", "Bitter Gourd", "Coconut", "Palm Oil", "Rubber"],  # Reference: Kho et al., 1979; Zainuddin, 1981
}

# Function to load the model and scaler and make predictions based on user inputs
def predict_soil_type(input_features, model_choice):
    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Convert input_features to DataFrame to ensure correct format
    input_features_df = pd.DataFrame([input_features])
    
    # Scale the input features
    input_features_scaled = scaler.transform(input_features_df)
    
    # Load the chosen model
    if model_choice == 'Gradient Boosting':
        clf = joblib.load('gradient_boosting_model.pkl')
        accuracy = gb_accuracy
    elif model_choice == 'Neural Network':
        clf = joblib.load('neural_network_model.pkl')
        accuracy = nn_accuracy
    
    # Make predictions
    prediction = clf.predict(input_features_scaled)
    
    # Map prediction to soil type
    predicted_soil_type = soil_type_mapping.get(prediction[0], "unrecognized")
    
    return predicted_soil_type, accuracy

# Streamlit UI for user input
st.title("Soil Type Predictor Based On It's Features In Peninsular Malaysia")

# Model choice
model_choice = st.selectbox("Choose the prediction model:", ["Gradient Boosting", "Neural Network"])

user_inputs = {}
for feature in features:
    value = float(data[feature].iloc[0])
    min_value = float(data[feature].min())
    max_value = float(data[feature].max())
    user_inputs[feature] = st.slider(f"{feature} (e.g., {value})", min_value=min_value, max_value=max_value, value=value, key=feature)

# Display crop recommendations with larger and bold font using Markdown
if st.button("Predict"):
    predicted_soil_type, accuracy = predict_soil_type(user_inputs, model_choice)
    st.write(f"The predicted soil type is: {predicted_soil_type}")
    st.write(f"Model Accuracy: {accuracy:.2f}")

    if predicted_soil_type in crop_recommendations:
        # Display the recommendations in bold and larger font using Markdown
        recommended_crops = ', '.join(crop_recommendations[predicted_soil_type])
        st.markdown(f"<span style='font-size: 1em;'>Suggested crops for {predicted_soil_type}: {recommended_crops}</span>", unsafe_allow_html=True)
    else:
        st.write("No crop recommendations available for this soil type.")


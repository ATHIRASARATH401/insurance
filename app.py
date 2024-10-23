
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import streamlit as st
import pandas as pd
import numpy as np

# Generate a synthetic dataset (replace with your real data)
X, y = make_classification(n_samples=1000, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('fraud_detection_model (8).pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the saved model
model_path = 'fraud_detection_model (8).pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app title and icon
st.set_page_config(page_title="Fraud Detection App", page_icon="🔍", layout="wide")
st.title("Vehicle Insurance Claim Fraud Detection")

# Introduction text
st.markdown("""
    <style>
        .big-font {
            font-size:30px !important;
        }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="big-font">Enter the details below:</p>', unsafe_allow_html=True)

# Input fields for user input
col1, col2 = st.columns(2)

with col1:
    week_of_month = st.number_input('Week Of Month', min_value=1, max_value=5, value=1)
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    policy_number = st.number_input('Policy Number', min_value=1000, max_value=100000, value=1000)
    deductible = st.number_input('Deductible', min_value=0, max_value=1000, value=500)

with col2:
    week_of_month_claimed = st.number_input('Week Of Month Claimed', min_value=1, max_value=5, value=1)
    rep_number = st.number_input('Rep Number', min_value=1, max_value=1000, value=1)
    driver_rating = st.slider('Driver Rating', min_value=1, max_value=5, value=3)
    year = st.number_input('Year', min_value=1990, max_value=2024, value=2020)

# Prepare input data for prediction
input_data = {
    'week_of_month': week_of_month,
    'week_of_month_claimed': week_of_month_claimed,
    'policy_number': policy_number,
    'rep_number': rep_number,
    'Age': age,
    'deductible': deductible,
    'driver_rating': driver_rating,
    'year': year
}
input_df = pd.DataFrame([input_data])

# Predict button
if st.button('Predict Fraud'):
    # Collect input into a numpy array
    features = np.array([[week_of_month, week_of_month_claimed, age, policy_number, rep_number, deductible, driver_rating, year]])
    
    # Use the model to make a prediction
    prediction = model.predict(features)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🚨 **Fraud detected!** 🚨")
    else:
        st.success("✅ **No fraud detected.**")

# Optional: Add footer
st.markdown("---")
st.markdown(" FRAUD DETECTION APP ")

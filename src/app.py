import streamlit as st
import pandas as pd
import os
import joblib

# -----------------------------
# Load trained model
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Dynamic Pricing Demand Predictor")

st.write("""
This app predicts product demand based on:
- Base Price
- Competitor Price
""")

# Product selection
product = st.selectbox("Select Product:", ["Wireless Earbuds", "Gaming Mouse", "Smartwatch"])

# Input prices
base_price = st.number_input("Enter Base Price ($):", min_value=1.0, value=50.0, step=1.0)
competitor_price = st.number_input("Enter Competitor Price ($):", min_value=1.0, value=50.0, step=1.0)

# Predict button
if st.button("Predict Demand"):
    import numpy as np
    X_new = pd.DataFrame([[base_price, competitor_price]], columns=["base_price", "competitor_price"])
    predicted_demand = model.predict(X_new)[0]
    st.success(f"Predicted Demand for {product}: {predicted_demand:.2f} units")

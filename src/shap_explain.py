import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "dynamic_pricing_data.csv")
df = pd.read_csv(data_path)

# -----------------------------
# Load trained model
# -----------------------------
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Prepare features
# -----------------------------
X = df[["base_price", "competitor_price"]]

# -----------------------------
# SHAP Explainability
# -----------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# -----------------------------
# Plot SHAP summary
# -----------------------------
shap.summary_plot(shap_values, X, show=True)

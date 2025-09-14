import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load data and model
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")

df = pd.read_csv(data_path)
model = joblib.load(model_path)

# ✅ Match feature names in your dataset
feature_cols = ["base_price", "avg_competitor_price", "promotion_flag", "inventory_level"]
X = df[feature_cols]
# -----------------------------
# SHAP analysis
# -----------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# ✅ Ensure reports folder exists
reports_dir = os.path.join(script_dir, "..", "reports")
os.makedirs(reports_dir, exist_ok=True)

# Summary plot
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
summary_path = os.path.join(reports_dir, "shap_summary.png")
plt.savefig(summary_path)
print(f"✅ Saved SHAP summary plot to {summary_path}")

# Force plot for a single prediction
shap.initjs()
sample = X.iloc[0]
shap.force_plot(explainer.expected_value, shap_values[0].values, sample, matplotlib=True, show=False)
plt.title("Force Plot for First Sample")
plt.tight_layout()
force_path = os.path.join(reports_dir, "shap_force.png")
plt.savefig(force_path)
print(f"✅ Saved SHAP force plot for first sample at {force_path}")

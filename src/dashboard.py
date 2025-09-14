# src/dashboard.py
import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
reports_dir = os.path.join(script_dir, "..", "reports")

# -----------------------------
# Load data and model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    return joblib.load(model_path)

df = load_data()
model = load_model()

# -----------------------------
# Dashboard layout
# -----------------------------
st.title("📊 Dynamic Pricing Dashboard")
st.write("Explore demand, pricing, and SHAP feature importance interactively.")

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Product selector
products = df["product"].unique()
selected_product = st.selectbox("Select a product:", products)

# Filter data for the selected product
product_data = df[df["product"] == selected_product]

# Plot demand vs. base_price for selected product
st.subheader(f"Demand vs Base Price for {selected_product}")
fig, ax = plt.subplots()
ax.scatter(product_data["base_price"], product_data["demand"], alpha=0.6)
ax.set_xlabel("Base Price")
ax.set_ylabel("Demand")
st.pyplot(fig)

# Show SHAP plots if they exist
shap_summary = os.path.join(reports_dir, "shap_summary.png")
shap_force = os.path.join(reports_dir, "shap_force.png")

st.subheader("Feature Importance (SHAP)")
if os.path.exists(shap_summary):
    st.image(shap_summary, caption="SHAP Summary Plot", use_column_width=True)
else:
    st.warning("SHAP summary plot not found. Run shap_analysis.py first.")

st.subheader("Force Plot (Single Prediction)")
if os.path.exists(shap_force):
    st.image(shap_force, caption="SHAP Force Plot", use_column_width=True)
else:
    st.warning("SHAP force plot not found. Run shap_analysis.py first.")

# Interactive prediction
st.subheader("🔮 Predict Demand")
base_price = st.slider("Base Price", int(df["base_price"].min()), int(df["base_price"].max()), 50)
avg_competitor_price = st.slider("Avg Competitor Price", int(df["avg_competitor_price"].min()), int(df["avg_competitor_price"].max()), 55)
promotion_flag = st.selectbox("Promotion Flag (discount)", [-10, -5, 0])
inventory_level = st.slider("Inventory Level", int(df["inventory_level"].min()), int(df["inventory_level"].max()), 200)

# Make a prediction
features = pd.DataFrame([{
    "base_price": base_price,
    "avg_competitor_price": avg_competitor_price,
    "promotion_flag": promotion_flag,
    "inventory_level": inventory_level
}])

predicted_demand = model.predict(features)[0]
st.write(f"**Predicted Demand:** {predicted_demand:.2f}")
st.subheader("📈 Revenue & Profit Projection")

# Check if required columns exist
if {"product", "revenue", "profit"}.issubset(df.columns):
    # Aggregate totals by product
    summary = df.groupby("product")[["revenue", "profit"]].sum().reset_index()
    st.dataframe(summary)

    # Bar chart
    st.bar_chart(summary.set_index("product")[["revenue", "profit"]])

    # Allow CSV download
    st.download_button(
        "Download Revenue & Profit CSV",
        summary.to_csv(index=False),
        file_name="revenue_profit_summary.csv"
    )
else:
    st.warning("Run the updated data generation script to include revenue and profit columns.")

st.success("✅ Dashboard loaded successfully!")

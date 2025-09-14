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
# Prepare revenue/profit columns
# -----------------------------
if "revenue" not in df.columns or "profit" not in df.columns:
    df["revenue"] = df["base_price"] * df["demand"]
    df["profit"] = df["revenue"] * 0.3  # 30% margin assumption

# -----------------------------
# Dashboard layout
# -----------------------------
st.title("📊 Dynamic Pricing Dashboard")
st.write("Explore demand, pricing, revenue, profit, and SHAP feature importance interactively.")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Product selector
products = df["product"].unique()
selected_product = st.selectbox("Select a product:", products)

# Filtered data for selected product
product_data = df[df["product"] == selected_product]

# Demand vs Base Price scatter plot
st.subheader(f"Demand vs Base Price for {selected_product}")
fig, ax = plt.subplots()
ax.scatter(product_data["base_price"], product_data["demand"], alpha=0.6)
ax.set_xlabel("Base Price")
ax.set_ylabel("Demand")
st.pyplot(fig)

# -----------------------------
# Revenue and Profit Summary
# -----------------------------
st.subheader("💰 Revenue and Profit by Product")
summary = df.groupby("product")[["revenue", "profit"]].sum()
st.bar_chart(summary)

# -----------------------------
# 📊 Revenue/Profit Trends Over Time
# -----------------------------
st.subheader("📊 Revenue/Profit Trends Over Time")
metric = st.radio("Select Metric:", ["revenue", "profit"])
products_to_compare = st.multiselect("Select Products:", products, default=products[:2])

trend_data = df[df["product"].isin(products_to_compare)]
trend_summary = trend_data.groupby(["day", "product"])[metric].sum().reset_index()

# Line chart for trends
st.line_chart(trend_summary.pivot(index="day", columns="product", values=metric))

# 🏆 Top Products by Profit
st.subheader("🏆 Top Products by Profit")
top_products = df.groupby("product")["profit"].sum().sort_values(ascending=False).head(3)
st.bar_chart(top_products)

# Download trend data
st.download_button(
    "⬇️ Download Trend Data",
    trend_summary.to_csv(index=False),
    file_name=f"{metric}_trends.csv"
)

# -----------------------------
# SHAP Feature Importance
# -----------------------------
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

# -----------------------------
# Interactive Prediction
# -----------------------------
st.subheader("🔮 Predict Demand")
base_price = st.slider("Base Price", int(df["base_price"].min()), int(df["base_price"].max()), 50)
avg_competitor_price = st.slider(
    "Avg Competitor Price",
    int(df["avg_competitor_price"].min()),
    int(df["avg_competitor_price"].max()),
    55,
)
promotion_flag = st.selectbox("Promotion Flag (discount)", [-10, -5, 0])
inventory_level = st.slider(
    "Inventory Level",
    int(df["inventory_level"].min()),
    int(df["inventory_level"].max()),
    200,
)

features = pd.DataFrame([{
    "base_price": base_price,
    "avg_competitor_price": avg_competitor_price,
    "promotion_flag": promotion_flag,
    "inventory_level": inventory_level
}])

predicted_demand = model.predict(features)[0]
st.write(f"**Predicted Demand:** {predicted_demand:.2f}")

st.success("✅ Dashboard loaded successfully!")

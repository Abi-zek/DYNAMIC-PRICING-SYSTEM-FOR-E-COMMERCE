import pandas as pd
import os
import matplotlib.pyplot as plt

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")
df = pd.read_csv(data_path)

# Add cost column if missing
if "cost_per_unit" not in df.columns:
    df["cost_per_unit"] = df["base_price"] * 0.6

# Calculate revenue and profit
df["revenue"] = df["base_price"] * df["demand"]
df["profit"] = (df["base_price"] - df["cost_per_unit"]) * df["demand"]

# Group by product
summary = df.groupby("product")[["revenue", "profit"]].sum().reset_index()

# Save summary
reports_dir = os.path.join(script_dir, "..", "reports")
os.makedirs(reports_dir, exist_ok=True)
output_path = os.path.join(reports_dir, "revenue_profit_summary.csv")
summary.to_csv(output_path, index=False)

# Plot revenue vs profit
summary.plot(x="product", y=["revenue", "profit"], kind="bar")
plt.title("Revenue and Profit by Product")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "revenue_profit_plot.png"))
print(f"✅ Revenue/profit summary saved to {output_path}")

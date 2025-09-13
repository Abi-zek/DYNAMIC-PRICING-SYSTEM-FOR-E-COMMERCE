import pandas as pd
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load advanced dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")
df = pd.read_csv(data_path)

# -----------------------------
# Load trained baseline model
# -----------------------------
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
model = joblib.load(model_path)

# -----------------------------
# Price optimization & elasticity function
# -----------------------------
def optimize_price_with_elasticity(product_name, price_range):
    product_df = df[df["product"] == product_name]
    avg_comp_price = product_df["avg_competitor_price"].mean()
    
    prices = []
    demands = []
    revenues = []
    elasticities = []

    for i, price in enumerate(price_range):
        X_new = pd.DataFrame([[price, avg_comp_price]], columns=["base_price", "competitor_price"])
        demand = model.predict(X_new)[0]
        revenue = price * demand

        # Elasticity calculation: %ΔQ / %ΔP
        if i > 0:
            delta_q = demand - demands[-1]
            delta_p = price - prices[-1]
            elasticity = (delta_q / demands[-1]) / (delta_p / prices[-1])
        else:
            elasticity = np.nan  # No previous point for first price

        prices.append(price)
        demands.append(demand)
        revenues.append(revenue)
        elasticities.append(elasticity)

    result_df = pd.DataFrame({
        "price": prices,
        "predicted_demand": demands,
        "predicted_revenue": revenues,
        "price_elasticity": elasticities
    })

    # Find optimal price (max revenue)
    best_idx = result_df["predicted_revenue"].idxmax()
    best_price = result_df.loc[best_idx, "price"]
    max_revenue = result_df.loc[best_idx, "predicted_revenue"]

    return best_price, max_revenue, result_df

# -----------------------------
# Example usage
# -----------------------------
product = "Wireless Earbuds"
price_range = np.arange(40, 100, 1)
best_price, max_rev, table = optimize_price_with_elasticity(product, price_range)

print(f"Optimal Price for {product}: ${best_price:.2f}")
print(f"Expected Maximum Revenue: ${max_rev:.2f}")
print("\nSample table:")
print(table.head())

# -----------------------------
# Plot scenario simulation
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(table["price"], table["predicted_demand"], label="Predicted Demand")
plt.plot(table["price"], table["predicted_revenue"], label="Predicted Revenue")
plt.axvline(best_price, color='red', linestyle='--', label=f"Optimal Price: ${best_price:.2f}")
plt.xlabel("Price")
plt.title(f"Price Optimization Scenario for {product}")
plt.legend()
plt.show()

# Elasticity plot
plt.figure(figsize=(10,6))
plt.plot(table["price"], table["price_elasticity"])
plt.axhline(-1, color='red', linestyle='--', label="Unit Elasticity")
plt.xlabel("Price")
plt.ylabel("Price Elasticity")
plt.title(f"Price Elasticity Curve for {product}")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import os

# -----------------------------
# Ensure data folder exists
# -----------------------------
os.makedirs("../data", exist_ok=True)

# -----------------------------
# Parameters for synthetic data
# -----------------------------
np.random.seed(42)
days = 365
products = ["Wireless Earbuds", "Gaming Mouse", "Smartwatch"]
competitors = ["CompA", "CompB", "CompC"]
customer_types = ["loyal", "occasional", "price_sensitive"]
holidays = [50, 120, 200, 300]  # Example special days

data = []

for product in products:
    base_price = np.random.randint(40, 100)
    seasonality = np.sin(np.linspace(0, 2 * np.pi, days)) * 10
    inventory = np.random.randint(50, 500)  # Stock level
    for day in range(days):
        weekday_effect = 5 if day % 7 in [5, 6] else 0  # Weekend boost
        holiday_effect = 20 if day in holidays else 0
        customer_effect = np.random.choice([1.0, 0.9, 1.1], p=[0.5, 0.3, 0.2])
        promotion_effect = np.random.choice([0, -5, -10], p=[0.7, 0.2, 0.1])  # Discounts

        # Competitor prices
        comp_prices = {comp: base_price + np.random.normal(0, 5) for comp in competitors}

        # Calculate average competitor price
        avg_comp_price = np.mean(list(comp_prices.values()))

        # Simulate demand
        demand = (
            200
            + seasonality[day] * 2
            - (avg_comp_price - base_price) * 3
            + weekday_effect
            + holiday_effect
        )
        demand = demand * customer_effect
        demand += promotion_effect
        demand = max(0, round(demand, 2))  # No negative demand

        data.append({
            "day": day + 1,
            "product": product,
            "base_price": base_price,
            "avg_competitor_price": round(avg_comp_price, 2),
            "inventory": inventory,
            "promotion_discount": promotion_effect,
            "customer_type": np.random.choice(customer_types),
            "demand": demand
        })

# Convert to DataFrame
# Convert to DataFrame
df = pd.DataFrame(data)

# Rename columns for consistency with SHAP analysis
df.rename(columns={
    "inventory": "inventory_level",
    "promotion_discount": "promotion_flag"
}, inplace=True)

# Save to CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")
df.to_csv(output_path, index=False)

print(f"✅ Advanced dataset created and saved to {output_path}")
print(df.head())

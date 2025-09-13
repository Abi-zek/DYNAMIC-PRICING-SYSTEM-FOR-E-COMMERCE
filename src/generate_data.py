import numpy as np
import pandas as pd
import os

# Ensure the data folder exists
os.makedirs("../data", exist_ok=True)

# -------------------------------
# Parameters for synthetic data
# -------------------------------
np.random.seed(42)  # For reproducibility
days = 365  # One year of data
products = ["Wireless Earbuds", "Gaming Mouse", "Smartwatch"]

data = []

for product in products:
    base_price = np.random.randint(40, 100)  # Base product price in dollars
    seasonality = np.sin(np.linspace(0, 2 * np.pi, days)) * 10  # Seasonal effect
    competitor_effect = np.random.normal(0, 5, days)  # Random competitor influence
    demand_noise = np.random.normal(0, 15, days)  # Random noise in demand

    for day in range(days):
        # Simulate competitor price fluctuations
        competitor_price = base_price + competitor_effect[day]
        
        # Simulate demand based on pricing and seasonality
        demand = (
            200  # Base demand
            + seasonality[day] * 2
            - (competitor_price - base_price) * 3  # Competitor influence on demand
            + demand_noise[day]
        )

        data.append({
            "day": day + 1,
            "product": product,
            "base_price": base_price,
            "competitor_price": round(competitor_price, 2),
            "demand": max(0, round(demand, 2))  # Ensure no negative demand
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "data", "dynamic_pricing_data.csv")

print(f"Saving CSV to: {output_path}")  # Debug line
df.to_csv(output_path, index=False)
print("✅ Finished saving CSV")
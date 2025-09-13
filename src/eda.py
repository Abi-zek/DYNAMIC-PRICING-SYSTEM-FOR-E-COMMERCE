import pandas as pd
import os
import matplotlib.pyplot as plt

# -----------------------------
# Locate and load the dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "dynamic_pricing_data.csv")

# Load CSV
df = pd.read_csv(data_path)

# Preview first rows
print("First 5 rows of data:")
print(df.head(), "\n")

# Basic info
print("Dataset Info:")
print(df.info(), "\n")

# Descriptive statistics
print("Summary Statistics:")
print(df.describe(), "\n")

# -----------------------------
# Visualizations
# -----------------------------
# Average demand per product
avg_demand = df.groupby("product")["demand"].mean()
avg_demand.plot(kind="bar", title="Average Demand per Product")
plt.ylabel("Average Demand")
plt.show()

# Demand trends over time for one product
earbuds = df[df["product"] == "Wireless Earbuds"]
plt.plot(earbuds["day"], earbuds["demand"])
plt.title("Demand Over Time - Wireless Earbuds")
plt.xlabel("Day")
plt.ylabel("Demand")
plt.show()

# Relationship between competitor price and demand
plt.scatter(df["competitor_price"], df["demand"], alpha=0.3)
plt.title("Competitor Price vs Demand")
plt.xlabel("Competitor Price")
plt.ylabel("Demand")
plt.show()

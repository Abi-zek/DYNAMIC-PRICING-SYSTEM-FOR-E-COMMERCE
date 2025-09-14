import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Locate and load the dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "advanced_dynamic_pricing_data.csv")

df = pd.read_csv(data_path)

# -----------------------------
# Prepare features and target
# -----------------------------
# Rename columns in your dataset to match these if needed:
# promotion_discount -> promotion_flag
df.rename(columns={"promotion_discount": "promotion_flag", "inventory": "inventory_level"}, inplace=True)

feature_cols = ["base_price", "avg_competitor_price", "promotion_flag", "inventory_level"]
X = df[feature_cols]
y = df["demand"]

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained with {len(feature_cols)} features")
print(f"📊 MSE: {mse:.2f}, R²: {r2:.2f}")

# -----------------------------
# Save model
# -----------------------------
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
os.makedirs(os.path.join(script_dir, "..", "models"), exist_ok=True)
joblib.dump(model, model_path)

print(f"💾 Model saved to {model_path}")

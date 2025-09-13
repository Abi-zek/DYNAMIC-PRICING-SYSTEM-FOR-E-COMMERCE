import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -----------------------------
# Load dataset
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "dynamic_pricing_data.csv")
df = pd.read_csv(data_path)

# -----------------------------
# Prepare features and target
# -----------------------------
# We'll predict 'demand' using 'base_price' and 'competitor_price'
X = df[["base_price", "competitor_price"]]
y = df["demand"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# -----------------------------
# Evaluate the model
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# -----------------------------
# Save the trained model
# -----------------------------
os.makedirs(os.path.join(script_dir, "..", "models"), exist_ok=True)
model_path = os.path.join(script_dir, "..", "models", "baseline_demand_model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

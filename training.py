import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
VS = pd.read_csv(r"C:\Users\Admin\Desktop\car\car_prediction_data.csv")

# Features and Target
X = VS.drop(["Selling_Price", "Car_Name"], axis=1)
y = VS["Selling_Price"]

# Categorical Columns
cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save Model
joblib.dump(pipeline, "models/car_price_model.pkl")

print("Model Saved Successfully!")
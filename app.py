import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")

# --------------------------
# Load Data
# --------------------------
VS= pd.read_csv(r"C:\Users\Admin\Desktop\car\car_prediction_data.csv")

# --------------------------
# Data Cleaning
# --------------------------
VS.drop_duplicates(inplace=True)

# --------------------------
# Dataset Preview
# --------------------------
st.subheader("Dataset Preview")

st.dataframe(VS.head())

# --------------------------
# Feature Engineering
# --------------------------
X = VS.drop(["Selling_Price", "Car_Name"], axis=1)
y = VS["Selling_Price"]

cat_cols = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat",
         OneHotEncoder(handle_unknown="ignore"),
         cat_cols)
    ],
    remainder="passthrough"
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X, y)

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("Enter Car Details")

year = st.sidebar.slider(
    "Year",
    int(VS["Year"].min()),
    int(VS["Year"].max()),
    2018
)

present_price = st.sidebar.number_input(
    "Present Price",
    min_value=0.0,
    value=5.0
)

kms_driven = st.sidebar.number_input(
    "Kms Driven",
    min_value=0,
    value=30000
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    VS["Fuel_Type"].unique()
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    VS["Seller_Type"].unique()
)

transmission = st.sidebar.selectbox(
    "Transmission",
    VS["Transmission"].unique()
)

owner = st.sidebar.selectbox(
    "Owner",
    sorted(VS["Owner"].unique())
)

# --------------------------
# Prediction
# --------------------------
input_VS = pd.DataFrame({
    "Year":[year],
    "Present_Price":[present_price],
    "Kms_Driven":[kms_driven],
    "Fuel_Type":[fuel_type],
    "Seller_Type":[seller_type],
    "Transmission":[transmission],
    "Owner":[owner]
})

prediction = pipeline.predict(input_VS)[0]

# --------------------------
# Display Prediction
# --------------------------
st.subheader("Predicted Car Price")

st.success(f"₹ {prediction:.2f} Lakhs")

# --------------------------
# Feature Importance
# --------------------------
st.subheader("Feature Importance")

encoded_features = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_

importance_VS = pd.DataFrame({
    "Feature": encoded_features,
    "Importance": importances
})

importance_VS = importance_VS.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    importance_VS["Feature"][:10],
    importance_VS["Importance"][:10]
)

ax.set_title("Top 10 Important Features")

st.pyplot(fig)

# --------------------------
# Price Distribution
# --------------------------
st.subheader("Selling Price Distribution")

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.hist(
    VS["Selling_Price"],
    bins=20
)

ax2.set_xlabel("Price")

ax2.set_ylabel("Count")

st.pyplot(fig2)

# --------------------------
# Actual vs Predicted Demo
# --------------------------
st.subheader("Price Comparison")

sample = X.head(20)

preds = pipeline.predict(sample)

comparison = pd.DataFrame({
    "Actual": y.head(20),
    "Predicted": preds
})

st.line_chart(comparison)

st.dataframe(comparison)
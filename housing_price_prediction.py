import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("HousingData.csv")

# Fill missing values with column means
data = data.fillna(data.mean())

# Handle outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Select important features based on correlation
correlation_matrix = data.corr()
important_features = correlation_matrix["MEDV"].abs().sort_values(ascending=False)
selected_features = important_features[important_features > 0.4].index.tolist()
selected_features.remove("MEDV")

# Split data into features and target
x = data[selected_features]
y = data["MEDV"]

# Scale features
scaler = StandardScaler()
x_norm = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.1, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_y_pred = lr_model.predict(x_test)
lr_mse = mean_squared_error(y_test, lr_y_pred)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)

# Gradient Boosting Regressor (XGBoost)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(x_train, y_train)
xgb_y_pred = xgb_model.predict(x_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)

# Determine best model
results = {
    "Linear Regression": lr_mse,
    "Random Forest": rf_mse,
    "XGBoost": xgb_mse
}
best_model = min(results, key=results.get)

# Print results
print("Mean Squared Error (MSE) for each model:")
for model, mse in results.items():
    print(f"{model}: {mse:.2f}")
print(f"Best model: {best_model} with MSE = {results[best_model]:.2f}")
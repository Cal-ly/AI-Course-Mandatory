import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import logging

# Configure structured logging
logging.basicConfig(filename='data/clean_main_output.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logging.info("\n--- Startup Profit Prediction Model (without OneHotEncoder) ---")

# Load dataset
dataset = pd.read_csv('data/50_Startups.csv')

# Drop categorical feature "State" (as per original instruction)
dataset = dataset.drop("State", axis=1)

# Separate features (X) and target variable (y)
X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

# Split dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Dataset split: Training set size = {len(X_train)}, Test set size = {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training (Linear Regression)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
logging.info("Linear Regression model successfully trained.")

# Sample predictions
sample_predictions = lin_reg.predict(X_test_scaled[:5])
logging.info("\nSample Predictions Analysis:")
logging.info(f"Predicted values: {sample_predictions.round(2)}")
logging.info(f"Actual values: {y_test.iloc[:5].values}")

# Evaluate prediction accuracy for sample
sample_rmse = np.sqrt(mean_squared_error(y_test.iloc[:5], sample_predictions))
sample_percentage_error = (np.abs(y_test.iloc[:5].values - sample_predictions) / y_test.iloc[:5].values) * 100
average_sample_percentage_error = np.mean(sample_percentage_error)

logging.info(f"Sample RMSE: {sample_rmse:.2f}")
logging.info(f"Percentage errors: {sample_percentage_error.round(2)}")
logging.info(f"Average sample percentage error: {average_sample_percentage_error:.2f}%")

# Final model evaluation on entire test set
predictions_test = lin_reg.predict(X_test_scaled)
final_rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
logging.info("\nFinal Model Performance:")
logging.info(f"Final Test RMSE: {final_rmse:.2f}")

# Cross-validation for robustness
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
cv_rmse_scores = np.sqrt(-cv_scores)

logging.info("\nCross-validation RMSE Analysis:")
logging.info(f"CV RMSE scores: {cv_rmse_scores.round(2)}")
logging.info(f"Average CV RMSE: {cv_rmse_scores.mean():.2f}")
logging.info(f"CV RMSE standard deviation: {cv_rmse_scores.std():.2f}")

# Feature importance analysis (Regression coefficients)
logging.info("\nFeature Importance (Regression Coefficients):")
for feature, coef in zip(X.columns, lin_reg.coef_):
    logging.info(f"{feature}: {coef:.2f}")

# Visualize Actual vs Predicted profits
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions_test, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression: Actual vs Predicted Profits")
plt.grid(True)
plt.savefig('data/clean_main_actual_vs_predicted.png')
logging.info("Plot of Actual vs. Predicted profits saved successfully.")

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Configure logging clearly and concisely
logging.basicConfig(filename='data/clean_ohe_output.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')
logging.info("\n--- Startup Profit Prediction Model (with OneHotEncoder) ---")

# Load dataset
dataset = pd.read_csv('data/50_Startups.csv')

# Separate features and labels
X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Dataset split: Training set size = {len(X_train)}, Test set size = {len(X_test)}")

# Define numerical and categorical features
num_features = ["R&D Spend", "Administration", "Marketing Spend"]
cat_features = ["State"]

# Setup preprocessing pipelines clearly
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(), cat_features)
])

# Data transformation
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# Linear Regression Model training
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

# Evaluate predictions on test sample
sample_predictions = lin_reg.predict(X_test_prepared[:5])
logging.info("\nSample Predictions:")
logging.info(f"Predicted: {sample_predictions.round(2)}")
logging.info(f"Actual: {y_test.iloc[:5].values}")

# Evaluate the RMSE on sample predictions
sample_rmse = np.sqrt(mean_squared_error(y_test.iloc[:5].values, sample_predictions))
logging.info(f"Sample RMSE: {sample_rmse:.2f}")

# Calculate percentage errors for sample predictions
percentage_errors = (np.abs(y_test.iloc[:5].values - sample_predictions) / y_test.iloc[:5].values) * 100
average_percentage_error = percentage_errors.mean()
logging.info(f"Percentage errors per prediction: {percentage_errors.round(2)}")
logging.info(f"Average sample percentage error: {average_percentage_error:.2f}%")

# Evaluate model on the full test set
final_predictions = lin_reg.predict(X_test_prepared)
test_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
logging.info(f"\nFinal Test RMSE: {test_rmse:.2f}")

# Cross-validation for robust performance estimation
pipeline = Pipeline([
    ('preprocessor', full_pipeline),
    ('lin_reg', LinearRegression())
])

cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
cv_rmse_scores = np.sqrt(-cv_scores)
logging.info("\nCross-validation RMSE scores:")
logging.info(cv_rmse_scores.round(2))
logging.info(f"Average CV RMSE: {cv_rmse_scores.mean():.2f}")
logging.info(f"CV RMSE standard deviation: {cv_rmse_scores.std():.2f}")

# Feature importance analysis (Regression coefficients)
feature_names = num_features + list(full_pipeline.named_transformers_['cat'].get_feature_names_out(cat_features))
feature_importances = lin_reg.coef_

logging.info("\nFeature Importance (coefficients):")
for name, coef in zip(feature_names, feature_importances):
    logging.info(f"{name}: {coef:.2f}")

# Explicitly log the coefficients of OneHotEncoded states
state_feature_names = list(full_pipeline.named_transformers_['cat'].get_feature_names_out(cat_features))
logging.info("\nOneHotEncoded State Feature Coefficients:")
for name in state_feature_names:
    coef_index = feature_names.index(name)
    logging.info(f"{name}: {feature_importances[coef_index]:.2f}")

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, lin_reg.predict(X_test_prepared), edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression (with OneHotEncoder): Actual vs Predicted Profits")
plt.grid(True)
plt.savefig('data/clean_ohe_actual_vs_predicted.png')
plt.close()

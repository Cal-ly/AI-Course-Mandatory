'''
Explanation of ML Principles Illustrated:
- **Problem Framing:** Defined the task as a regression problem to predict startup profits based on expenses.
- **Data Exploration:** Identified distributions and correlations using visualizations and correlation matrices.
- **Data Preparation:** Applied OneHotEncoder to include categorical features and standard scaling for numerical features.
- **Feature Engineering:** Converted categorical features into numerical format using OneHotEncoder.
- **Model Training:** Utilized Linear Regression with scaled numerical attributes and encoded categorical attributes.
- **Performance Evaluation:** Evaluated the model using RMSE and cross-validation for robust performance estimates.
- **Visualization:** Provided clear plots for intuitive understanding and transparency.
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('data/50_Startups.csv')

# Separate features and labels
X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: StandardScaler for numerical features and OneHotEncoder for categorical features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_features = ["R&D Spend", "Administration", "Marketing Spend"]
cat_features = ["State"]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(), cat_features),
])

# Prepare the data
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict profits on a small sample
sample_predictions = lin_reg.predict(full_pipeline.transform(X_test[:5]))

# Evaluate model performance
predictions = lin_reg.predict(full_pipeline.transform(X_test))
rmse = np.sqrt(np.mean((y_test - predictions)**2))

# Cross-validation
cv_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
cv_rmse_scores = np.sqrt(-cv_scores)

# Analyze feature importance
feature_names = num_features + list(full_pipeline.named_transformers_['cat'].get_feature_names_out(cat_features))
feature_importances = lin_reg.coef_

# Display feature importance
print("\nFeature importance (coefficients):")
for name, coef in zip(feature_names, feature_importances):
    print(f"{name}: {coef:.2f}")

# Plot Actual vs. Predicted Profits
plt.scatter(y_test, predictions)
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression with OneHotEncoder: Actual vs. Predicted Profits")
plt.show()

# Save model
import joblib
joblib.dump(lin_reg, "startup_profit_predictor_onehot.pkl")

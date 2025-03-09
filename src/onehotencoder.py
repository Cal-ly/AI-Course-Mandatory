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
import io
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

# Configure logging
logging.basicConfig(filename='data/ohe_output.log', level=logging.INFO, format='%(message)s')
logging.info("---")
logging.info("Startup Profit Prediction Model")
logging.info(f"Date: {pd.Timestamp.now()}")
logging.info("---")

# Load the dataset
dataset = pd.read_csv('data/50_Startups.csv')

# Separate features and labels
X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline: StandardScaler for numerical features and OneHotEncoder for categorical features
num_features = ["R&D Spend", "Administration", "Marketing Spend"]
cat_features = ["State"]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

'''
The ColumnTransformer class is used to apply different transformations to different columns in the dataset.
In this case, we apply the num_pipeline to the numerical features and OneHotEncoder to the categorical features.
The full_pipeline combines these transformations into a single pipeline for preprocessing the data.
Mathematical transformations are applied to numerical features, while categorical features are encoded into numerical format.
In this example, the OneHotEncoder is used to convert the "State" categorical feature into numerical format.
The numerical format looks like this: [1, 0, 0] for New York, [0, 1, 0] for California, and [0, 0, 1] for Florida.
The matrix is then concatenated with the numerical features to create a combined feature matrix. It could look like this:
[[165349.2, 136897.8, 471784.1, 1, 0, 0],
 [162597.7, 151377.59, 443898.53, 0, 1, 0],
 [153441.51, 101145.55, 407934.54, 0, 0, 1]]
The combined feature matrix is then used for training the Linear Regression model.
The model learns the relationship between the features and the target variable (Profit) to make predictions.
The numerical features are scaled using StandardScaler to ensure that all features have the same scale.
'''
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", OneHotEncoder(), cat_features),
])

# Prepare the data
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, y_train)

# Predict profits on a small sample
sample_predictions = lin_reg.predict(X_test_prepared[:5])

# Evaluate model performance
logging.info("\nSample predictions:")
logging.info(f"Predictions: {sample_predictions.round(2)}")
logging.info(f"Actual values: {y_test.iloc[:5].values}")

# Calculate prediction accuracy
logging.info("\nPrediction accuracy:")
rmse_array = np.sqrt(mean_squared_error(y_test.iloc[:5].values, sample_predictions))
logging.info(f"RMSE for sample predictions: {rmse_array.round(2)}")

# Calculate percentage error for sample predictions
percentage_error = (np.abs(y_test.iloc[:5].values - sample_predictions) / y_test.iloc[:5].values) * 100
logging.info(f"Percentage error for each prediction: {percentage_error.round(3)}")
average_percentage_error = np.mean(percentage_error)
logging.info(f"Average percentage error for sample predictions: {average_percentage_error.round(3)}")

# Evaluate model using Root Mean Square Error (RMSE) on the test set
predictions_test = lin_reg.predict(X_test_prepared)
rmse = np.sqrt(np.mean((y_test - predictions_test)**2))
logging.info(f"\nTest RMSE: {rmse:.2f}")

# Cross-validation
cv_scores = cross_val_score(lin_reg, X_train_prepared, y_train, scoring="neg_mean_squared_error", cv=5)
cv_rmse_scores = np.sqrt(-cv_scores)
logging.info("\nCross-validation RMSE scores:")
logging.info(cv_rmse_scores.round(2))
logging.info(f"Average CV RMSE: {cv_rmse_scores.mean().round(2)}")

# Analyze feature importance
feature_names = num_features + list(full_pipeline.named_transformers_['cat'].get_feature_names_out(cat_features))
feature_importances = lin_reg.coef_

logging.info("\nFeature importance (coefficients):")
for name, coef in zip(feature_names, feature_importances):
    logging.info(f"{name}: {coef:.2f}")


# Plot Actual vs. Predicted Profits
plt.scatter(y_test, lin_reg.predict(X_test_prepared))
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression with OneHotEncoder: Actual vs. Predicted Profits")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.savefig('data/onehotencoder_actual_vs_predicted.png')

# Save model
import joblib
joblib.dump(lin_reg, "startup_profit_predictor_onehot.pkl")

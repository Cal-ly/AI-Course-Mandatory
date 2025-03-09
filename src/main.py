'''
Explanation of ML Principles Illustrated:
- **Problem Framing:** Defined the task as a regression problem to predict startup profits based on expenses.
- **Data Exploration:** Identified distributions and correlations using visualizations and correlation matrices.
- **Data Preparation:** Dropped irrelevant categorical attribute (`State`), applied standard scaling for numerical features.
- **Model Selection & Training:** Used a linear regression model, suitable for continuous numeric predictions.
- **Performance Measurement (RMSE):** Used as it aligns with business needs by penalizing large prediction errors.
- **Cross-validation:** Ensured the robustness of the model through repeated training-validation cycles.
- **Visualization:** Provided intuitive understanding and transparency to stakeholders through clear plots.
'''

# Import necessary libraries
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import logging

# Configure logging
logging.basicConfig(filename='data/main_output.log', level=logging.INFO, format='%(message)s')
logging.info("---")
logging.info("Startup Profit Prediction Model")
logging.info(f"Date: {pd.Timestamp.now()}")
logging.info("---")

# Load the dataset from GitHub (or local file if preferred):
dataset = pd.read_csv('data/50_Startups.csv')

# Inspect dataset structure (initial exploration)
logging.info("Dataset info:")
buffer = io.StringIO()
dataset.info(buf=buffer)
info_str = buffer.getvalue()
logging.info(info_str)
logging.info("\nDataset preview:")
logging.info(dataset.head())
logging.info("\nDataset summary statistics:")
logging.info(dataset.describe())

# Drop the categorical "State" feature as initially suggested by the assignment
dataset = dataset.drop("State", axis=1)

# Step 2: Create a stratified test set (here, simple split since data is small)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Step 3: Explore data – Understand data distribution
logging.info("\nHistograms of numerical attributes:")
dataset.hist(bins=20, figsize=(12, 8))
plt.savefig('data/main_histogram.png')

# Explore correlations with Profit (target variable)
corr_matrix = dataset.corr()
logging.info("\nCorrelation matrix with Profit:")
logging.info(corr_matrix["Profit"].sort_values(ascending=False))

# Step 4: Prepare the Data
# Separate features and labels
X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

'''
The data is split into features (X) and labels (y) to separate the input variables from the target variable.
This creates a training and test sets (20% data reserved for testing) and random state for reproducibility, 
in this case 42, which means the same split will be generated each time the code is run.
This is important to ensure consistent evaluation of the model's performance.
If the random state is not set, the data will be split differently each time, leading to varying results.
A random state of 42 is commonly used as it is the answer to the Ultimate Question of Life, the Universe, and Everything.
If random_state were 5, for example, the split would be different, and the results would vary.
'''
# Create training and test sets (20% data reserved for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Standard Scaling to numerical features to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Select and Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict profits on a small sample to visualize predictions
logging.info("\nSample predictions:")
sample_predictions = lin_reg.predict(X_test_scaled[:5])
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
predictions_test = lin_reg.predict(X_test_scaled)
rmse = np.sqrt(np.mean((y_test - predictions_test)**2))
logging.info(f"\nTest RMSE: {rmse:.2f}")

'''
The pipeline includes scaling and linear regression steps for convenience
It works by applying the same transformations to the training and validation sets in each fold
A fold is a subset of the data used for training and validation in each iteration
In this case we use 5-fold cross-validation (cv=5) to evaluate the model, meaning the data is split into 5 parts
The model is trained on 4 parts and validated on the remaining part, repeated 5 times with different splits
We gain the cross-validation set from the training set
'''

# Perform Cross-validation to estimate model generalization performance
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
cv_rmse_scores = np.sqrt(-cv_scores)

logging.info("\nCross-validation RMSE scores:")
logging.info(cv_rmse_scores.round(2))
logging.info(f"Average CV RMSE: {cv_rmse_scores.mean().round(2)}")

# Analyze feature importance (coefficients indicate the impact on Profit)
lin_reg.fit(X_train_scaled, y_train)
feature_importances = lin_reg.coef_

logging.info("\nFeature importance (coefficients):")
for name, coef in zip(X.columns, feature_importances):
    logging.info(f"{name}: {coef:.2f}")

'''
Feature importance (coefficients) is a measure of how much each feature contributes to the target variable (Profit).
The higher the coefficient, the more impact the feature has on the target variable. If it has a negative coefficient,
it means that as the feature increases, the target variable decreases. In this case, R&D Spend has the highest positive
impact on Profit, followed by Marketing Spend. Administration has a negative impact on Profit.
We can use this information to identify which features are most relevant for predicting the target variable.
Model-wise, the Linear Regression model has an RMSE of 8995.91 on the test set, which indicates the average error in
predicted Profit values. The model's generalization performance is estimated using cross-validation, with an average RMSE
of 9546.94. This provides a more robust evaluation of the model's performance on unseen data.
Comparing the results of the test set and cross-validation helps ensure the model's reliability and generalization ability.
The model has a slightly worse performance on the cross-validation set, which is expected due to the smaller training set size.
'''	

# Plot Actual vs. Predicted Profits to visually inspect prediction accuracy
plt.scatter(y_test, lin_reg.predict(X_test_scaled))
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression: Actual vs. Predicted Profits")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.savefig('data/main_actual_vs_predicted.png')

# Finally, save the trained model pipeline for future use
import joblib
joblib.dump(pipeline, "startup_profit_predictor.pkl")

'''
The "startup_profit_predictor.pkl" contains the trained model pipeline and has been saved for future use.
A .pkl file is a serialized object file that can be loaded and used to make predictions. It can be loaded using:

```python
import joblib
pipeline = joblib.load("startup_profit_predictor.pkl")
```

The loaded pipeline can then be used to make predictions on new data. For example:

```python
new_data = pd.DataFrame({"R&D Spend": [50000], "Administration": [10000], "Marketing Spend": [20000]})
new_data_scaled = scaler.transform(new_data)
prediction = pipeline.predict(new_data_scaled)
print(prediction)
```

'''
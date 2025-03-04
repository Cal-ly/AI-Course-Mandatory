import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
dataset = pd.read_csv('data/50_Startups.csv')

dataset.drop('State', axis=1, inplace=True)
dataset.drop('Administration', axis=1, inplace=True)

# Split the data into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Separate the features and the target variable
X_train = train_set.drop('Profit', axis=1)
y_train = train_set['Profit']
X_test = test_set.drop('Profit', axis=1)
y_test = test_set['Profit']

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Model Score:", model.score(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)
y_pred_round = np.round(y_pred, 2)
print("Predictions:", y_pred_round)

# Evaluate model performance
mean_squared_error = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mean_squared_error)
r2_score = model.score(X_test, y_test)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot the predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.savefig('data/predictions.png')
plt.show()


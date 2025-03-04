import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
dataset = pd.read_csv('data/50_Startups.csv')

# Check the shape of the data
print(dataset.shape)  # Check the number of rows and columns
print(dataset.dtypes) # Check the types of each column
dataset.head()        # Display the first few rows

# Make a copy of the data for exploration
data_exploration = dataset.copy()

# Check general information
data_exploration.info()

# Check missing values
print("Missing Values:\n", data_exploration.isnull().sum())

# Get summary statistics
data_exploration.describe()

# Plot histograms of the numerical columns
data_exploration.hist(figsize=(12, 8), bins=30)
plt.savefig('data/data_exploration_histogram.png')

# Plot a bar chart of the 'State' column, how many times each state appears
data_exploration['State'].value_counts().plot(kind='bar')
plt.savefig('data/state_counts.png')

# We calculate the correlation matrix between attributes
data_exploration_no_state = data_exploration.drop('State', axis=1)
corr_matrix = data_exploration_no_state.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig('data/correlation_matrix.png')

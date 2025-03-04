import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
dataset = pd.read_csv('data/50_Startups.csv')

# Check the shape of the data
print(dataset.shape)  # Check the number of rows and columns
print(dataset.dtypes) # Check the types of each column
dataset.head()        # Display the first few rows

# Split the data into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Mandatory Assignment 1

Write e.g. Python code 
```python
def method (params)
    doSomething = 1

```


## Step 1: Define the Problem

### Questions & Tasks

1. **Business Objective:** Clearly define how the model’s predictions will be used.
- To predict the profit of a startup before the profit is public, to determine whether the startup is a sound investment?

2. **Solution Usage:** Define user stories.
- *As an* investor, *I want to* get predictions on a startups profitability, *in order to* make data-driven descision.

3. **Current Solutions/Workarounds:** Not applicable.

4. **Problem Type:** Supervised learning (regression task).
- Supervised, Train once, run offline?

5. **Performance Measurement:** 
RMSE (Root Mean Square Error).

6. **Alignment with Business Goals:** Validate the selected performance measure.
We get a approximation to how profitable the startup is (+/-)

7. **Minimum Performance Requirement:** Not defined (no baseline).

8. **Comparable Problems:** Reference similar past projects.
InjuredAndKilledInTrafic-Dk 
[Linear Regression Life Expectancy](https://github.com/jpandersen61/Machine-Learning/blob/main/Linear_Regression_Lifeexpectancy_Solution.ipynb)

9. **Human Expertise:** Not available.

10. **Manual Solution:** Not applicable.

11. **Assumptions:** Document any key assumptions made.
    1. Past data is ""point" to furture trends
    2. The given Features are also sufficient to estimate profit
    3. Drop State - if not relevant? Let us compare models..

12. **Verify Assumptions:** Out of scope for now.

## Step 2: 

### Questions & Tasks

1. **List the data you need and how much you need:**
- We don't have more data

2. **Find and Document where you can get data:**
- In the file [50StartUps.csv](/data/50_Startups.csv)

3. **How much space:**
- Dataset can be held in memory

4. **Check legal obligations and get auth if nessecary**
- N/A

5. **Get Access Authorization:**
- N/A

6. **Create Workspace:**
- We created a GitHub repository, and used VSCode with Live Share, to collaborate on all the files in the repo. All documentation is held in this file [Documentation](/documentation/documentation.md)

7. **Get the data:**
- We have the data.

```python
import pandas as pd

# Load dataset
datafile = "/data/50_Startups.csv"
dataset = pd.read_csv(datafile)
```

8. **Convert the data to format, that can be manipulated:**
- We can covnert the data with `Pandas DataFrame`

9. **Ensure Sensitive information protections:**
- N/A

10. **Check size and type of data:**
Let's inspect the data structure:

```python
print(dataset.shape)  # Check the number of rows and columns
print(dataset.dtypes) # Check the types of each column
dataset.head()        # Display the first few rows
```

## **Step 3 - Data exploration**

### **Task 1 Create Copy for Data exploration:**
See [Data Exploration Script](/src/data_exploration.py)

### **Task 2 Create Notebook:**
We will not do that

### **Task 3 Study Each attribute and charateristics**
We can see, that we don't have any *missing* values, however this is because 0 has been inserted. Also we have *float64* in four of the columns and an *object* in the last.
This means that, we have to consider how to handle the feature `State`, in some way.


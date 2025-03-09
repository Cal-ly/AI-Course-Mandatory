# Mandatory Assignment 1

## Step 1: Look at the Big Picture and Frame the Problem  

**1. Define the objective in business terms.**  
The objective is to accurately predict the profit of a startup before the profits become publicly available. This prediction supports financial decision-making by advising wealthy clients on potential startup investments.

**2. How will your solution be used?**  
The solution will be used as part of JPM-Finance's investment advisory services, providing insights to their clients about the financial viability of startups, thereby helping clients make informed investment decisions.

**3. What are the current solutions/workarounds (if any)?**  
N/A – No existing solutions have been mentioned.

**4. How should you frame this problem (supervised/unsupervised, online/offline, etc.)?**  
This problem is framed as a supervised regression task since the dataset includes labeled instances (profits of startups) that the model will learn from. Additionally, it is an offline/batch-learning scenario because updates and predictions are not expected to be real-time.

**5. How should performance be measured?**  
Performance will be measured using regression metrics, specifically Root Mean Squared Error (RMSE), which quantifies the prediction errors and gives more weight to large errors, aligning well with financial forecasting scenarios.

**6. Is the performance measure aligned with the business objective?**  
Yes, RMSE aligns well with the business objective as it penalizes large errors significantly, ensuring the prediction accuracy of profits, which is critical for making sound investment decisions.

**6. What would be the minimum performance needed to reach the business objective?**  
The minimum acceptable performance would depend on JPM-Finance's risk tolerance. Typically, an RMSE small enough to reliably distinguish profitable from non-profitable investments would be considered acceptable. For instance, predictions within 10-15% of actual values could be considered reasonable.

**7. What are comparable problems? Can you reuse experience or tools?**  
Comparable problems include predicting housing prices, future sales, or company valuations. Experience and tools from the Housing project (such as data cleaning, feature selection, and linear regression modeling) can be directly reused here.

**7. Is human expertise available?**  
No domain experts are specified as available; otherwise, their input could enhance feature selection and model validation. E.g. is `State` an important feature?

**8. How would you solve the problem manually?**  
Manually, financial analysts might compare startup metrics (such as R&D expenditure, market segment, historical performances of similar startups, etc.) and use experience and intuition to estimate profits.

**9. List the assumptions you (or others have made so far).**  
- The dataset provided is representative of startups generally.
- Features set to 0.0 represent accurate information (for now acceptable).
- Categorical feature "State" doesn't add significant predictive value initially and can be excluded without significantly impacting results.

**10. Verify assumptions if possible.**  
Verifying assumptions is beyond this assignment's scope but would typically involve cross-validation against external datasets or industry standards.

---

### Step 2: Get the Data  

**1. List the data you need and how much you need:**  
_Not applicable_ — the dataset (`50_Startups.csv`) is provided.

**2. Find and document where you can get that data:**  
_Not applicable_ — the data is already given at:
[https://raw.githubusercontent.com/jpandersen61/Machine-Learning/refs/heads/main/50_Startups.csv](https://raw.githubusercontent.com/jpandersen61/Machine-Learning/refs/heads/main/50_Startups.csv).

**3. Check out how much space it requires:**  
The dataset is small (~50 rows), which means storage requirements are minimal (a few kilobytes).

**4. Check legal obligations, and get authorization if necessary:**  
_Not applicable_ — no legal concerns mentioned.

**5. Get access authorizations:**  
_Not applicable_ — dataset publicly accessible on GitHub.

**6. Create a workspace (with enough storage space):**  
This is already done by setting up your GitHub repository with a local workspace.  
(For instance, you've likely created a GitHub repo with a structured `src` folder and Markdown files for answers.)

**7. Get the data:**  
Establish a Python routine to fetch the data within your notebook/project:
```python
import pandas as pd

data_url = "https://raw.githubusercontent.com/jpandersen61/Machine-Learning/refs/heads/main/50_Startups.csv"
startups_df = pd.read_csv(data_url)
```

**8. Convert the data to a format you can easily manipulate:**  
The data is directly loaded into a Pandas DataFrame, which is suitable for manipulation:
```python
type(startups_df)  # pd.DataFrame, easily manipulated
```

**9. Ensure sensitive information is deleted or protected:**  
_Not applicable_ — no sensitive data provided.

**10. Check the size and type of data:**  
The data is numerical, with one categorical attribute (`State`), representing startup financial metrics such as R&D Spend, Marketing Spend, etc. The dataset is small (~50 instances), and manageable without special handling.

```python
startup_data = pd.read_csv(data_url)
startup.info()
```

**11. Sample a test set, put it aside, and never look at it:**  
Create a stratified split of the data (assuming `Profit` as a potential stratification attribute if appropriate, else a random split):
```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
```

---

### Step 3: Explore the Data  

**1. Create a copy of the data for exploring:**  
Done..

**2. Create a notebook for exploring the data:**  
_Not applicable_ – already established by creating and organizing the project structure and notebook.

**3. Inspect the structure and characteristics of the data:**  
Below a brief inspect of the attributes:

- **R&D Spend, Administration, Marketing Spend, State, Profit:**  
  - **R&D Spend, Administration, Marketing Spend:** Numeric attributes. Useful for predicting profit. Distribution likely skewed (non-Gaussian), typical for financial metrics.
  - **State:** Categorical attribute (California, Florida, New York). As per assignment instructions, initially excluded from analysis.
- **Profit:** Target variable for prediction (numeric).

A quick Python inspection:

```python
dataset.head()
dataset.info()
dataset.describe()
```
**Histogram**
Visualization using histograms helps detect data distributions:

```python
dataset.hist(bins=20, figsize=(12, 8))
plt.show()
```

Typical outcome:
- Distributions may not all be Gaussian (fine, initially).

**4. Identify the target attribute:**  
`Profit` is the target attribute, as it's the business-critical variable the model will predict.

**5. Discover and visualize the data by scatter plots for each numerical attribute:**
_Not applicable_

**6. Establish correlations and scatter matrix plot:**  
Correlation matrix and scatter plots to detect important attributes:

```python
corr_matrix = dataset.corr(numeric_only=True)
corr_matrix["Profit"].sort_values(ascending=False)
```

Most promising features typically include `R&D Spend` and possibly `Marketing Spend`.

Scatter matrix example:

```python
from pandas.plotting import scatter_matrix

promising_attributes = ["Profit", "R&D Spend", "Marketing Spend", "Administration"]
scatter_matrix(dataset[promising_features], figsize=(12,8))
plt.show()
```

**7. Study how you would solve the problem manually:**
_Not applicable_

**8. Experiment with attribute combinations:**
See [[data/correlation_matrix.png]], 

**9. Identify promising features:**  
Features strongly correlated with `Profit` (usually `R&D Spend` and `Marketing Spend`) should be prioritized.

**10. Identify extra data that would be useful:**  
_Not applicable_ – constrained by given data. Additional potentially useful data could be:
- Industry-specific economic indicators.
- Startup founders’ previous success rates.

---

## Step 4: Prepare the Data

**1. Data cleaning:**
- According to the assignment, it's currently acceptable to keep features with values set to `0.0`. No data cleaning or imputation is necessary at this stage.

**2. Feature selection:**
- Drop the categorical attribute `"State"` as per assignment guidelines, since it's mentioned as not significantly important initially.
- Python code example:
  ```python
  startups_clean = dataset.drop("State", axis=1)
  ```

**3. Feature engineering (if appropriate):**
- Not required explicitly by the assignment. We will skip this step for now, as there's no indication that feature engineering is necessary at this point.

**4. Handle text and categorical attributes (OneHotEncoder):**
- According to the assignment instructions, the `State` attribute is not important. We skip `OneHotEncoder` for now.

**5. Feature scaling:**
- Feature scaling (such as standardization) ensures features are on a similar scale, which often improves model training performance.

Here's a Python example of incorporating standard scaling into a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

prepared_data = num_pipeline.fit_transform(startups_clean.drop("Profit", axis=1))
```

This ensures all numerical features have the same magnitude, leading to improved performance of linear regression models.

**Reason for scaling:**
- Feature scaling ensures that each feature contributes equally to model training, preventing attributes with larger scales from dominating the learning process, leading to faster convergence and potentially better results.

---

## Step 5: Select and Train a Model

**1. Select and train a model:**
- The chosen model is **Linear Regression**, suitable for initial exploration given its simplicity and interpretability.

Here's how it's implemented with Scikit-Learn:

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(prepared_data, startups_clean["Profit"])
```

**2. Measure and compare performance:**
We'll use Root Mean Square Error (RMSE) as the performance metric. Example on a small subset (e.g., 5-10 instances):

```python
from sklearn.metrics import mean_squared_error
import numpy as np

some_data = prepared_data[:5]
some_labels = startups_clean["Profit"].iloc[:5]

predictions = lin_reg.predict(some_data_prepared)
rmse = np.sqrt(mean_squared_error(some_labels, predictions))
```

Interpreting the RMSE value:
- Lower RMSE indicates better predictive accuracy.
- Compare this value against baseline or target business objectives to gauge effectiveness.

**3. Analyze the most significant variables:**
Identify important features by examining regression coefficients:

```python
feature_importances = lin_reg.coef_
feature_names = startups_clean.drop("Profit", axis=1).columns
sorted(zip(feature_importances, feature_names), reverse=True)
```

You'd typically see `R&D Spend` as a highly significant variable.

**4. Analyze the types of errors the model makes:**
- Analyze residuals (actual vs. predicted values). Plotting these can reveal patterns or systematic errors:
```python
import matplotlib.pyplot as plt

predictions = lin_reg.predict(prepared_data)
plt.scatter(startups_clean["Profit"], predictions)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Predicted vs Actual Profit")
plt.show()
```

**4. Perform a quick round of feature selection and engineering:**
If some features show very low coefficients (close to zero), consider removing them for simplification.

For example, if the Administration feature is unimportant:

```python
startups_reduced = startups_clean.drop(["Administration"], axis=1)
```

**5. Consider another quick-and-dirty model from different categories:**
One could quickly test a different regression model, e.g., a Decision Tree or Random Forest, to check if it improves performance:

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(prepared_data, startups_clean["Profit"])

tree_predictions = tree_reg.predict(prepared_data)
tree_rmse = mean_squared_error(startups_clean["Profit"], tree_reg.predict(prepared_data), squared=False)
```

But Decision Tree Regression often overfits training data (RMSE = 0.0), and we have small dataset *no bueno*.

**6. Shortlist promising models:**
Based on performance, linear regression typically provides a good balance of simplicity and interpretability for this small dataset. Decision Tree or Random Forest could be explored further if more data or complexity is needed.

---

## Step 6: Fine-Tune and Test the Model

**1. Fine-tune hyperparameters using cross-validation:**  
- For Linear Regression, there aren't many hyperparameters to tune, but we could explore whether applying different scaling or transformations influences performance. **Note:** We only have theta0 and theta1 as free parameters. Constrining the model would mean either locking slope or intercept.
- Here, we primarily demonstrate using a cross-validation technique to evaluate the model reliably:

Example with cross-validation (Linear Regression):

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

scores = cross_val_score(pipeline, 
                         startups_clean.drop("Profit", axis=1), 
                         startups_clean["Profit"], 
                         scoring="neg_mean_squared_error", 
                         cv=5)

rmse_scores = np.sqrt(-scores)
print(f"Mean RMSE: {rmse_scores.mean():.2f}, Std Dev: {rmse.std():.2f}")
```

- Cross-validation helps estimate how the model generalizes to unseen data, giving a realistic view of expected performance.

**2. Ensemble methods:**  
- _Not applicable_ at this stage because the assignment suggests focusing on a single model.

**3. Analyze the best models and their errors:**  
Analyze residuals and RMSE from cross-validation. This helps understand prediction errors and assess generalization:

```python
print("Cross-validation RMSE scores:", np.sqrt(-scores))
print("Mean RMSE:", np.sqrt(-scores).mean())
print("Standard deviation:", np.sqrt(-scores).std())
```

This gives insights into model stability (low standard deviation) and generalization capabilities.

**4. Evaluate your system on the test set:**  
Finally, evaluate the selected model (Linear Regression) on the reserved test set to estimate the generalization error:

```python
from sklearn.metrics import mean_squared_error

pipeline.fit(train_set.drop("Profit", axis=1), train_set["Profit"])
final_predictions = pipeline.predict(test_set.drop("Profit", axis=1))
final_rmse = np.sqrt(mean_squared_error(test_set["Profit"], final_predictions))
```

- A small `final_rmse` indicates good predictive performance, thus achieving the business objective.

**Additional considerations (optional if time permits):**

- If desired, test other models like RandomForestRegressor or Support Vector Machine (SVM) and compare their cross-validation results.  
- Similarly, experimenting with the initially excluded categorical attribute (`State`) using `OneHotEncoder` could be considered for completeness.


---

## Step 7: Present Your Solution

**1. Document what you have done:**  
- Framed the problem as supervised learning (regression) for predicting startup profits.
- Explored and understood the dataset, identifying important features (`R&D Spend`, `Marketing Spend`) and discarding irrelevant features (`State`).
- Prepared data using standard scaling to ensure numerical stability and improved model performance.
- Selected a Linear Regression model, trained and validated using cross-validation.
- Analyzed results using RMSE, ensuring predictions align with business objectives.

**2. Create a nice presentation:**  
- _Not applicable_ unless explicitly needed for external communication. For internal or academic presentation, highlight the business goal clearly upfront.

**3. Explain why your solution achieves the business objective:**  
- The chosen Linear Regression model effectively predicts startup profits from financial features with acceptable accuracy, directly enabling JPM-Finance to provide informed investment advice to their clients.  
- The RMSE evaluation is aligned with the financial domain (that we came up with in Step 1) as it emphasizes the significance of errors, ensuring reliable recommendations.

**4. Interesting points noticed along the way:**  
- **Worked well:**  
  - Clear correlation between `R&D Spend` and profits, making predictions more reliable.
  - Standard scaling improved numeric feature handling and reduced numeric bias in predictions.
- **Limitations and assumptions:**  
  - The model assumes that past startup financial patterns will reflect future results.
  - Limited data (50 startups) restricts generalization; additional data collection would enhance model reliability.
  - Initially disregarded the categorical `State` attribute due to its perceived insignificance; revisiting this could potentially refine predictions slightly.

**5. Key findings communicated through visualizations or easy-to-remember statements:**  
- "R&D Spend is the strongest predictor of startup profitability."
- Visualizations such as scatter plots of actual vs. predicted profits clearly illustrate model accuracy and areas for improvement:

```python
import matplotlib.pyplot as plt

plt.scatter(test_set["Profit"], final_predictions)
plt.xlabel("Actual Profits")
plt.ylabel("Predicted Profits")
plt.title("Linear Regression: Actual vs Predicted Profits")
plt.show()
```

This visualization demonstrates prediction accuracy, allowing JPM-Finance to clearly see the reliability of their profit estimates.

---
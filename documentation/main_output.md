# Analysis of Model Training Output Log:

### Dataset Overview:
- **Total Instances:** 50
- **Features:**  
  - Numeric: `R&D Spend`, `Administration`, `Marketing Spend`, and `Profit` (target).
  - Categorical: `State` (ignored as per instructions).

### Correlation Insights:
- Strongest positive correlation with `Profit`:  
  - **`R&D Spend`** (correlation = **highest**, crucial predictor).
- Weaker positive correlation with `Profit`:  
  - `Marketing Spend` (moderately strong).
- Very weak correlation with `Administration`.

### Model Predictions (Linear Regression):
| Instance | Actual Profit | Predicted Profit | Percentage Error |
|----------|---------------|------------------|------------------|
| 1        | 134307.35     | 134644.89        | 5.662%           |
| 2        | 81005.76      | 84894.94         | 4.801%           |
| 3        | 99937.59      | 96989.42         | 4.045%           |
| 4        | 64926.08      | 46596.20         | 28.377%          |
| 5        | 125370.37     | 128090.83        | 5.030%           |

- **Average RMSE on test predictions:** `9247.92`
- **Average percentage error:** approx. `5%`

This indicates the model predicts startup profit with reasonable accuracy on the test set, though performance varies slightly between predictions.

### Cross-Validation Evaluation:
Cross-validation (CV) provides a robust estimate of model performance:

- **CV RMSE scores:** `[9247.92, 7747.92, 14741.44]` with an average CV RMSE of `9546.94`.

The cross-validation shows slightly higher error (`9546.94`) than the test RMSE (`9247.92`). This difference, while small, suggests the model's generalization capability is stable, although there’s some variance across different folds—particularly indicated by the high value (`14741.44`) in one fold.

### Feature Importance Analysis (Regression Coefficients):
- **R&D Spend:** `38014.74`  
  This feature strongly positively impacts profit. Higher investment in R&D significantly increases expected profits.
  
- **Marketing Spend:** `3543.39`  
  Positive but less influential than R&D Spend. Increased marketing expenditure tends to boost profits moderately.

- **Administration:** `-1841.48`  
  Negatively impacts predicted profit slightly, possibly indicating administrative costs don't strongly drive profit and higher costs might be inversely related.

### Overall Conclusion:
- **Strong predictor:** R&D Spend.
- **Less important:** Administration (minimal negative influence).
- **Acceptable predictive performance:**  
  RMSE (~9,247.92) and percentage errors (~5%) indicate good, though not perfect accuracy for practical business purposes. This performance level is acceptable, especially considering the small dataset size (50 instances).

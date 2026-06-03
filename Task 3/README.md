# Task 3: Linear Regression Modeling

This repository contains the deliverables for Task 3 of the AI & ML Internship. The core objective was to implement, evaluate, and interpret simple and multiple linear regression architectures to predict housing valuations using structural and categorical features from a kaggle dataset.

##  Core Analytical Steps

1. **Feature Engineering & Preprocessing:** Executed algorithmic categorical conversion using automated One-Hot Encoding via `pd.get_dummies()` to transform structural text variables into numerical indicators without matrix loss.
2. **Data Splitting Architecture:** Segregated the processed feature matrices into distinct training and testing subsets using an 80/20 division split to secure reliable validation bounds.
3. **Model Fitting & Optimization:** Initialized and fitted an ordinary least squares Linear Regression pipeline using `sklearn.linear_model` to model underlying pricing dynamics.
4. **Performance Matrix Diagnostics:** Evaluated predictive precision using a multi-metric framework tracking Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R^2 Score.
5. **Coefficient & Visual Interpretation:** Isolated the final model weights (slopes and intercept) and projected actual vs. predicted values onto a scatter plane to assess global model fit.

---

## 📈 Visual Discoveries & Inferences

### 1. Pricing Drivers & Weight Isolation
* **Inference:** Mathematical coefficient checks confirmed that specific structural features significantly dominate price determination. For example, additions like air conditioning systems and premium furnishing configurations yield high positive weights, acting as primary valuation drivers.

### 2. Error Profile & Fit Assessment
* **Inference:** The evaluation metrics and scatter plot distribution show that while the regression pipeline successfully catches the central linear data trend, residual variances spread wider at ultra-high price points, suggesting potential non-linear relationships or a need for advanced regularized models down the line.

---

## Folder Deliverables
* `task3_linear_regression.ipynb` — Complete interactive Jupyter Notebook hosting step-by-step model training, preprocessing blocks, metric calculations, and documentation notes.
* `Linear Regression.png` — Exported data visualization asset tracking true valuations against model predictions.
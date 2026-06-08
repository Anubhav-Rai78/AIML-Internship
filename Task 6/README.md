# Task 6: K-Nearest Neighbors (KNN) Classification

This repository contains the deliverables for Task 6 of the AI & ML Internship. The core objective was to implement an instance-based learning algorithm (KNN), validate the necessity of feature scaling, and map spatial decision boundaries using proximity-based classification.

##  Core Analytical Steps

1. **Euclidean Scaling Preprocessing:** Applied `StandardScaler` to uniformize feature variances. Because KNN relies purely on spatial distance mathematics, raw scales (e.g., millimeter variations vs. centimeter variations) were standardized to prevent magnitude dominance.
2. **Algorithm Implementation:** Initialized a base `KNeighborsClassifier` utilizing Euclidean distance metrics to map nearest spatial neighbors and execute majority-vote classifications.
3. **Hyperparameter Optimization (Elbow Method):** Executed a programmatic loop to test K values ranging from 1 to 20. Tracked the variance between training accuracy and testing accuracy to identify the exact threshold where the model transitions from overfitting (capturing noise) to underfitting (losing boundary resolution).
4. **Evaluation Matrix:** Calculated raw predictive accuracy and generated a categorical Confusion Matrix to verify performance parity across distinct multi-class targets.
5. **Spatial Boundary Visualization:** Reduced the dimensional space to 2D parameters and plotted a `contourf` meshgrid to visually track where the algorithmic boundaries shift between classes.

---

## 📈 Visual Discoveries & Inferences

### 1. The Impact of $K$ Selection
* **Inference:** The K-Value tuning chart clearly demonstrated that $K=1$ achieves 100% training accuracy but represents a critically overfit state. The test accuracy stabilized optimally in the K=3 to K=7 range. Increasing K too far beyond that began to degrade testing accuracy as the local neighborhood boundaries became too generalized.

### 2. Decision Boundary Dynamics
* **Inference:** The 2D contour plot provided clear spatial evidence of how KNN operates. Unlike Logistic Regression, which draws rigid, straight lines, the KNN decision boundaries are highly non-linear and perfectly contour to the localized density of the scaled data points.

---

##  Folder Deliverables
* `task6_knn.ipynb` — Complete interactive Jupyter Notebook hosting scaling pipelines, algorithmic tuning loops, and boundary visualizations.
* `KNN_Elbow_Curve.png` & `KNN_Decision_Boundary.png` — Exported graphical assets detailing the hyperparameter tuning curve and spatial target boundaries.
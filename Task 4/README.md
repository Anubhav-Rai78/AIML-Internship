# Task 4: Classification with Logistic Regression

This repository contains the deliverables for Task 4 of the AI & ML Internship. The core objective was to construct, evaluate, and fine-tune a binary classification pipeline using Logistic Regression to predict target classes from raw diagnostic measurements.

##  Core Analytical Steps

1. **Target Isolation & Mapping:** Converted raw text target identifiers into categorical binary outcomes (0 and 1) to align the targets with logistic mathematical boundaries.
2. **Feature Scaling Optimization:** Implemented feature standardization utilizing `StandardScaler` to uniformize input variances, preventing multi-scale features from skewing the gradient optimization phase.
3. **Discriminative Pipeline Fitting:** Initialized and fit a standard Logistic Regression classification algorithm using `sklearn.linear_model`.
4. **Multi-Threshold Matrix Diagnostics:** Generated a granular evaluation suite extracting a full Confusion Matrix heatmap, Precision, Recall, and F1-scores to track edge error rates.
5. **Discriminative Sensitivity Mapping:** Mapped the True Positive vs. False Positive tradeoff across all potential operational boundaries using the ROC-AUC Curve metric space.

---

## 📈 Visual Discoveries & Inferences

### 1. Diagnostic Matrix Breakdown
* **Inference:** The evaluation metrics indicate a robust model fit with exceptional precision and recall scores across both benign and malignant targets. The confusion matrix indicates that feature normalization effectively isolated the boundaries without generating significant overlapping error risks.

### 2. Operational Threshold Balancing
* **Inference:** Plotting the ROC-AUC profile yielded an exceptional score near 1.0, proving high class-separation reliability. Testing custom cutoff parameters highlighted that lowering the activation threshold below 0.5 can significantly compress False Negative counts, maximizing clinical sensitivity without destroying global precision thresholds.

---

##  Folder Deliverables
* `task4_logistic_regression.ipynb` — Complete interactive Jupyter Notebook hosting step-by-step feature transformations, algorithm fitting, evaluation metrics, and documentation blocks.
* `Confusion_Matrix_Heatmap.png` & `ROC_Curve.png` — Exported visual diagnostic assets tracking true class alignments against pipeline classification outputs.
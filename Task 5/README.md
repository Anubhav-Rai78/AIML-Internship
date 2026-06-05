# Task 5: Decision Trees & Random Forests

This repository contains the deliverables for Task 5 of the AI & ML Internship. The core objective was to construct, evaluate, and contrast single-tree architectures against ensemble bagging techniques (Random Forests) to predict target classifications using a clinical dataset.

##  Core Analytical Steps

1. **Overfitting Diagnostics:** Initialized a fully-grown Decision Tree to intentionally overfit the training matrix, contrasting its performance against a pruned tree (max depth tuned) to demonstrate variance control.
2. **Algorithmic Visualization:** Mapped the pruned decision logic using `sklearn.tree.plot_tree` to visually isolate root features, Gini impurity drops, and conditional split thresholds.
3. **Ensemble Implementation (Bagging):** Upgraded the architecture to a Random Forest Classifier, leveraging bootstrap aggregating to neutralize individual tree bias and stabilize predictive variance.
4. **Cross-Validation Rigor:** Passed the Random Forest pipeline through a 5-Fold Stratified Cross-Validation matrix to ensure accuracy metrics were robust across multiple data permutations, rather than a lucky train-test split.
5. **Feature Importance Extraction:** Extracted and ranked the internal node impurities to quantify and plot exactly which clinical features drove the final classification targets.

---

## 📈 Visual Discoveries & Inferences

### 1. The Overfitting Trap
* **Inference:** The fully unconstrained decision tree achieved near-perfect accuracy on the training matrix but suffered a drop on testing data. Imposing a `max_depth` restriction explicitly forced the model to generalize, trading a minor loss in training accuracy for a more robust testing score.

### 2. Ensemble Superiority & Feature Dominance
* **Inference:** Transitioning from a single tree to a Random Forest stabilized the performance variance significantly. Furthermore, the Feature Importance bar chart explicitly isolated the top drivers (e.g., specific clinical markers like chest pain type or max heart rate) heavily outweighing other noise variables in the dataset, providing actionable clinical interpretability.

---

##  Folder Deliverables
* `task5_tree_ensembles.ipynb` — Complete interactive Jupyter Notebook hosting tree visualizations, forest modeling, hyperparameter pruning, and cross-validation code.
* `Decision_Tree_Visualization.png` & `Feature_Importances.png` — Exported graphical assets detailing the algorithm's conditional logic and feature weights.
# Task 2: Exploratory Data Analysis (EDA)

This repository contains the deliverables for Task 2 of the AI & ML Internship. The core objective was to perform deep-dive statistical analysis and data visualization on a raw dataset to uncover underlying patterns, track structural integrity, and isolate key behavioral drivers before applying machine learning workflows.

##  Core Analytical Steps

1. **Descriptive Statistics:** Executed mathematical matrix breakdowns using `.describe()` to extract feature means, medians, variances, and standard deviations across passenger profiles.
2. **Missing Data Diagnostics:** Visualized missing record distributions using tracking heatmaps and quantified missing data percentages to isolate structural holes.
3. **Distribution & Outlier Mapping:** Evaluated numerical feature shapes (Age, Fare) using Kernel Density Estimates (KDE) and Boxplots to catch skewness and extreme data anomalies.
4. **Bivariate Feature Inferences:** Explored cross-feature connections to isolate major target drivers (e.g., assessing socio-economic and gender impacts on survival rates).
5. **Linear Correlation Checks:** Plotted a comprehensive Pearson Correlation Heatmap across numeric variables to intercept multicollinearity risks early.

---

## 📈 Visual Discoveries & Inferences

### 1. Data Integrity Scan
* **Inference:** The `Cabin` attribute contains a fatal vacancy rate (>77%), requiring complete deletion during preprocessing. The `Age` column displays moderate operational gaps (~19.9%) suitable for median imputation.

### 2. Behavioral Patterns & Survival Drivers
* **Inference:** A massive demographic trend is confirmed: female passengers achieved a starkly superior survival probability rate compared to males (validating historical "Women and Children First" protocols). Additionally, 3rd Class passengers suffered the highest absolute casualty rates, pointing to socioeconomic position as a primary survival weight.

---

##  Folder Deliverables
* `task2_eda.ipynb` — Complete interactive Jupyter Notebook hosting step-by-step statistical calculations, visualization grids, and documentation notes.
* `Categorical Trends & Survival Drivers.png` & `Feature Correlation Matrix.png` — Exported data visualization assets tracking raw dataset health.
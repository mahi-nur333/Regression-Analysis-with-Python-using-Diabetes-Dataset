# Diabetes Prediction using Logistic Regression

## Project Overview
This project predicts whether a patient has diabetes based on various
health-related features using **Logistic Regression** and **Random Forest**
classification models. The dataset is the **Pima Indians Diabetes Database**,
containing health metrics like glucose levels, BMI, blood pressure, insulin
levels, and age.

---

## Dataset
| Property | Details |
|---|---|
| Source | Pima Indians Diabetes Database |
| Total Records | 768 individuals |
| Features | 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) |
| Target | `Outcome` — 0 = No Diabetes, 1 = Diabetes |

---

## Tools & Libraries
| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models and evaluation |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Project Steps

1. **Setup & Imports** — Load all required libraries
2. **Load Dataset** — Read CSV using pandas
3. **Exploratory Data Analysis (EDA)** — Inspect data, correlation heatmap
4. **Data Preprocessing** — Handle missing values (`SimpleImputer`), scale features (`StandardScaler`)
5. **Train-Test Split** — 80/20 split with `random_state=42`
6. **Model Training** — Train Logistic Regression (`max_iter=1000`)
7. **Prediction & Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
8. **Random Forest Comparison** — Compare Logistic Regression vs Random Forest
9. **Cross-Validation** — 5-fold CV with accuracy scoring
10. **ROC-AUC Curve** — Visualize model discrimination ability
11. **Confusion Matrix Heatmap** — Visual breakdown of predictions

---

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Test Accuracy | ~78% | ~77% |
| Cross-Val Accuracy (mean) | ~77% | — |
| AUC-ROC | ~0.84 | — |

> ⚠️ Update the values above with your actual output after running the notebook.

---

## Key Visualizations
- 📊 Correlation Heatmap (EDA)
- 📈 ROC Curve with AUC Score
- 🔲 Confusion Matrix Heatmap

---

## Why Logistic Regression for this Problem?
The target variable `Outcome` is **binary (0 or 1)**, making this a
**classification problem** — not regression. Logistic Regression is ideal because:
- It is designed for binary classification
- It outputs probability scores for each class
- It is interpretable and computationally efficient
- It performs well on tabular medical data with proper scaling

---

## Author
**Mahinur Rahman Pamel**

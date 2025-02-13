# Diabetes Prediction using Linear Regression

## Project Overview
This project aims to predict whether a patient has diabetes based on various health-related features using a **Linear Regression** model. The dataset used is the **Pima Indians Diabetes Database**, which includes information about pregnancies, glucose levels, blood pressure, insulin levels, BMI, diabetes pedigree function, and age. The objective is to build a predictive model and evaluate its performance.

## Project Structure
- **Dataset**: The dataset used in this project contains information on 768 individuals, with 8 features and 1 target variable (Outcome).
- **Tools & Libraries**: 
  - Python (v3.x)
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning algorithms and model evaluation
  - `matplotlib` and `seaborn` for data visualization

## Requirements
To run the project, you need to install the following libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Steps

### 1. Setup and Imports
We begin by importing necessary libraries like `numpy`, `pandas`, `sklearn`, and visualization libraries `matplotlib` and `seaborn`.

### 2. Load and Explore the Dataset
The dataset is loaded into a DataFrame using `pandas`. An exploratory data analysis (EDA) is conducted, and we check for any categorical variables that need conversion to numeric.

### 3. Data Preprocessing
Missing values are handled using `SimpleImputer` from `sklearn`, which fills missing values with the mean. Features are scaled using `StandardScaler` to ensure that all features contribute equally to the model.

### 4. Train-Test Split
The data is split into training and testing sets using an 80/20 split, ensuring that the model can be tested on unseen data.

### 5. Model Selection and Training
A **Linear Regression** model is selected and trained using the training set. The model's performance is evaluated on the test set.

### 6. Prediction and Evaluation
The model's predictions are compared with actual values, and metrics like **Mean Squared Error (MSE)** and **R² Score** are calculated to evaluate the model's performance.

### 7. Cross-Validation
To further evaluate the model's robustness, **5-fold cross-validation** is performed, providing an average R² score to assess model stability.

### 8. Visualization of Results
A scatter plot is generated to visually compare actual vs predicted values.

## Results

- **Mean Squared Error (MSE)**: 0.171
- **R² Score**: 0.255
- **Cross-Validation R² Scores**: [0.2568, 0.2576, 0.2907, 0.2811, 0.2941]
- **Mean Cross-Validation R² Score**: 0.276

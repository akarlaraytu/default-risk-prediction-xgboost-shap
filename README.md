# Credit Default Risk Prediction: An Explainable AI Approach with XGBoost and SHAP

## Project Overview

This project aims to develop a robust machine learning model for predicting credit default risk in a financial institution. Using a widely recognized credit card client dataset, an **XGBoost** classifier was trained to identify customers at high risk of default. A key focus of this project was to provide model interpretability using **SHAP (SHapley Additive exPlanations)**, allowing us to understand the reasoning behind the model's predictions.

## Key Methodology & Technologies

* **Data Analysis & Preprocessing:** Handling of categorical variables with one-hot encoding, and preparing the dataset for model training.
* **Model Development:** The dataset was split into training and testing sets using `stratify=y` to handle class imbalance.
* **Algorithm:** **XGBoost Classifier**
* **Evaluation Metrics:** The model's performance was evaluated using `Accuracy`, `Precision`, `Recall`, `F1-Score`, and `ROC AUC Score`.
* **Model Explainability:** **SHAP** library was used to analyze both global and local feature importance.

## Project Outcomes & Insights

* **Reliable Performance:** The model achieved a strong **81.6% accuracy** and a **0.7712 ROC AUC score**, demonstrating its effectiveness in predicting credit risk.
* **Most Influential Features:** The SHAP analysis revealed that a client's recent payment status (`PAY_0`, `PAY_2`) is the most significant factor in predicting default risk, confirming that past behavior is a powerful predictor.
* **Explainable Decisions:** With SHAP's **Force Plot**, we can now explain the specific factors that drive each individual prediction, offering a transparent and data-driven approach to credit decisions.

## How to Run the Project

This project was developed using Google Colab. To run the code:

1.  Clone this repository to your local machine.
2.  Open the `.ipynb` file in a Google Colab notebook.
3.  Install the required libraries: `!pip install pandas scikit-learn xgboost shap matplotlib seaborn`
4.  Run the notebook cells sequentially to reproduce the analysis and results.

---

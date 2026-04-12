# 💳 Credit Risk Default Prediction Model

This project predicts **loan default probability** for credit applicants using **Machine Learning**, with a focus on handling class imbalance and model interpretability.  
It includes an interactive **Streamlit web application** for real-time default risk assessment.

---

## 📌 Project Overview

A predictive model to assess credit risk by estimating the likelihood of a borrower defaulting on a loan. The project leverages various borrower attributes and bureau data to build a robust classification model, with special attention to recall for the minority (default) class.

**Key Objectives:**
- Identify high-risk loan applicants
- Minimize false negatives (missed defaults)
- Provide interpretable risk scores
- Deploy an interactive web application for real-time predictions

---

## 🧠 Machine Learning Approach

- **Problem Type:** Binary Classification
- **Final Model Used:** Logistic Regression (after hyperparameter tuning with Optuna)
- **Target Variable:** `default` (1 = Default, 0 = Non-Default)
- **Class Imbalance Handling:** SMOTE Tomek (combines over-sampling and under-sampling)
- **Feature Selection:** Information Value (IV) and Variance Inflation Factor (VIF)
- **Evaluation Metrics:**  
  - **Recall** (Primary metric for default detection)  
  - Precision, F1-Score  
  - ROC-AUC Score  
  - Gini Coefficient  
  - KS Statistic & Rank Ordering (Decile Analysis)

---

## 📊 Features Used

### Numeric Features
| Feature | Description |
|---------|-------------|
| `age` | Age of the applicant |
| `number_of_dependants` | Number of dependants |
| `years_at_current_address` | Years living at current address |
| `loan_tenure_months` | Loan tenure in months |
| `bank_balance_at_application` | Bank balance at time of application |
| `number_of_open_accounts` | Number of open credit accounts |
| `number_of_closed_accounts` | Number of closed credit accounts |
| `enquiry_count` | Number of credit enquiries |
| `credit_utilization_ratio` | Credit utilization percentage |

### Engineered Features
| Feature | Formula |
|---------|---------|
| `loan_to_income` | `loan_amount / income` |
| `delinquency_ratio` | `(delinquent_months / total_loan_months) * 100` |
| `avg_dpd_per_delinquency` | `total_dpd / delinquent_months` (handles zero division) |

### Categorical Features
- `residence_type` (Owned, Mortgage, Rented)
- `loan_purpose` (Auto, Education, Home, Personal)
- `loan_type` (Secured, Unsecured)

---

## 🔍 Key Insights from EDA

- Higher `loan_tenure_months`, `delinquent_months`, `total_dpd`, and `credit_utilization_ratio` are strong predictors of default.
- `loan_to_income` ratio shows that higher ratios correlate with higher default likelihood.
- `delinquency_ratio` and `avg_dpd_per_delinquency` are powerful engineered features for risk assessment.
- Categorical features like `loan_purpose` and `residence_type` also show significant predictive power (IV > 0.02).

---

## 📈 Model Performance Summary

| Metric              | Score   |
|---------------------|---------|
| Accuracy            | 93%     |
| Recall (Default)    | 94%     |
| Precision (Default) | 55%     |
| F1-Score (Default)  | 0.70    |
| ROC-AUC             | 0.984   |
| Gini Coefficient    | 0.967   |
| KS Statistic        | 85.9%   |

> **Rank Ordering:** Excellent separation between high-risk and low-risk applicants, with most defaults captured in the top deciles.

---

## 🗂 Project Structure

```text
ml-project-credit-risk-model/
│
├── artifacts/
│   └── model_data.joblib          # Trained model + scaler + feature columns
│
├── main.py                         # Streamlit application
├── prediction.py                   # Backend prediction logic
├── requirements.txt                # App dependencies
│
├── notebooks/
│   └── credit_risk_model_classification.ipynb   # Full EDA, feature engineering & model training
│
├── data/
│   └── bureau_data.csv             # Bureau data (credit history)
│   └── customers.csv               # Customer demographic data
│   └── loans.csv                   # Loan application data
│
├── README.md
└── .gitignore

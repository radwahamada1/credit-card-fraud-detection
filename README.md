# 💳 Credit Card Fraud Detection

An end-to-end **Machine Learning pipeline** for detecting fraudulent credit card transactions in a highly imbalanced financial dataset.

The project focuses on **model selection, class imbalance, and the Precision–Recall trade-off**, with each modeling approach evaluated based on both performance and practical business impact.

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

---

## 🎯 Problem

Fraud detection is an extremely imbalanced classification problem.

Only **0.172%** of the transactions in the dataset are fraudulent, meaning a model could achieve very high accuracy simply by predicting most transactions as legitimate.

Therefore, **Accuracy is not a reliable metric** for this problem.

The project focuses on:

- **Precision** → How many flagged transactions are actually fraud?
- **Recall** → How many actual fraud cases were detected?
- **F1-Score** → Balance between Precision and Recall.

---

## 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

- **284,807** transactions
- **0.172%** fraudulent transactions
- **99.828%** legitimate transactions
- Highly imbalanced binary classification problem

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/radwahamada/credit-card-dataset)

---

## 🔄 Modeling Process

### 1. Logistic Regression — Baseline

Logistic Regression was used as a **simple linear baseline** to establish a reference point before moving to more complex models.

`class_weight='balanced'` was applied because of the severe class imbalance.

**Result:**

- Precision: **0.06**
- Recall: **0.87**
- F1-Score: **0.11**
- False Positives: **1,389**

Although recall was high, the model produced too many false positives.

**Conclusion:**  
The model detected fraud aggressively but generated too many false alarms, making it impractical for this use case.

---

### 2. Random Forest — Non-Linear Patterns

Random Forest was introduced to capture **non-linear relationships and feature interactions** that Logistic Regression cannot model effectively.

**Result:**

- Precision: **0.97**
- Recall: **0.74**
- F1-Score: **0.84**

**Why Random Forest?**

- Handles non-linear relationships.
- Captures feature interactions.
- Performs well on tabular data.
- Requires less feature engineering than linear models.

However, it still missed some fraudulent transactions.

---

### 3. Decision Threshold Tuning

Instead of changing the model itself, the classification threshold was reduced from the default **0.50 to 0.30**.

The goal was to classify more uncertain transactions as fraud and improve recall.

**Result:**

- Precision: **0.89**
- Recall: **0.75**
- F1-Score: **0.81**

The small recall improvement demonstrated that simply changing the threshold could not solve all classification errors.

**Key Insight:**  
The decision threshold controls the **Precision–Recall trade-off** and should be selected according to the business cost of false positives and false negatives.

---

### 4. SMOTE — Handling Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** was tested to create synthetic examples of fraudulent transactions instead of simply duplicating existing ones.

**Result:**

- Precision: **0.91**
- Recall: **0.76**
- F1-Score: **0.83**

SMOTE produced only a marginal improvement compared with the original Random Forest.

**Conclusion:**  
Oversampling was not enough to significantly improve fraud detection in this dataset.

---

### 5. XGBoost — Advanced Non-Linear Modeling

XGBoost was introduced to build a stronger ensemble model capable of learning complex non-linear patterns.

Initial tuning with `scale_pos_weight` demonstrated how changing the cost assigned to fraud cases affects the Precision–Recall trade-off.

---

## 🏆 Final Model — Cost-Sensitive XGBoost

The final approach used **XGBoost with cost-sensitive learning**, using `scale_pos_weight` based on the class imbalance ratio.

Instead of generating synthetic fraud examples, the model directly assigns **higher training cost to misclassifying fraudulent transactions**.

### Final Performance

| Metric | Fraud Class |
|---|---:|
| Precision | **0.81** |
| Recall | **0.80** |
| F1-Score | **0.80** |

### Why XGBoost?

XGBoost was selected because it:

- Captures complex non-linear relationships.
- Handles feature interactions effectively.
- Performs strongly on structured/tabular data.
- Supports cost-sensitive learning.
- Provides a strong balance between Precision and Recall.

---

## 📈 Model Comparison

| Model | Strategy | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Logistic Regression | Balanced Class Weight | 0.06 | **0.87** | 0.11 |
| Random Forest | Default | **0.97** | 0.74 | **0.84** |
| Random Forest | Threshold = 0.30 | 0.89 | 0.75 | 0.81 |
| Random Forest | SMOTE | 0.91 | 0.76 | 0.83 |
| **XGBoost** | **Cost-Sensitive** | **0.81** | **0.80** | **0.80** |

The final model was selected based on the practical balance between **detecting fraud** and **limiting false alarms**, rather than optimizing a single metric.

---

## 🧠 Key Takeaways

- **Accuracy is misleading** for extreme class imbalance.
- A simple baseline helps determine whether added complexity provides real value.
- **Random Forest** improved precision by modeling non-linear relationships.
- **Threshold tuning** provides direct control over the Precision–Recall trade-off.
- **SMOTE** did not provide a significant improvement in this dataset.
- **Cost-sensitive XGBoost** provided a strong balance between fraud detection and false alarms.
- Model selection should consider **business consequences**, not metrics alone.

---

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_random_forest_and_threshold_tuning.ipynb
│   └── 04_advanced_xgboost_and_smote.ipynb
│
├── README.md
└── requirements.txt

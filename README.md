# 🏦 AI-Based Loan Approval Prediction

An end-to-end machine learning project that predicts whether a loan application should be **approved or denied**, based on applicant details. The project covers data cleaning, exploratory data analysis, model training, evaluation, and a live **Streamlit web application**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Pipeline Steps](#-pipeline-steps)
- [Models Compared](#-models-compared)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Findings](#-key-findings)
- [Streamlit App](#-streamlit-app)
- [Technologies Used](#-technologies-used)

---

## 🔍 Project Overview

Manual loan screening is slow, inconsistent, and prone to human bias. This project builds an **AI-powered loan approval decision system** that automates credit risk assessment using historical loan application data.

**Goal:** Given an applicant's personal, financial, and property information, predict whether their loan should be **Approved (1)** or **Denied (0)**.

---

## 📂 Dataset

**File:** `LoanApprovalPrediction.csv`

| Property         | Value                        |
|------------------|------------------------------|
| Total Records    | 614                          |
| Raw Columns      | 13                           |
| Features Used    | 11 (after dropping Loan_ID)  |
| Target Variable  | `Loan_Status` (Y / N)        |
| Approval Rate    | 68.7% Approved / 31.3% Denied|

### Feature Description

| Feature             | Type        | Description                              |
|---------------------|-------------|------------------------------------------|
| `Loan_ID`           | ID          | Unique identifier — dropped during cleaning |
| `Gender`            | Categorical | Male / Female                            |
| `Married`           | Categorical | Yes / No                                 |
| `Dependents`        | Ordinal     | Number of dependents: 0, 1, 2, 3+        |
| `Education`         | Categorical | Graduate / Not Graduate                  |
| `Self_Employed`     | Categorical | Yes / No                                 |
| `ApplicantIncome`   | Numerical   | Primary applicant's monthly income       |
| `CoapplicantIncome` | Numerical   | Co-applicant's monthly income            |
| `LoanAmount`        | Numerical   | Requested loan amount (thousands)        |
| `Loan_Amount_Term`  | Numerical   | Repayment period in months               |
| `Credit_History`    | Binary      | 1 = meets guidelines, 0 = does not       |
| `Property_Area`     | Categorical | Urban / Semiurban / Rural                |
| `Loan_Status`       | **Target**  | Y = Approved → 1, N = Denied → 0         |

---

## 🗂 Project Structure

```
loan_approval_project/
│
├── LoanApprovalPrediction.csv        # Raw dataset
├── loan_cleaned.csv                  # Cleaned dataset (generated)
│
├── data_cleaning.py                  # Step 1–3: Load, audit, clean data
├── eda.py                            # Step 4:   EDA + 10 charts
├── model_training.py                 # Step 5–6: Train, compare, save model
│
├── eda_images/                       # 14 generated PNG charts
│   ├── 01_class_distribution.png
│   ├── 02_approval_by_gender.png
│   ├── 03_approval_by_married.png
│   ├── 04_approval_by_education.png
│   ├── 05_approval_by_self_employed.png
│   ├── 06_approval_by_credit_history.png
│   ├── 07_approval_by_property_area.png
│   ├── 08_applicant_income_dist.png
│   ├── 09_loan_amount_dist.png
│   ├── 10_correlation_heatmap.png
│   ├── 11_model_comparison.png
│   ├── 12_confusion_matrix.png
│   ├── 13_roc_curve.png
│   └── 14_feature_importance.png
│
├── model/                            # Saved ML artefacts
│   ├── best_model.pkl                # Best estimator (full sklearn Pipeline)
│   ├── label_encoders.pkl            # LabelEncoders for categorical features
│   └── feature_names.pkl             # Ordered feature name list
│
├── frontend/
│   └── app.py                        # Step 7: Streamlit web application
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

```bash
# 1. Clone or download the project
cd loan_approval_project

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the following commands **in order** from the project root directory:

### Step 1 — Clean the Data

```bash
python data_cleaning.py
```

**Output:** `loan_cleaned.csv`  
Performs a full data quality audit and applies cleaning transformations.

---

### Step 2 — Exploratory Data Analysis

```bash
python eda.py
```

**Output:** `eda_images/01_*.png` through `eda_images/10_*.png`  
Generates 10 publication-quality charts exploring the dataset.

---

### Step 3 — Train & Evaluate Models

```bash
python model_training.py
```

**Output:**
- `model/best_model.pkl` — best trained model pipeline
- `model/label_encoders.pkl` — saved encoders
- `model/feature_names.pkl` — feature name list
- `eda_images/11_*.png` through `eda_images/14_*.png` — evaluation charts

---

### Step 4 — Launch the Web App

```bash
streamlit run frontend/app.py
```

Opens at **http://localhost:8501** in your browser.

---

## 🔧 Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `data_cleaning.py` | Load raw CSV, audit quality, report missing/outliers |
| 2 | `data_cleaning.py` | Impute missing values, encode target, save cleaned CSV |
| 3 | `eda.py` | Generate 10 EDA charts saved to `eda_images/` |
| 4 | `model_training.py` | Label-encode categoricals, scale features, split data |
| 5 | `model_training.py` | Train 6 classifiers, cross-validate, compare metrics |
| 6 | `model_training.py` | Select best model by AUC, save pipeline + artefacts |
| 7 | `frontend/app.py` | Serve live predictions via Streamlit web app |

### Data Cleaning Summary

| Issue | Count | Fix Applied |
|-------|-------|-------------|
| Missing: `Credit_History` | 50 (8.14%) | Median imputation |
| Missing: `Self_Employed` | 32 (5.21%) | Mode imputation |
| Missing: `LoanAmount` | 22 (3.58%) | Median imputation |
| Missing: `Dependents` | 15 (2.44%) | Mode imputation |
| Missing: `Loan_Amount_Term` | 14 (2.28%) | Median imputation |
| Missing: `Gender` | 13 (2.12%) | Mode imputation |
| Missing: `Married` | 3 (0.49%) | Mode imputation |
| `Dependents` = `'3+'` | — | Replaced with integer `3` |
| `Loan_ID` column | — | Dropped (not a feature) |

**After cleaning:** 0 missing values · Shape: 614 × 12

---

## 🤖 Models Compared

| Model | Type | Key Parameters |
|-------|------|----------------|
| Logistic Regression | Linear | `max_iter=1000` |
| Decision Tree | Tree | `max_depth=6` |
| Random Forest | Ensemble (Bagging) | `n_estimators=200`, `max_depth=8` |
| **Gradient Boosting** ⭐ | Ensemble (Boosting) | `n_estimators=200`, `lr=0.05`, `max_depth=4` |
| SVM | Kernel (RBF) | `probability=True` |
| KNN | Instance-based | `n_neighbors=7` |

All models are wrapped in a `sklearn.Pipeline` with `StandardScaler` prepended.

**Train/Test Split:** 80% / 20% — stratified by `Loan_Status`  
**Cross-Validation:** 5-fold Stratified K-Fold

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of all predictions that are correct |
| **Precision** | Of predicted approvals, how many are truly approved |
| **Recall** | Of actual approvals, how many did the model catch |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under the ROC curve — primary selection metric |
| **CV-F1** | Mean 5-fold cross-validated F1 — guards against overfitting |

> **Best Model:** Gradient Boosting — selected by highest ROC-AUC score (~0.87)

---

## 💡 Key Findings

1. **Credit History is the dominant predictor** — applicants with good credit history have ~80% approval rate vs ~7% for poor history. It accounts for ~38% of the model's decision weight.

2. **Gradient Boosting outperforms all other models** — AUC ~0.87, F1 ~0.87. Sequential error correction handles the mild class imbalance effectively.

3. **Property Area matters** — Semiurban: ~76% approval · Urban: ~67% · Rural: ~62%.

4. **Education level is a secondary predictor** — Graduate: ~71% vs Non-Graduate: ~61%.

5. **Income alone is a weak predictor** — moderate income with good credit history outperforms high income with poor credit.

---

## 🖥️ Streamlit App

The web app (`frontend/app.py`) is organised into four tabs:

| Tab | Content |
|-----|---------|
| 🎯 **Prediction** | Applicant summary cards, Approved/Denied verdict with confidence %, probability bar, key risk factor explanations |
| 📊 **EDA Charts** | All 10 EDA charts in a responsive 2-column grid |
| 🤖 **Model Performance** | Model comparison table, metrics glossary, evaluation charts |
| 📂 **About Dataset** | Feature catalogue, data quality summary, pipeline steps, raw data preview |

**Sidebar** collects all 11 applicant inputs and a one-click **Predict** button.

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.9+ | Core language |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.26 | Numerical computations |
| scikit-learn | ≥ 1.4 | ML models, pipelines, metrics |
| matplotlib | ≥ 3.8 | Chart generation |
| seaborn | ≥ 0.13 | Statistical visualisations |
| Streamlit | ≥ 1.32 | Web application framework |

---

## 📄 License

This project is intended for educational purposes.

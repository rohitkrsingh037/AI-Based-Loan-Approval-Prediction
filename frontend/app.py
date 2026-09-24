"""
app.py  —  Streamlit Frontend
==============================
AI-Based Loan Approval Prediction
-----------------------------------
A clean, readable, well-structured Streamlit application that lets a user:
  • Enter applicant details in a sidebar form
  • See the prediction (Approved / Denied) with confidence score
  • Explore EDA charts (tabbed layout)
  • Review model performance charts (tabbed layout)

Run with:
    streamlit run app.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #546e7a;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    /* Prediction cards */
    .pred-approved {
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.35rem;
        font-weight: 600;
        color: #2e7d32;
    }
    .pred-denied {
        background: #ffebee;
        border-left: 6px solid #e53935;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        font-size: 1.35rem;
        font-weight: 600;
        color: #b71c1c;
    }
    /* Metric cards */
    .metric-card {
        background: #f8f9ff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #78909c; margin-bottom: 4px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #1a237e; }
    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #283593;
        border-bottom: 2px solid #e8eaf6;
        padding-bottom: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── load artefacts ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open("model/best_model.pkl",     "rb") as f: model    = pickle.load(f)
    with open("model/label_encoders.pkl", "rb") as f: encoders = pickle.load(f)
    with open("model/feature_names.pkl",  "rb") as f: features = pickle.load(f)
    return model, encoders, features

try:
    model, label_encoders, FEATURE_NAMES = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ── EDA image catalogue ───────────────────────────────────────────────────
EDA_IMGS = {
    "Class Distribution":         "eda_images/01_class_distribution.png",
    "By Gender":                  "eda_images/02_approval_by_gender.png",
    "By Marital Status":          "eda_images/03_approval_by_married.png",
    "By Education":               "eda_images/04_approval_by_education.png",
    "By Self-Employment":         "eda_images/05_approval_by_self_employed.png",
    "By Credit History":          "eda_images/06_approval_by_credit_history.png",
    "By Property Area":           "eda_images/07_approval_by_property_area.png",
    "Applicant Income":           "eda_images/08_applicant_income_dist.png",
    "Loan Amount":                "eda_images/09_loan_amount_dist.png",
    "Correlation Heatmap":        "eda_images/10_correlation_heatmap.png",
}

MODEL_IMGS = {
    "Model Comparison":           "eda_images/11_model_comparison.png",
    "Confusion Matrix":           "eda_images/12_confusion_matrix.png",
    "ROC Curve":                  "eda_images/13_roc_curve.png",
    "Feature Importance":         "eda_images/14_feature_importance.png",
}

# ─────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏦 AI-Based Loan Approval Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter applicant details in the sidebar to get an instant loan decision powered by machine learning.</p>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────
#  SIDEBAR — input form
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Applicant Details")
    st.markdown("Fill in all fields and click **Predict** to get the result.")
    st.markdown("---")

    gender         = st.selectbox("Gender",                  ["Male", "Female"])
    married        = st.selectbox("Marital Status",          ["Yes", "No"])
    dependents     = st.selectbox("Number of Dependents",    [0, 1, 2, 3])
    education      = st.selectbox("Education Level",         ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Self-Employed",           ["No", "Yes"])

    st.markdown("---")
    applicant_income  = st.number_input("Applicant Monthly Income (₹)",
                                         min_value=0, max_value=100_000,
                                         value=5_000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)",
                                          min_value=0, max_value=50_000,
                                          value=0, step=500)
    loan_amount       = st.number_input("Loan Amount (thousands ₹)",
                                         min_value=1, max_value=700,
                                         value=150, step=5)
    loan_term         = st.selectbox("Loan Amount Term (months)",
                                      [360, 180, 120, 60, 84, 240, 300, 480, 36, 12],
                                      index=0)

    st.markdown("---")
    credit_history = st.selectbox("Credit History",          [1.0, 0.0],
                                   format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)")
    property_area  = st.selectbox("Property Area",           ["Urban", "Semiurban", "Rural"])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────
#  MAIN — tabs
# ─────────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_model, tab_data = st.tabs([
    "🎯 Prediction", "📊 EDA Charts", "🤖 Model Performance", "📂 About Dataset"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════
with tab_pred:
    if not model_ready:
        st.warning(
            "⚠️ Model not found. Please run the following scripts first:\n\n"
            "```\npython data_cleaning.py\npython eda.py\npython model_training.py\n```",
        )
    else:
        # ── applicant summary ──────────────────────────────────────────
        st.markdown('<p class="section-header">Applicant Summary</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gender</div>
                <div class="metric-value">{gender}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Education</div>
                <div class="metric-value">{"Grad" if education == "Graduate" else "Non-Grad"}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Credit History</div>
                <div class="metric-value">{"Good ✅" if credit_history == 1.0 else "Bad ❌"}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Property Area</div>
                <div class="metric-value">{property_area}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Applicant Income</div>
                <div class="metric-value">₹{applicant_income:,}</div>
            </div>""", unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Co-applicant Income</div>
                <div class="metric-value">₹{coapplicant_income:,}</div>
            </div>""", unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Loan Amount</div>
                <div class="metric-value">₹{loan_amount}K</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── prediction ─────────────────────────────────────────────────
        if predict_btn:
            # encode categorical inputs
            def encode(col: str, val: str) -> int:
                return int(label_encoders[col].transform([val])[0])

            row = {
                "Gender":             encode("Gender",        gender),
                "Married":            encode("Married",       married),
                "Dependents":         int(dependents),
                "Education":          encode("Education",     education),
                "Self_Employed":      encode("Self_Employed", self_employed),
                "ApplicantIncome":    applicant_income,
                "CoapplicantIncome":  coapplicant_income,
                "LoanAmount":         loan_amount,
                "Loan_Amount_Term":   loan_term,
                "Credit_History":     credit_history,
                "Property_Area":      encode("Property_Area", property_area),
            }

            # ensure column order matches training
            X_input = np.array([[row[f] for f in FEATURE_NAMES]])
            prediction = model.predict(X_input)[0]
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[prediction] * 100

            if prediction == 1:
                st.markdown(
                    f'<div class="pred-approved">✅ Loan Approved &nbsp;|&nbsp; '
                    f'Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="pred-denied">❌ Loan Denied &nbsp;|&nbsp; '
                    f'Confidence: {confidence:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("&nbsp;", unsafe_allow_html=True)

            # probability bar
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Approval Probability",  f"{proba[1]*100:.1f}%")
            with col_b:
                st.metric("Denial Probability",    f"{proba[0]*100:.1f}%")

            st.progress(float(proba[1]), text="Approval confidence")

            # key factors
            st.markdown('<p class="section-header" style="margin-top:1.5rem;">Key Factors</p>',
                        unsafe_allow_html=True)
            factors = []
            if credit_history == 0.0:
                factors.append("⚠️ **No credit history** — this is the strongest single predictor of denial.")
            if loan_amount > 200:
                factors.append("⚠️ **High loan amount** — large loans carry higher risk.")
            if applicant_income < 3000:
                factors.append("⚠️ **Low applicant income** — may affect repayment capacity.")
            if applicant_income + coapplicant_income > 0:
                emi_ratio = (loan_amount * 1000 / (loan_term * (applicant_income + coapplicant_income) / 12))
                if emi_ratio > 0.5:
                    factors.append(f"⚠️ **Estimated EMI/Income ratio** is high ({emi_ratio:.1%}) — risk of over-leverage.")
            if credit_history == 1.0:
                factors.append("✅ **Good credit history** — strong positive signal.")
            if property_area in ("Semiurban", "Urban"):
                factors.append("✅ **Urban/Semiurban area** — historically higher approval rates.")
            if not factors:
                factors.append("ℹ️ No major risk flags detected.")
            for f in factors:
                st.markdown(f"- {f}")
        else:
            st.info("👈  Fill in the applicant details in the sidebar and click **Predict Loan Approval** to see the result.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA CHARTS
# ══════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<p class="section-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        "These charts summarise the loan dataset used to train the model. "
        "Each chart shows how different applicant attributes relate to loan approval outcomes."
    )
    st.markdown("---")

    eda_keys = list(EDA_IMGS.keys())
    # display in a 2-column grid
    for i in range(0, len(eda_keys), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(eda_keys):
                key  = eda_keys[idx]
                path = EDA_IMGS[key]
                with col:
                    st.markdown(f"**{key}**")
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.warning(f"Image not found: `{path}`  \nRun `python eda.py` to generate it.")
        st.markdown("&nbsp;")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<p class="section-header">Model Training & Evaluation</p>', unsafe_allow_html=True)
    st.markdown(
        "Six machine learning classifiers were trained and compared. "
        "The best-performing model (by AUC score) was saved and powers the prediction tab."
    )

    # models evaluated table
    st.markdown("#### Models Evaluated")
    model_info = pd.DataFrame({
        "Model":       ["Logistic Regression", "Decision Tree", "Random Forest",
                        "Gradient Boosting", "SVM", "KNN"],
        "Type":        ["Linear", "Tree", "Ensemble", "Ensemble", "Kernel", "Instance-based"],
        "Description": [
            "Linear classifier, fast & interpretable",
            "Rule-based splits — easy to visualise",
            "Bagged trees — robust to noise",
            "Boosted trees — generally top performer",
            "Margin maximisation — good for small datasets",
            "Distance-based — non-parametric",
        ],
    })
    st.dataframe(model_info, use_container_width=True, hide_index=True)

    st.markdown("#### Evaluation Metrics Explained")
    metrics_info = pd.DataFrame({
        "Metric":      ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Meaning":     [
            "% of all predictions that are correct",
            "Of predicted approvals, how many are truly approved",
            "Of actual approvals, how many did the model catch",
            "Harmonic mean of Precision and Recall",
            "Area under the ROC curve — overall discrimination ability",
        ],
    })
    st.dataframe(metrics_info, use_container_width=True, hide_index=True)

    st.markdown("---")

    model_keys = list(MODEL_IMGS.keys())
    for i in range(0, len(model_keys), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(model_keys):
                key  = model_keys[idx]
                path = MODEL_IMGS[key]
                with col:
                    st.markdown(f"**{key}**")
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                    else:
                        st.warning(f"Image not found: `{path}`  \nRun `python model_training.py` to generate it.")
        st.markdown("&nbsp;")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT DATASET
# ══════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<p class="section-header">About the Dataset</p>', unsafe_allow_html=True)

    st.markdown("""
**Source:** Loan Approval Prediction Dataset (`LoanApprovalPrediction.csv`)

This dataset contains historical loan application records with the final approval decision.
It is commonly used as a benchmark for binary classification tasks in the financial domain.
""")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature Descriptions")
        features_df = pd.DataFrame({
            "Feature":       ["Loan_ID", "Gender", "Married", "Dependents", "Education",
                              "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                              "LoanAmount", "Loan_Amount_Term", "Credit_History",
                              "Property_Area", "Loan_Status"],
            "Type":          ["ID", "Categorical", "Categorical", "Ordinal", "Categorical",
                              "Categorical", "Numerical", "Numerical",
                              "Numerical", "Numerical", "Binary",
                              "Categorical", "Target (Binary)"],
            "Description":   [
                "Unique loan identifier",
                "Applicant gender: Male / Female",
                "Marital status: Yes / No",
                "Number of dependents: 0 / 1 / 2 / 3+",
                "Graduate / Not Graduate",
                "Whether the applicant is self-employed",
                "Monthly income of the primary applicant",
                "Monthly income of the co-applicant",
                "Requested loan amount (in thousands)",
                "Loan repayment period in months",
                "1 = meets guidelines, 0 = does not",
                "Urban / Semiurban / Rural",
                "Y = Approved, N = Denied",
            ],
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Data Quality Summary")
        dq = pd.DataFrame({
            "Aspect":        ["Total Records", "Total Features", "Target Classes",
                              "Missing Values (raw)", "Cleaned Records",
                              "Cleaning Method"],
            "Detail":        ["614", "13 (incl. Loan_ID)", "Approved (Y) / Denied (N)",
                              "Yes — imputed with mode/median", "614",
                              "Mode for categoricals, Median for numericals"],
        })
        st.dataframe(dq, use_container_width=True, hide_index=True)

        st.markdown("#### Pipeline Steps")
        steps = [
            "1. Load raw CSV (`data_cleaning.py`)",
            "2. Audit data quality (missing, duplicates, outliers)",
            "3. Impute missing values, encode target",
            "4. Exploratory Data Analysis (`eda.py`)",
            "5. Train 6 ML models, cross-validate (`model_training.py`)",
            "6. Select best model by AUC, save artefacts",
            "7. Predict via this Streamlit app (`app.py`)",
        ]
        for s in steps:
            st.markdown(f"- {s}")

    # raw data preview
    st.markdown("---")
    st.markdown("#### Dataset Preview")
    raw_csv = "LoanApprovalPrediction.csv"
    if os.path.exists(raw_csv):
        df_raw = pd.read_csv(raw_csv)
        st.markdown(f"**Shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(15), use_container_width=True, hide_index=True)
    else:
        st.warning(f"`{raw_csv}` not found in the working directory.")

# ── footer ────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#9e9e9e;font-size:0.78rem;'>"
    "AI-Based Loan Approval Prediction &nbsp;|&nbsp; "
    "Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM · KNN"
    "</p>",
    unsafe_allow_html=True,
)

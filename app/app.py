from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
for path_entry in [str(APP_DIR), str(BASE_DIR)]:
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

try:
    from .utils import FIGURES_DIR, REPORTS_DIR, load_resources, score_payload
except ImportError:
    from utils import FIGURES_DIR, REPORTS_DIR, load_resources, score_payload


st.set_page_config(
    page_title="Xente Loan Default Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f2ea 0%, #fbfaf7 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card, .soft-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid #e8edf2;
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 12px 28px rgba(18, 38, 58, 0.06);
    }
    .hero-title {
        color: #12263A;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #627286;
        font-size: 1rem;
        margin-bottom: 0;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_resources():
    return load_resources()


resources = get_resources()
metadata = resources["metadata"]

with st.sidebar:
    st.markdown("## Xente Loan Default")
    st.caption(f"Prepared by {metadata['student_name']} ({metadata['student_id']})")
    page = st.radio(
        "Navigate",
        [
            "Home",
            "Data Overview",
            "EDA Dashboard",
            "Cleaning & Features",
            "Model Performance",
            "Predict Default",
            "Business Recommendations",
        ],
    )
    st.markdown("---")
    st.caption(f"Final model: {metadata['best_model']}")
    st.caption(f"Decision threshold: {metadata['threshold']:.2f}")


if page == "Home":
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Xente Loan Default Prediction</div>
            <div class="hero-subtitle">
                <strong>Uganda Christian University</strong><br>
                Faculty of Engineering, Design and Technology<br>
                Department of Computing and Technology<br>
                EASTER 2026 SEMESTER EXAMINATION<br>
                Programme: Master of Science in Data Science<br>
                Year / Semester: 2 / 1<br>
                Course Code: DSC8305<br>
                Course Name: Business, Management and Financial Data Analytics<br>
                Assessment: Project Based Exam<br>
                Examination Date: April 2026<br>
                <strong>Section B</strong><br><br>
                Prepared by <strong>{metadata['student_name']}</strong> (<strong>{metadata['student_id']}</strong>)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets Used", "4")
    with col2:
        best_metric = resources["metrics"].sort_values("f1", ascending=False).iloc[0]
        st.metric("Best Validation F1", f"{best_metric['f1']:.3f}")
    with col3:
        st.metric("Best ROC-AUC", f"{best_metric['roc_auc']:.3f}")

    st.markdown("### Project Goal")
    st.write(
        "Predict which borrowers are likely to default so Xente can improve credit decision-making, "
        "reduce losses, and make lending more sustainable."
    )
    st.image(str(FIGURES_DIR / "model_comparison_metrics.png"), width="stretch")

elif page == "Data Overview":
    st.markdown("## Data Overview")
    st.dataframe(resources["dataset_summary"], use_container_width=True)
    st.markdown("### Variable Summary")
    st.dataframe(resources["variable_summary"], use_container_width=True, height=520)
    st.markdown("### Modelling Dataset Sample")
    st.dataframe(resources["modelling_df"].head(20), use_container_width=True)

elif page == "EDA Dashboard":
    st.markdown("## EDA Dashboard")
    st.image(str(FIGURES_DIR / "target_distribution.png"), width="stretch")
    st.image(str(FIGURES_DIR / "monthly_trends.png"), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIGURES_DIR / "distribution_Amount.png"), width="stretch")
        st.image(str(FIGURES_DIR / "categories_ProductCategory.png"), width="stretch")
    with col2:
        st.image(str(FIGURES_DIR / "distribution_Value.png"), width="stretch")
        st.image(str(FIGURES_DIR / "relationship_categorical_ProductCategory.png"), width="stretch")

elif page == "Cleaning & Features":
    st.markdown("## Cleaning and Feature Engineering")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Missingness Summary")
        st.dataframe(resources["missingness"].head(15), use_container_width=True)
        st.image(str(FIGURES_DIR / "missingness_top_columns.png"), width="stretch")
    with col2:
        st.markdown("### Leakage and Identifier Exclusions")
        st.dataframe(resources["leakage"], use_container_width=True)
    st.markdown("### Feature Selection")
    st.dataframe(resources["feature_selection"], use_container_width=True, height=500)

elif page == "Model Performance":
    st.markdown("## Model Performance")
    metrics = resources["metrics"]
    best = metrics.sort_values("f1", ascending=False).iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best["model"])
    col2.metric("F1 Score", f"{best['f1']:.3f}")
    col3.metric("ROC-AUC", f"{best['roc_auc']:.3f}")
    st.dataframe(metrics, use_container_width=True)
    st.image(str(FIGURES_DIR / "model_comparison_metrics.png"), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIGURES_DIR / "roc_curve_comparison.png"), width="stretch")
        st.image(str(FIGURES_DIR / "threshold_tradeoff.png"), width="stretch")
    with col2:
        st.image(str(FIGURES_DIR / "confusion_matrix_logistic_regression.png"), width="stretch")
        st.image(str(FIGURES_DIR / "confusion_matrix_random_forest.png"), width="stretch")
    st.image(str(FIGURES_DIR / "best_model_feature_effects.png"), width="stretch")

elif page == "Predict Default":
    st.markdown("## Predict Default")
    st.write(
        "Enter a current borrower profile plus optional behaviour summary inputs. "
        "Any advanced field you leave untouched will fall back to a representative project default."
    )

    modelling_df = resources["modelling_df"]

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=15000.0, step=500.0)
            value = st.number_input("Transaction Value", min_value=0.0, value=15000.0, step=500.0)
            product_category = st.selectbox(
                "Product Category",
                sorted(modelling_df["ProductCategory"].dropna().astype(str).unique().tolist()),
            )
            product_id = st.selectbox(
                "Product ID",
                sorted(modelling_df["ProductId"].dropna().astype(str).unique().tolist()),
            )
            investor_id = st.selectbox(
                "Investor ID",
                sorted(modelling_df["InvestorId"].dropna().astype(str).unique().tolist()),
            )
        with right:
            transaction_start = st.datetime_input("Transaction Start Time", value=datetime(2019, 3, 15, 12, 0))
            issue_date = st.datetime_input("Issued Date Loan", value=datetime(2019, 3, 15, 11, 59))
            prior_txn_count = st.number_input("Prior Customer Transaction Count", min_value=0.0, value=5.0, step=1.0)
            prior_amount_sum = st.number_input("Prior Customer Amount Sum", min_value=0.0, value=60000.0, step=1000.0)
            days_since_prev = st.number_input("Days Since Previous Transaction", min_value=0.0, value=2.0, step=1.0)

        with st.expander("Advanced Behaviour Inputs"):
            c1, c2, c3 = st.columns(3)
            with c1:
                prior_value_sum = st.number_input("Prior Customer Value Sum", min_value=0.0, value=60000.0, step=1000.0)
                prior_amount_mean = st.number_input("Prior Customer Amount Mean", min_value=0.0, value=12000.0, step=500.0)
                prior_value_mean = st.number_input("Prior Customer Value Mean", min_value=0.0, value=12000.0, step=500.0)
                prior_max_amount = st.number_input("Prior Customer Max Amount", min_value=0.0, value=20000.0, step=500.0)
            with c2:
                tenure_days = st.number_input("Customer Tenure Days", min_value=0.0, value=75.0, step=1.0)
                prior_unique_products = st.number_input("Prior Unique Product Categories", min_value=0.0, value=3.0, step=1.0)
                prior_unique_channels = st.number_input("Prior Unique Channels", min_value=0.0, value=1.0, step=1.0)
                prior_same_product = st.number_input("Prior Same Product Count", min_value=0.0, value=2.0, step=1.0)
            with c3:
                prior_same_channel = st.number_input("Prior Same Channel Count", min_value=0.0, value=5.0, step=1.0)
                prior_weekend_share = st.slider("Prior Weekend Share", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        submitted = st.form_submit_button("Score Borrower")

    if submitted:
        probability, prediction, risk_band = score_payload(
            resources,
            {
                "Amount": amount,
                "Value": value,
                "ProductCategory": product_category,
                "ProductId": product_id,
                "InvestorId": investor_id,
                "TransactionStartTime": transaction_start,
                "IssuedDateLoan": issue_date,
                "customer_prior_txn_count": prior_txn_count,
                "customer_prior_abs_amount_sum": prior_amount_sum,
                "customer_prior_abs_value_sum": prior_value_sum,
                "customer_prior_abs_amount_mean": prior_amount_mean,
                "customer_prior_abs_value_mean": prior_value_mean,
                "customer_prior_max_abs_amount": prior_max_amount,
                "customer_days_since_prev_txn": days_since_prev,
                "customer_tenure_days": tenure_days,
                "customer_prior_unique_product_categories": prior_unique_products,
                "customer_prior_unique_channels": prior_unique_channels,
                "customer_prior_same_product_count": prior_same_product,
                "customer_prior_same_channel_count": prior_same_channel,
                "customer_prior_weekend_share": prior_weekend_share,
            },
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Default", f"{probability:.2%}")
        col2.metric("Predicted Class", "Default" if prediction == 1 else "Non-default")
        col3.metric("Risk Band", risk_band)
        st.info(
            "Interpretation: this score is a decision-support signal. It should guide review intensity, not replace lending judgment."
        )

elif page == "Business Recommendations":
    st.markdown("## Business Recommendations")
    st.markdown((REPORTS_DIR / "final_report.md").read_text(encoding="utf-8"))
    st.markdown("### Preview of TEST.csv Predictions")
    st.dataframe(resources["predictions"].head(25), use_container_width=True)

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_prep import add_engineered_features


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
CLEANED_DIR = BASE_DIR / "outputs" / "cleaned"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
REPORTS_DIR = BASE_DIR / "reports"


def load_resources() -> dict[str, object]:
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    modelling_df = pd.read_csv(CLEANED_DIR / "modelling_dataset.csv", parse_dates=["TransactionStartTime", "IssuedDateLoan"])

    resources = {
        "model": model,
        "metadata": metadata,
        "modelling_df": modelling_df,
        "dataset_summary": pd.read_csv(METRICS_DIR / "dataset_summary.csv"),
        "variable_summary": pd.read_csv(METRICS_DIR / "variable_summary.csv"),
        "missingness": pd.read_csv(METRICS_DIR / "missingness_summary.csv"),
        "feature_selection": pd.read_csv(METRICS_DIR / "feature_selection_summary.csv"),
        "metrics": pd.read_csv(METRICS_DIR / "model_comparison_metrics.csv"),
        "leakage": pd.read_csv(METRICS_DIR / "leakage_and_id_exclusions.csv"),
        "predictions": pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv"),
    }
    return resources


def default_profiles(modelling_df: pd.DataFrame, metadata: dict[str, object]) -> tuple[dict[str, float], dict[str, str]]:
    numeric_defaults = {}
    for column in metadata["numeric_features"]:
        numeric_defaults[column] = float(modelling_df[column].median()) if column in modelling_df else 0.0

    categorical_defaults = {}
    for column in metadata["categorical_features"]:
        if column in modelling_df and not modelling_df[column].mode(dropna=True).empty:
            categorical_defaults[column] = str(modelling_df[column].mode(dropna=True).iloc[0])
        else:
            categorical_defaults[column] = "Unknown"

    return numeric_defaults, categorical_defaults


def build_prediction_frame(
    metadata: dict[str, object],
    modelling_df: pd.DataFrame,
    payload: dict[str, object],
) -> pd.DataFrame:
    numeric_defaults, categorical_defaults = default_profiles(modelling_df, metadata)

    base = {
        "Amount": float(payload.get("Amount", numeric_defaults.get("Amount", 0.0))),
        "Value": float(payload.get("Value", numeric_defaults.get("Value", 0.0))),
        "ProductCategory": str(payload.get("ProductCategory", categorical_defaults.get("ProductCategory", "Unknown"))),
        "ProductId": str(payload.get("ProductId", categorical_defaults.get("ProductId", "Unknown"))),
        "InvestorId": str(payload.get("InvestorId", categorical_defaults.get("InvestorId", "Unknown"))),
        "TransactionStartTime": pd.to_datetime(payload.get("TransactionStartTime")),
        "IssuedDateLoan": pd.to_datetime(payload.get("IssuedDateLoan")),
        "customer_prior_txn_count": float(payload.get("customer_prior_txn_count", numeric_defaults.get("customer_prior_txn_count", 0.0))),
        "customer_prior_abs_amount_sum": float(payload.get("customer_prior_abs_amount_sum", numeric_defaults.get("customer_prior_abs_amount_sum", 0.0))),
        "customer_prior_abs_value_sum": float(payload.get("customer_prior_abs_value_sum", numeric_defaults.get("customer_prior_abs_value_sum", 0.0))),
        "customer_prior_abs_amount_mean": float(payload.get("customer_prior_abs_amount_mean", numeric_defaults.get("customer_prior_abs_amount_mean", 0.0))),
        "customer_prior_abs_value_mean": float(payload.get("customer_prior_abs_value_mean", numeric_defaults.get("customer_prior_abs_value_mean", 0.0))),
        "customer_prior_max_abs_amount": float(payload.get("customer_prior_max_abs_amount", numeric_defaults.get("customer_prior_max_abs_amount", 0.0))),
        "customer_days_since_prev_txn": float(payload.get("customer_days_since_prev_txn", numeric_defaults.get("customer_days_since_prev_txn", 0.0))),
        "customer_tenure_days": float(payload.get("customer_tenure_days", numeric_defaults.get("customer_tenure_days", 0.0))),
        "customer_prior_unique_product_categories": float(payload.get("customer_prior_unique_product_categories", numeric_defaults.get("customer_prior_unique_product_categories", 0.0))),
        "customer_prior_unique_channels": float(payload.get("customer_prior_unique_channels", numeric_defaults.get("customer_prior_unique_channels", 0.0))),
        "customer_prior_same_product_count": float(payload.get("customer_prior_same_product_count", numeric_defaults.get("customer_prior_same_product_count", 0.0))),
        "customer_prior_same_channel_count": float(payload.get("customer_prior_same_channel_count", numeric_defaults.get("customer_prior_same_channel_count", 0.0))),
        "customer_prior_weekend_share": float(payload.get("customer_prior_weekend_share", numeric_defaults.get("customer_prior_weekend_share", 0.0))),
    }

    frame = pd.DataFrame([base])
    engineered = add_engineered_features(frame)

    for column in metadata["numeric_features"]:
        if column not in engineered.columns:
            engineered[column] = numeric_defaults.get(column, np.nan)
    for column in metadata["categorical_features"]:
        if column not in engineered.columns:
            engineered[column] = categorical_defaults.get(column, "Unknown")

    return engineered[metadata["numeric_features"] + metadata["categorical_features"]]


def score_payload(resources: dict[str, object], payload: dict[str, object]) -> tuple[float, int, str]:
    frame = build_prediction_frame(resources["metadata"], resources["modelling_df"], payload)
    probability = float(resources["model"].predict_proba(frame)[0, 1])
    threshold = float(resources["metadata"]["threshold"])
    prediction = int(probability >= threshold)

    if probability < threshold * 0.8:
        risk_band = "Low risk"
    elif probability < threshold * 1.15:
        risk_band = "Medium risk"
    else:
        risk_band = "High risk"

    return probability, prediction, risk_band


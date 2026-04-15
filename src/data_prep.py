from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    CLEANED_DIR,
    DATA_DIR,
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    METRICS_DIR,
    VALIDATION_SHARE,
)


def load_datasets() -> dict[str, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "TRAIN.csv", na_values=["NA"])
    test = pd.read_csv(DATA_DIR / "TEST.csv", na_values=["NA"])
    unlinked = pd.read_csv(DATA_DIR / "unlinked_masked_final.csv", na_values=["NA"])
    variable_definitions = pd.read_csv(DATA_DIR / "VariableDefinitions.csv")

    train["source_dataset"] = "TRAIN"
    test["source_dataset"] = "TEST"
    unlinked["source_dataset"] = "UNLINKED"

    train["row_id"] = [f"TRAIN_{idx}" for idx in range(len(train))]
    test["row_id"] = [f"TEST_{idx}" for idx in range(len(test))]
    unlinked["row_id"] = [f"UNLINKED_{idx}" for idx in range(len(unlinked))]

    return {
        "train": train,
        "test": test,
        "unlinked": unlinked,
        "variable_definitions": variable_definitions,
    }


def parse_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = datasets["train"].copy()
    test = datasets["test"].copy()
    unlinked = datasets["unlinked"].copy()

    train["TransactionStartTime"] = pd.to_datetime(train["TransactionStartTime"])
    test["TransactionStartTime"] = pd.to_datetime(test["TransactionStartTime"])
    unlinked["TransactionStartTime"] = pd.to_datetime(
        unlinked["TransactionStartTime"], format="%d/%m/%y %H:%M:%S"
    )

    for frame in [train, test]:
        if "IssuedDateLoan" in frame.columns:
            frame["IssuedDateLoan"] = pd.to_datetime(frame["IssuedDateLoan"])
        if "PaidOnDate" in frame.columns:
            frame["PaidOnDate"] = pd.to_datetime(frame["PaidOnDate"])
        if "DueDate" in frame.columns:
            frame["DueDate"] = pd.to_datetime(frame["DueDate"])

    numeric_columns = ["Value", "Amount", "AmountLoan", "CountryCode", "TransactionStatus"]
    for frame in [train, test, unlinked]:
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    train["IsDefaulted"] = pd.to_numeric(train["IsDefaulted"], errors="coerce")
    datasets["train"] = train
    datasets["test"] = test
    datasets["unlinked"] = unlinked
    return datasets


def dataset_summary(train: pd.DataFrame, test: pd.DataFrame, unlinked: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "dataset": "TRAIN",
                "rows": len(train),
                "columns": train.shape[1],
                "unique_customers": train["CustomerId"].nunique(),
                "unique_transactions": train["TransactionId"].nunique(),
                "date_min": train["TransactionStartTime"].min(),
                "date_max": train["TransactionStartTime"].max(),
            },
            {
                "dataset": "TEST",
                "rows": len(test),
                "columns": test.shape[1],
                "unique_customers": test["CustomerId"].nunique(),
                "unique_transactions": test["TransactionId"].nunique(),
                "date_min": test["TransactionStartTime"].min(),
                "date_max": test["TransactionStartTime"].max(),
            },
            {
                "dataset": "UNLINKED",
                "rows": len(unlinked),
                "columns": unlinked.shape[1],
                "unique_customers": unlinked["CustomerId"].nunique(),
                "unique_transactions": unlinked["TransactionId"].nunique(),
                "date_min": unlinked["TransactionStartTime"].min(),
                "date_max": unlinked["TransactionStartTime"].max(),
            },
        ]
    )
    summary.to_csv(METRICS_DIR / "dataset_summary.csv", index=False)
    return summary


def modelling_unit_summary(train: pd.DataFrame) -> pd.DataFrame:
    labelled = train.loc[train["IsDefaulted"].notna()].copy()
    summary = pd.DataFrame(
        [
            {"check": "Known-target rows", "value": len(labelled)},
            {"check": "Distinct customers", "value": labelled["CustomerId"].nunique()},
            {"check": "Distinct transactions", "value": labelled["TransactionId"].nunique()},
            {"check": "Distinct loans", "value": labelled["LoanId"].nunique()},
            {"check": "Loans appearing on multiple rows", "value": int((labelled["LoanId"].value_counts() > 1).sum())},
            {"check": "Customers appearing multiple times", "value": int((labelled["CustomerId"].value_counts() > 1).sum())},
            {"check": "Repeated transaction ids", "value": int(labelled["TransactionId"].duplicated().sum())},
        ]
    )
    summary.to_csv(METRICS_DIR / "modelling_unit_summary.csv", index=False)

    per_loan = labelled.groupby("LoanId").agg(
        rows_per_loan=("row_id", "count"),
        unique_targets=("IsDefaulted", "nunique"),
    )
    per_loan.reset_index().to_csv(METRICS_DIR / "loan_level_repetition.csv", index=False)
    return summary


def variable_summary(modelling_df: pd.DataFrame, variable_definitions: pd.DataFrame) -> pd.DataFrame:
    definition_map = dict(
        zip(variable_definitions["Variable"], variable_definitions["Definition"], strict=False)
    )
    rows = []
    for column in modelling_df.columns:
        if column in ["row_id", "source_dataset"]:
            continue

        series = modelling_df[column]
        if pd.api.types.is_numeric_dtype(series):
            var_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            var_type = "datetime"
        else:
            var_type = "categorical"

        if column in IDENTIFIER_COLUMNS:
            role = "identifier"
        elif column in LEAKAGE_COLUMNS:
            role = "leakage-prone"
        elif column == "IsDefaulted":
            role = "target"
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = "time"
        else:
            role = "candidate predictor"

        rows.append(
            {
                "variable": column,
                "definition": definition_map.get(column, "Engineered or undocumented variable"),
                "type": var_type,
                "missing_pct": round(series.isna().mean() * 100, 2),
                "unique_values": series.nunique(dropna=True),
                "role": role,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["role", "variable"])
    summary.to_csv(METRICS_DIR / "variable_summary.csv", index=False)
    return summary


def leakage_and_id_summary() -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "feature": column,
                "decision": "Excluded",
                "reason": "This variable is only known after repayment events or directly reflects post-outcome behaviour.",
            }
            for column in LEAKAGE_COLUMNS
        ]
        + [
            {
                "feature": column,
                "decision": "Excluded",
                "reason": "Dropped as a raw identifier because it labels records rather than describing borrower risk behaviour.",
            }
            for column in IDENTIFIER_COLUMNS + ["row_id", "source_dataset"]
        ]
    )
    summary.to_csv(METRICS_DIR / "leakage_and_id_exclusions.csv", index=False)
    return summary


def build_history_frame(train: pd.DataFrame, test: pd.DataFrame, unlinked: pd.DataFrame) -> pd.DataFrame:
    common_columns = [
        "row_id",
        "source_dataset",
        "CustomerId",
        "TransactionId",
        "TransactionStartTime",
        "Value",
        "Amount",
        "ProductCategory",
        "ChannelId",
        "ProviderId",
        "ProductId",
    ]

    history = pd.concat(
        [train[common_columns], test[common_columns], unlinked[common_columns]],
        ignore_index=True,
    )
    history = history.sort_values(["CustomerId", "TransactionStartTime", "row_id"]).reset_index(drop=True)
    history["abs_amount"] = history["Amount"].abs()
    history["abs_value"] = history["Value"].abs()
    history["is_weekend_txn"] = history["TransactionStartTime"].dt.weekday >= 5
    return history


def _prior_unique_count(values: Iterable[object]) -> list[int]:
    seen: set[object] = set()
    counts: list[int] = []
    for value in values:
        counts.append(len(seen))
        if pd.notna(value):
            seen.add(value)
    return counts


def compute_history_features(history: pd.DataFrame) -> pd.DataFrame:
    feature_frames: list[pd.DataFrame] = []
    for _, customer_df in history.groupby("CustomerId", sort=False):
        customer_df = customer_df.copy()
        customer_df["customer_prior_txn_count"] = np.arange(len(customer_df))
        customer_df["customer_prior_abs_amount_sum"] = customer_df["abs_amount"].cumsum().shift(fill_value=0)
        customer_df["customer_prior_abs_value_sum"] = customer_df["abs_value"].cumsum().shift(fill_value=0)
        customer_df["customer_prior_abs_amount_mean"] = (
            customer_df["customer_prior_abs_amount_sum"] / customer_df["customer_prior_txn_count"].replace(0, np.nan)
        )
        customer_df["customer_prior_abs_value_mean"] = (
            customer_df["customer_prior_abs_value_sum"] / customer_df["customer_prior_txn_count"].replace(0, np.nan)
        )
        customer_df["customer_prior_max_abs_amount"] = customer_df["abs_amount"].cummax().shift(fill_value=0)
        customer_df["customer_days_since_prev_txn"] = customer_df["TransactionStartTime"].diff().dt.total_seconds().div(86400)
        customer_df["customer_tenure_days"] = (
            customer_df["TransactionStartTime"] - customer_df["TransactionStartTime"].iloc[0]
        ).dt.total_seconds().div(86400)
        customer_df["customer_prior_unique_product_categories"] = _prior_unique_count(
            customer_df["ProductCategory"].tolist()
        )
        customer_df["customer_prior_unique_channels"] = _prior_unique_count(
            customer_df["ChannelId"].tolist()
        )
        customer_df["customer_prior_same_product_count"] = customer_df.groupby("ProductCategory").cumcount()
        customer_df["customer_prior_same_channel_count"] = customer_df.groupby("ChannelId").cumcount()
        weekend_cumulative = customer_df["is_weekend_txn"].astype(int).cumsum().shift(fill_value=0)
        customer_df["customer_prior_weekend_share"] = (
            weekend_cumulative / customer_df["customer_prior_txn_count"].replace(0, np.nan)
        )
        feature_frames.append(customer_df)

    features = pd.concat(feature_frames, ignore_index=True)[
        [
            "row_id",
            "customer_prior_txn_count",
            "customer_prior_abs_amount_sum",
            "customer_prior_abs_value_sum",
            "customer_prior_abs_amount_mean",
            "customer_prior_abs_value_mean",
            "customer_prior_max_abs_amount",
            "customer_days_since_prev_txn",
            "customer_tenure_days",
            "customer_prior_unique_product_categories",
            "customer_prior_unique_channels",
            "customer_prior_same_product_count",
            "customer_prior_same_channel_count",
            "customer_prior_weekend_share",
        ]
    ].copy()
    features.to_csv(CLEANED_DIR / "customer_history_features.csv", index=False)
    return features


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["abs_amount"] = enriched["Amount"].abs()
    enriched["abs_value"] = enriched["Value"].abs()
    enriched["amount_value_gap"] = enriched["abs_amount"] - enriched["abs_value"]
    enriched["amount_to_value_ratio"] = np.where(
        enriched["abs_value"].fillna(0).eq(0),
        np.nan,
        enriched["abs_amount"] / enriched["abs_value"],
    )
    enriched["log_abs_amount"] = np.log1p(enriched["abs_amount"])
    enriched["log_abs_value"] = np.log1p(enriched["abs_value"])
    enriched["txn_month"] = enriched["TransactionStartTime"].dt.month
    enriched["txn_day"] = enriched["TransactionStartTime"].dt.day
    enriched["txn_hour"] = enriched["TransactionStartTime"].dt.hour
    enriched["txn_weekday"] = enriched["TransactionStartTime"].dt.weekday
    enriched["txn_is_weekend"] = (enriched["txn_weekday"] >= 5).astype(int)

    if "IssuedDateLoan" in enriched.columns:
        enriched["issue_month"] = enriched["IssuedDateLoan"].dt.month
        enriched["issue_day"] = enriched["IssuedDateLoan"].dt.day
        enriched["issue_hour"] = enriched["IssuedDateLoan"].dt.hour
        enriched["issue_to_txn_minutes"] = (
            enriched["TransactionStartTime"] - enriched["IssuedDateLoan"]
        ).dt.total_seconds().div(60)

    for column in [
        "customer_prior_abs_amount_sum",
        "customer_prior_abs_value_sum",
        "customer_prior_abs_amount_mean",
        "customer_prior_abs_value_mean",
        "customer_prior_max_abs_amount",
        "customer_days_since_prev_txn",
        "customer_tenure_days",
    ]:
        if column in enriched.columns:
            enriched[f"log_{column}"] = np.log1p(enriched[column].clip(lower=0))

    return enriched


def build_modelling_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    history_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enriched = add_engineered_features(train.merge(history_features, on="row_id", how="left"))
    test_enriched = add_engineered_features(test.merge(history_features, on="row_id", how="left"))

    model_df = train_enriched.loc[train_enriched["IsDefaulted"].notna()].copy()
    model_df["IsDefaulted"] = model_df["IsDefaulted"].astype(int)

    model_df.to_csv(CLEANED_DIR / "modelling_dataset.csv", index=False)
    test_enriched.to_csv(CLEANED_DIR / "test_scoring_dataset.csv", index=False)
    return model_df, test_enriched


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.isna().mean().mul(100).sort_values(ascending=False).reset_index().rename(
            columns={"index": "column", 0: "missing_pct"}
        )
    )
    table["missing_count"] = table["column"].map(df.isna().sum())
    table.to_csv(METRICS_DIR / "missingness_summary.csv", index=False)
    return table


def feature_lists(model_df: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    drop_columns = set(LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS)
    drop_columns.update(
        ["IsDefaulted", "TransactionStartTime", "IssuedDateLoan", "Currency", "AmountLoan", "row_id", "source_dataset"]
    )

    usable = [column for column in model_df.columns if column not in drop_columns]
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    rows = []

    for column in usable:
        series = model_df[column]
        unique_values = series.nunique(dropna=True)
        missing_pct = series.isna().mean() * 100

        if unique_values <= 1:
            selected = False
            reason = "Dropped because the feature is constant."
        elif missing_pct > 85:
            selected = False
            reason = "Dropped because missingness is too high."
        elif pd.api.types.is_numeric_dtype(series):
            selected = True
            reason = "Selected as a numeric signal available at prediction time."
            numeric_features.append(column)
        else:
            selected = True
            reason = "Selected as a categorical signal available at prediction time."
            categorical_features.append(column)

        rows.append(
            {
                "feature": column,
                "selected": selected,
                "missing_pct": round(missing_pct, 2),
                "unique_values": unique_values,
                "reason": reason,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["selected", "feature"], ascending=[False, True])
    summary.to_csv(METRICS_DIR / "feature_selection_summary.csv", index=False)
    return numeric_features, categorical_features, summary


def time_based_split(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = model_df.sort_values(["TransactionStartTime", "row_id"]).reset_index(drop=True)
    split_index = math.floor(len(ordered) * (1 - VALIDATION_SHARE))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def clip_outliers(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clipped_train = train_df.copy()
    clipped_valid = valid_df.copy()
    clipped_test = test_df.copy()
    bounds_rows = []

    for column in numeric_features:
        if column.startswith("txn_") or column.startswith("issue_"):
            lower = train_df[column].min()
            upper = train_df[column].max()
            action = "kept natural range"
        else:
            q1 = train_df[column].quantile(0.25)
            q3 = train_df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            clipped_train[column] = clipped_train[column].clip(lower=lower, upper=upper)
            clipped_valid[column] = clipped_valid[column].clip(lower=lower, upper=upper)
            clipped_test[column] = clipped_test[column].clip(lower=lower, upper=upper)
            action = "IQR clip"

        bounds_rows.append(
            {
                "feature": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "action": action,
            }
        )

    bounds = pd.DataFrame(bounds_rows)
    bounds.to_csv(METRICS_DIR / "outlier_bounds.csv", index=False)
    return clipped_train, clipped_valid, clipped_test, bounds

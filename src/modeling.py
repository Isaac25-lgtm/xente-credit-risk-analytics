from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PREMIUM_COLORS,
    RANDOM_STATE,
    STUDENT_ID,
    STUDENT_NAME,
)
from .utils import save_figure, write_json


@dataclass
class ModelRun:
    name: str
    pipeline: Pipeline
    threshold: float
    metrics: dict[str, float]
    y_valid_pred: np.ndarray
    y_valid_proba: np.ndarray


def build_feature_matrix(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    return df[numeric_features + categorical_features].copy()


def create_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", Pipeline(steps=categorical_steps), categorical_features),
        ]
    )


def find_best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.arange(0.20, 0.81, 0.02)
    best_threshold = 0.50
    best_score = -1.0

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def evaluate_model(name: str, pipeline: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> ModelRun:
    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    threshold = find_best_threshold(y_valid, probabilities)
    preds = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_valid, preds),
        "precision": precision_score(y_valid, preds, zero_division=0),
        "recall": recall_score(y_valid, preds, zero_division=0),
        "f1": f1_score(y_valid, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_valid, probabilities),
        "threshold": threshold,
    }
    return ModelRun(
        name=name,
        pipeline=pipeline,
        threshold=threshold,
        metrics=metrics,
        y_valid_pred=preds,
        y_valid_proba=probabilities,
    )


def train_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[list[ModelRun], pd.Series]:
    X_train = build_feature_matrix(train_df, numeric_features, categorical_features)
    y_train = train_df["IsDefaulted"].astype(int)
    X_valid = build_feature_matrix(valid_df, numeric_features, categorical_features)
    y_valid = valid_df["IsDefaulted"].astype(int)

    logreg_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.7,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=450,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logreg_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    runs = [
        evaluate_model("Logistic Regression", logreg_pipeline, X_valid, y_valid),
        evaluate_model("Random Forest", rf_pipeline, X_valid, y_valid),
    ]
    return runs, y_valid


def write_classification_reports(runs: list[ModelRun], y_valid: pd.Series) -> None:
    report_rows = []
    for run in runs:
        report = classification_report(y_valid, run.y_valid_pred, output_dict=True, zero_division=0)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                report_rows.append(
                    {
                        "model": run.name,
                        "label": label,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )
    pd.DataFrame(report_rows).to_csv(METRICS_DIR / "classification_reports.csv", index=False)


def build_model_comparison_plots(runs: list[ModelRun], y_valid: pd.Series) -> pd.DataFrame:
    metrics_df = pd.DataFrame([{"model": run.name, **run.metrics} for run in runs])
    metrics_df.to_csv(METRICS_DIR / "model_comparison_metrics.csv", index=False)

    melted = metrics_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=melted,
        x="variable",
        y="value",
        hue="model",
        palette=[PREMIUM_COLORS["teal"], PREMIUM_COLORS["gold"]],
        ax=ax,
    )
    ax.set_title("Validation Metrics by Model")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    save_figure(fig, "model_comparison_metrics.png")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for run, color in zip(runs, [PREMIUM_COLORS["teal"], PREMIUM_COLORS["gold"]], strict=False):
        fpr, tpr, _ = roc_curve(y_valid, run.y_valid_proba)
        ax.plot(fpr, tpr, linewidth=2.2, color=color, label=f"{run.name} (AUC={run.metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=PREMIUM_COLORS["slate"])
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    save_figure(fig, "roc_curve_comparison.png")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for run, color in zip(runs, [PREMIUM_COLORS["teal"], PREMIUM_COLORS["gold"]], strict=False):
        precision, recall, _ = precision_recall_curve(y_valid, run.y_valid_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2.2, color=color, label=f"{run.name} (AUC={pr_auc:.3f})")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    save_figure(fig, "precision_recall_curve_comparison.png")

    for run in runs:
        fig, ax = plt.subplots(figsize=(6.4, 5.5))
        matrix = confusion_matrix(y_valid, run.y_valid_pred)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(PREMIUM_COLORS["teal"], as_cmap=True),
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {run.name}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        save_figure(fig, f"confusion_matrix_{run.name.lower().replace(' ', '_')}.png")

    return metrics_df


def plot_threshold_tradeoff(model_name: str, y_valid: pd.Series, probabilities: np.ndarray) -> None:
    rows = []
    for threshold in np.arange(0.20, 0.81, 0.02):
        preds = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_valid, preds, zero_division=0),
                "recall": recall_score(y_valid, preds, zero_division=0),
                "f1": f1_score(y_valid, preds, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], color=PREMIUM_COLORS["gold"], label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], color=PREMIUM_COLORS["rose"], label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], color=PREMIUM_COLORS["teal"], label="F1")
    ax.set_title(f"Threshold Trade-off for {model_name}")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    save_figure(fig, "threshold_tradeoff.png")


def extract_feature_effects(run: ModelRun) -> pd.DataFrame:
    preprocessor = run.pipeline.named_steps["preprocessor"]
    model = run.pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "coef_"):
        effects = pd.DataFrame(
            {
                "feature": feature_names,
                "effect": model.coef_[0],
                "abs_effect": np.abs(model.coef_[0]),
            }
        ).sort_values("abs_effect", ascending=False)
        value_column = "effect"
        title = f"Strongest Logistic Drivers of Default: {run.name}"
    else:
        effects = pd.DataFrame(
            {
                "feature": feature_names,
                "effect": model.feature_importances_,
                "abs_effect": model.feature_importances_,
            }
        ).sort_values("abs_effect", ascending=False)
        value_column = "effect"
        title = f"Random Forest Feature Importance: {run.name}"

    effects.to_csv(METRICS_DIR / "best_model_feature_effects.csv", index=False)
    top = effects.head(12).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PREMIUM_COLORS["rose"] if value < 0 else PREMIUM_COLORS["teal"] for value in top[value_column]]
    ax.barh(top["feature"], top[value_column], color=colors)
    ax.set_title(title)
    ax.set_xlabel("Model effect")
    save_figure(fig, "best_model_feature_effects.png")
    return effects


def refit_best_model(
    best_run: ModelRun,
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    final_pipeline = best_run.pipeline
    X_full = build_feature_matrix(full_df, numeric_features, categorical_features)
    y_full = full_df["IsDefaulted"].astype(int)
    X_test = build_feature_matrix(test_df, numeric_features, categorical_features)

    final_pipeline.fit(X_full, y_full)
    joblib.dump(final_pipeline, MODELS_DIR / "final_model.joblib")

    write_json(
        MODELS_DIR / "model_metadata.json",
        {
            "student_name": STUDENT_NAME,
            "student_id": STUDENT_ID,
            "best_model": best_run.name,
            "threshold": best_run.threshold,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "leakage_columns": LEAKAGE_COLUMNS,
            "identifier_columns": IDENTIFIER_COLUMNS,
        },
    )

    probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= best_run.threshold).astype(int)
    predictions_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"],
            "TransactionId": test_df["TransactionId"],
            "CustomerId": test_df["CustomerId"],
            "predicted_probability_default": probabilities,
            "predicted_default_class": predictions,
        }
    )
    predictions_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=False)
    return predictions_df

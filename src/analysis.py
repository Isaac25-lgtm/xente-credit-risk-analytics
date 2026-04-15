from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from .config import METRICS_DIR, PREMIUM_COLORS
from .utils import save_figure


def build_target_distribution_plot(y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=["Non-default (0)", "Default (1)"],
        y=counts.values,
        palette=[PREMIUM_COLORS["teal"], PREMIUM_COLORS["rose"]],
        hue=["Non-default (0)", "Default (1)"],
        dodge=False,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title("Target Class Distribution in the Modelling Dataset")
    ax.set_xlabel("")
    ax.set_ylabel("Number of observations")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + max(counts.values) * 0.02, f"{value:,}", ha="center")
    save_figure(fig, "target_distribution.png")


def build_missingness_plot(missingness_table: pd.DataFrame) -> None:
    top_missing = missingness_table.head(12).sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_missing["column"], top_missing["missing_pct"], color=PREMIUM_COLORS["gold"])
    ax.set_title("Top Missingness Rates in the Modelling Dataset")
    ax.set_xlabel("Missing values (%)")
    ax.xaxis.set_major_formatter(PercentFormatter())
    save_figure(fig, "missingness_top_columns.png")


def build_numeric_distribution_plots(df: pd.DataFrame, features: list[str]) -> None:
    for feature in features[:6]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        sns.histplot(df[feature], kde=True, ax=axes[0], color=PREMIUM_COLORS["teal"])
        axes[0].set_title(f"Distribution of {feature}")
        sns.boxplot(x=df[feature], ax=axes[1], color=PREMIUM_COLORS["gold"])
        axes[1].set_title(f"Outlier Profile of {feature}")
        save_figure(fig, f"distribution_{feature}.png")


def build_categorical_distribution_plots(df: pd.DataFrame, features: list[str]) -> None:
    for feature in features[:4]:
        top_counts = df[feature].fillna("Missing").value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top_counts.index.astype(str), top_counts.values, color=PREMIUM_COLORS["teal"])
        ax.set_title(f"Top Categories for {feature}")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        save_figure(fig, f"categories_{feature}.png")


def build_time_trend_plots(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(tx_month_period=df["TransactionStartTime"].dt.to_period("M").astype(str))
        .groupby("tx_month_period")
        .agg(transactions=("row_id", "count"), default_rate=("IsDefaulted", "mean"))
        .reset_index()
    )
    monthly.to_csv(METRICS_DIR / "monthly_activity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(monthly["tx_month_period"], monthly["transactions"], marker="o", color=PREMIUM_COLORS["teal"])
    axes[0].set_title("Monthly Volume of Labelled Loan-linked Transactions")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].plot(monthly["tx_month_period"], monthly["default_rate"], marker="o", color=PREMIUM_COLORS["rose"])
    axes[1].set_title("Monthly Default Rate")
    axes[1].set_ylim(0, max(monthly["default_rate"].max() * 1.2, 0.2))
    axes[1].tick_params(axis="x", rotation=35)
    save_figure(fig, "monthly_trends.png")
    return monthly


def run_relationship_tests(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> None:
    y = df["IsDefaulted"].astype(int)

    numeric_rows = []
    for feature in numeric_features[:8]:
        defaults = df.loc[y.eq(1), feature].dropna()
        non_defaults = df.loc[y.eq(0), feature].dropna()
        if len(defaults) > 5 and len(non_defaults) > 5:
            mann_whitney = stats.mannwhitneyu(defaults, non_defaults, alternative="two-sided")
            corr = stats.pointbiserialr(y, df[feature].fillna(df[feature].median()))
            numeric_rows.append(
                {
                    "feature": feature,
                    "default_median": defaults.median(),
                    "non_default_median": non_defaults.median(),
                    "mann_whitney_pvalue": mann_whitney.pvalue,
                    "point_biserial_corr": corr.statistic,
                    "point_biserial_pvalue": corr.pvalue,
                }
            )
    pd.DataFrame(numeric_rows).sort_values("mann_whitney_pvalue").to_csv(
        METRICS_DIR / "numeric_relationship_tests.csv", index=False
    )

    categorical_rows = []
    for feature in categorical_features[:6]:
        contingency = pd.crosstab(df[feature].fillna("Missing"), y)
        if contingency.shape[0] > 1:
            chi2, pvalue, dof, _ = stats.chi2_contingency(contingency)
            default_rates = (
                df.groupby(feature, dropna=False)["IsDefaulted"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            categorical_rows.append(
                {
                    "feature": feature,
                    "chi_square_stat": chi2,
                    "chi_square_pvalue": pvalue,
                    "degrees_of_freedom": dof,
                    "highest_defaulting_category": default_rates.index[0] if not default_rates.empty else None,
                    "highest_default_rate": default_rates.iloc[0] if not default_rates.empty else None,
                }
            )
    pd.DataFrame(categorical_rows).sort_values("chi_square_pvalue").to_csv(
        METRICS_DIR / "categorical_relationship_tests.csv", index=False
    )


def build_relationship_plots(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> None:
    for feature in numeric_features[:4]:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.boxplot(
            data=df,
            x="IsDefaulted",
            y=feature,
            palette=[PREMIUM_COLORS["teal"], PREMIUM_COLORS["rose"]],
            hue="IsDefaulted",
            dodge=False,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_title(f"{feature} by Default Outcome")
        ax.set_xlabel("IsDefaulted")
        save_figure(fig, f"relationship_numeric_{feature}.png")

    for feature in categorical_features[:3]:
        rates = (
            df.groupby(feature, dropna=False)["IsDefaulted"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(rates.index.astype(str), rates.values, color=PREMIUM_COLORS["rose"])
        ax.set_title(f"Default Rate by {feature}")
        ax.set_xlabel("Default rate")
        save_figure(fig, f"relationship_categorical_{feature}.png")

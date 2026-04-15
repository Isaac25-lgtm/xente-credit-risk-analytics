from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR, STUDENT_ID, STUDENT_NAME
from .modeling import ModelRun


def write_business_report(best_run: ModelRun, metrics_df: pd.DataFrame) -> None:
    best_row = metrics_df.loc[metrics_df["model"].eq(best_run.name)].iloc[0]
    report = f"""# Xente Loan Default Project Report

Prepared by **{STUDENT_NAME} ({STUDENT_ID})**

## Executive Summary

This project built a leakage-aware binary classification solution to predict whether a Xente borrower is likely to default on a loan. The modelling workflow prioritised business realism by excluding post-outcome fields, engineering transaction-history features from prior behaviour, and validating the models with a time-based split.

The strongest validation model was **{best_run.name}**, evaluated at an optimized probability threshold of **{best_run.threshold:.2f}**.

## Best Validation Metrics

- Accuracy: **{best_row['accuracy']:.3f}**
- Precision: **{best_row['precision']:.3f}**
- Recall: **{best_row['recall']:.3f}**
- F1-score: **{best_row['f1']:.3f}**
- ROC-AUC: **{best_row['roc_auc']:.3f}**

## Method Highlights

- Used only observations with known `IsDefaulted` for supervised learning.
- Removed repayment and payback variables that would create leakage.
- Built customer history features from prior transactions rather than relying on raw identifiers.
- Treated extreme monetary values with a combination of business-aware inspection, log transformation, and IQR clipping.
- Compared Logistic Regression and Random Forest using the same cleaned feature set.

## Business Interpretation

- Default risk is not random: it varies with transaction context, customer history, and timing patterns.
- Behavioural summaries are more decision-useful than raw IDs because they capture repeat usage intensity and borrower familiarity with the platform.
- Threshold tuning improved the balance between catching likely defaulters and avoiding excessive false alarms.

## Recommendations

1. Use the model as an early warning screening layer in digital credit approval.
2. Send high-risk and borderline cases for manual review instead of using a purely automated accept-reject rule.
3. Track performance drift over time because customer behaviour and product mix can change.
4. Expand future versions with additional verified behavioural history if Xente can make it available at decision time.
"""
    (REPORTS_DIR / "final_report.md").write_text(report, encoding="utf-8")


def write_presentation_outline(best_run: ModelRun, metrics_df: pd.DataFrame) -> None:
    best_row = metrics_df.loc[metrics_df["model"].eq(best_run.name)].iloc[0]
    outline = f"""# Xente Loan Default Prediction Presentation

Prepared by **{STUDENT_NAME} ({STUDENT_ID})**

## Slide 1: Title
- Xente Loan Default Prediction
- Student: {STUDENT_NAME}
- Student Number: {STUDENT_ID}

## Slide 2: Business Problem
- Xente offers digital credit and payment services.
- Loan default harms profitability and sustainability.
- The key question is whether transaction behaviour can predict default risk early.

## Slide 3: Objective and Goal
- Objective: predict `IsDefaulted` using transaction and loan-linked data.
- Goal: support better, faster, and more consistent credit decisions.

## Slide 4: Dataset Overview
- `TRAIN.csv` for development
- `TEST.csv` for unseen scoring
- `VariableDefinitions.csv` for the data dictionary
- `unlinked_masked_final.csv` as an optional behaviour-enrichment source

## Slide 5: Data Quality and Modelling Unit
- Not every row in `TRAIN.csv` has a usable target.
- Repeated customers and loans indicate behavioural sequences, not just isolated events.
- A practical row-level approach was adopted for the exam.

## Slide 6: Leakage Screening
- Excluded fields such as `PaidOnDate`, `IsFinalPayBack`, and `PayBackId`.
- Removed raw identifiers from direct modelling.
- Kept only fields available at or before the prediction point.

## Slide 7: EDA Highlights
- The target is imbalanced, with defaults as the minority class.
- Monetary fields are strongly skewed and contain meaningful high-end outliers.
- Default rates vary across product, channel, timing, and behavioural history patterns.

## Slide 8: Feature Engineering
- Current transaction intensity features
- Time and calendar features
- Customer prior transaction history
- Behavioural diversity and recency signals

## Slide 9: Models Compared
- Logistic Regression for interpretability
- Random Forest for nonlinear pattern capture

## Slide 10: Best Model Result
- Best model: **{best_run.name}**
- Accuracy: **{best_row['accuracy']:.3f}**
- Precision: **{best_row['precision']:.3f}**
- Recall: **{best_row['recall']:.3f}**
- F1-score: **{best_row['f1']:.3f}**
- ROC-AUC: **{best_row['roc_auc']:.3f}**

## Slide 11: Threshold Discussion
- The standard 0.50 cut-off was not assumed to be optimal.
- Validation-based threshold tuning improved the precision-recall trade-off.
- Recommended threshold: **{best_run.threshold:.2f}**

## Slide 12: Business Recommendations
- Use model scores for risk segmentation.
- Flag high-risk cases for manual review.
- Recalibrate the model periodically.

## Slide 13: Streamlit App
- Data overview and EDA pages
- Cleaning and leakage explanation page
- Model comparison page
- Prediction page for new borrower profiles

## Slide 14: Conclusion
- A leak-safe, interpretable credit-risk workflow was built successfully.
- Behavioural transaction features can support more disciplined lending decisions.
"""
    (REPORTS_DIR / "presentation_outline.md").write_text(outline, encoding="utf-8")


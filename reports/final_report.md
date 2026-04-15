# Xente Loan Default Project Report

Prepared by **OMODING ISAAC (B31331)**

## Executive Summary

This project built a leakage-aware binary classification solution to predict whether a Xente borrower is likely to default on a loan. The modelling workflow prioritised business realism by excluding post-outcome fields, engineering transaction-history features from prior behaviour, and validating the models with a time-based split.

The strongest validation model was **Random Forest**, evaluated at an optimized probability threshold of **0.30**.

## Best Validation Metrics

- Accuracy: **0.936**
- Precision: **0.731**
- Recall: **0.980**
- F1-score: **0.838**
- ROC-AUC: **0.972**

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

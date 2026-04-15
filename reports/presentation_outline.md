# Xente Loan Default Prediction Presentation

Prepared by **OMODING ISAAC (B31331)**

## Slide 1: Title
- Xente Loan Default Prediction
- Student: OMODING ISAAC
- Student Number: B31331

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
- Best model: **Random Forest**
- Accuracy: **0.936**
- Precision: **0.731**
- Recall: **0.980**
- F1-score: **0.838**
- ROC-AUC: **0.972**

## Slide 11: Threshold Discussion
- The standard 0.50 cut-off was not assumed to be optimal.
- Validation-based threshold tuning improved the precision-recall trade-off.
- Recommended threshold: **0.30**

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

# Xente Loan Default Prediction Project

Prepared by **OMODING ISAAC (B31331)**

## Project Summary

This project builds a leakage-aware binary classification workflow to predict whether a borrower is likely to default on a Xente loan using transaction and loan-linked behaviour data. The solution includes:

- premium exploratory visuals
- cleaned and documented modelling outputs
- two compared classification models
- a selected final model with saved artifacts
- Jupyter notebooks for each major project phase
- a Streamlit application for exploration and borrower scoring

## Project Structure

- `data/` contains the four source CSV files.
- `notebooks/` contains the four project notebooks.
- `src/` contains the reproducible pipeline modules.
- `outputs/` stores figures, metrics, cleaned datasets, and predictions.
- `models/` stores the saved final model and metadata.
- `reports/` stores the report and presentation outline.
- `app/` contains the Streamlit app.

## How To Run

1. Install dependencies:

```powershell
C:\Users\USER\venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. Regenerate the analysis pipeline:

```powershell
C:\Users\USER\venv\Scripts\python.exe -m src.run_pipeline
```

3. Build the notebooks:

```powershell
C:\Users\USER\venv\Scripts\python.exe -m src.build_notebooks
```

4. Launch the Streamlit app:

```powershell
C:\Users\USER\venv\Scripts\streamlit.exe run streamlit_app.py
```

## Modelling Notes

- The supervised target is `IsDefaulted`.
- Only labelled rows from `TRAIN.csv` are used for model training.
- Leakage-prone repayment variables are excluded from modelling.
- The validation strategy is time-aware to mirror real credit scoring conditions.
- The final deliverable uses a saved model and metadata so the app does not retrain on each run.
- For Streamlit Community Cloud, use `streamlit_app.py` as the app entry point.

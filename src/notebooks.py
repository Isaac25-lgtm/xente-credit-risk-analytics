from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from .config import NOTEBOOKS_DIR, STUDENT_ID, STUDENT_NAME


COMMON_IMPORTS = """from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

BASE_DIR = Path.cwd()
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
CLEANED_DIR = BASE_DIR / "outputs" / "cleaned"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "predictions"
"""


def _image_cell(filename: str, width: int = 1100) -> str:
    return f"display(Image(filename=str(FIGURES_DIR / '{filename}'), width={width}))"


def build_notebooks() -> None:
    notebooks = {
        "01_data_understanding.ipynb": [
            new_markdown_cell(
                f"# Xente Loan Default Project\n## 01. Data Understanding\n\nPrepared by **{STUDENT_NAME} ({STUDENT_ID})**"
            ),
            new_markdown_cell(
                "This notebook introduces the business problem, inspects the datasets, and clarifies the modelling unit used in the project."
            ),
            new_code_cell(COMMON_IMPORTS),
            new_code_cell("dataset_summary = pd.read_csv(METRICS_DIR / 'dataset_summary.csv')\ndataset_summary"),
            new_code_cell("modelling_unit = pd.read_csv(METRICS_DIR / 'modelling_unit_summary.csv')\nmodelling_unit"),
            new_code_cell("variable_summary = pd.read_csv(METRICS_DIR / 'variable_summary.csv')\nvariable_summary.head(20)"),
            new_code_cell("model_df = pd.read_csv(CLEANED_DIR / 'modelling_dataset.csv')\nmodel_df[['IsDefaulted']].value_counts().rename('count').reset_index(name='count')"),
            new_code_cell(_image_cell("target_distribution.png")),
        ],
        "02_cleaning_and_eda.ipynb": [
            new_markdown_cell(
                f"# Xente Loan Default Project\n## 02. Cleaning and Exploratory Data Analysis\n\nPrepared by **{STUDENT_NAME} ({STUDENT_ID})**"
            ),
            new_markdown_cell(
                "This notebook documents missingness, outlier treatment, target balance, key distributions, and the most important descriptive visuals."
            ),
            new_code_cell(COMMON_IMPORTS),
            new_code_cell("missingness = pd.read_csv(METRICS_DIR / 'missingness_summary.csv')\nmissingness.head(15)"),
            new_code_cell("outlier_bounds = pd.read_csv(METRICS_DIR / 'outlier_bounds.csv')\noutlier_bounds.head(15)"),
            new_code_cell(_image_cell("missingness_top_columns.png")),
            new_code_cell(_image_cell("distribution_Amount.png")),
            new_code_cell(_image_cell("distribution_Value.png")),
            new_code_cell(_image_cell("categories_ProductCategory.png")),
            new_code_cell(_image_cell("monthly_trends.png")),
        ],
        "03_feature_engineering.ipynb": [
            new_markdown_cell(
                f"# Xente Loan Default Project\n## 03. Feature Engineering and Selection\n\nPrepared by **{STUDENT_NAME} ({STUDENT_ID})**"
            ),
            new_markdown_cell(
                "This notebook explains how behavioural, temporal, and transaction-intensity features were engineered, and it records the explicit leakage exclusions used before modelling."
            ),
            new_code_cell(COMMON_IMPORTS),
            new_code_cell("leakage = pd.read_csv(METRICS_DIR / 'leakage_and_id_exclusions.csv')\nleakage"),
            new_code_cell("feature_selection = pd.read_csv(METRICS_DIR / 'feature_selection_summary.csv')\nfeature_selection.head(30)"),
            new_code_cell("history_features = pd.read_csv(CLEANED_DIR / 'customer_history_features.csv')\nhistory_features.head()"),
            new_code_cell("numeric_tests = pd.read_csv(METRICS_DIR / 'numeric_relationship_tests.csv')\nnumeric_tests"),
            new_code_cell("categorical_tests = pd.read_csv(METRICS_DIR / 'categorical_relationship_tests.csv')\ncategorical_tests"),
            new_code_cell(_image_cell("relationship_numeric_Amount.png")),
            new_code_cell(_image_cell("relationship_categorical_ProductCategory.png")),
        ],
        "04_modeling_and_evaluation.ipynb": [
            new_markdown_cell(
                f"# Xente Loan Default Project\n## 04. Modelling and Evaluation\n\nPrepared by **{STUDENT_NAME} ({STUDENT_ID})**"
            ),
            new_markdown_cell(
                "This notebook compares the candidate models, discusses threshold tuning, highlights the most influential variables, and previews the generated test predictions."
            ),
            new_code_cell(COMMON_IMPORTS),
            new_code_cell("metrics = pd.read_csv(METRICS_DIR / 'model_comparison_metrics.csv')\nmetrics"),
            new_code_cell("classification_reports = pd.read_csv(METRICS_DIR / 'classification_reports.csv')\nclassification_reports"),
            new_code_cell("thresholds = pd.read_csv(METRICS_DIR / 'threshold_analysis.csv')\nthresholds.head()"),
            new_code_cell(_image_cell("model_comparison_metrics.png")),
            new_code_cell(_image_cell("roc_curve_comparison.png")),
            new_code_cell(_image_cell("threshold_tradeoff.png")),
            new_code_cell(_image_cell("best_model_feature_effects.png")),
            new_code_cell("predictions = pd.read_csv(PREDICTIONS_DIR / 'test_predictions.csv')\npredictions.head(15)"),
        ],
    }

    for notebook_name, cells in notebooks.items():
        notebook = new_notebook(cells=cells)
        notebook_path = NOTEBOOKS_DIR / notebook_name
        nbformat.write(notebook, notebook_path)
        client = NotebookClient(notebook, timeout=600, kernel_name="xente_project")
        client.execute(cwd=str(Path.cwd()))
        nbformat.write(notebook, notebook_path)

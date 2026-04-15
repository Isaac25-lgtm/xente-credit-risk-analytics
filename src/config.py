from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CLEANED_DIR = OUTPUTS_DIR / "cleaned"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
APP_DIR = BASE_DIR / "app"

STUDENT_NAME = "OMODING ISAAC"
STUDENT_ID = "B31331"
RANDOM_STATE = 42
VALIDATION_SHARE = 0.20

PREMIUM_COLORS = {
    "ink": "#12263A",
    "teal": "#157A6E",
    "gold": "#C08B30",
    "rose": "#C65F5C",
    "sand": "#F6F0E8",
    "mist": "#E8EDF2",
    "slate": "#627286",
    "success": "#2E8B57",
}

LEAKAGE_COLUMNS = [
    "PaidOnDate",
    "IsFinalPayBack",
    "DueDate",
    "PayBackId",
    "IsThirdPartyConfirmed",
]

IDENTIFIER_COLUMNS = [
    "CustomerId",
    "TransactionId",
    "BatchId",
    "SubscriptionId",
    "LoanId",
    "LoanApplicationId",
    "ThirdPartyId",
]


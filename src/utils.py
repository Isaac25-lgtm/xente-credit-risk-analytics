from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    APP_DIR,
    CLEANED_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PREMIUM_COLORS,
    REPORTS_DIR,
)


def ensure_directories() -> None:
    for directory in [
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        CLEANED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        NOTEBOOKS_DIR,
        APP_DIR / "model",
        APP_DIR / "assets",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def apply_visual_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": PREMIUM_COLORS["sand"],
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": PREMIUM_COLORS["mist"],
            "axes.labelcolor": PREMIUM_COLORS["ink"],
            "axes.titlecolor": PREMIUM_COLORS["ink"],
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "font.family": "DejaVu Serif",
            "grid.color": PREMIUM_COLORS["mist"],
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "text.color": PREMIUM_COLORS["ink"],
            "xtick.color": PREMIUM_COLORS["slate"],
            "ytick.color": PREMIUM_COLORS["slate"],
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


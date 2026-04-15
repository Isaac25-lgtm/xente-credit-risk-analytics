from __future__ import annotations

import pandas as pd

from .analysis import (
    build_categorical_distribution_plots,
    build_missingness_plot,
    build_numeric_distribution_plots,
    build_relationship_plots,
    build_target_distribution_plot,
    build_time_trend_plots,
    run_relationship_tests,
)
from .data_prep import (
    build_history_frame,
    build_modelling_frames,
    clip_outliers,
    compute_history_features,
    dataset_summary,
    feature_lists,
    leakage_and_id_summary,
    load_datasets,
    missingness_table,
    modelling_unit_summary,
    parse_datasets,
    time_based_split,
    variable_summary,
)
from .modeling import (
    build_model_comparison_plots,
    extract_feature_effects,
    plot_threshold_tradeoff,
    refit_best_model,
    train_models,
    write_classification_reports,
)
from .reporting import write_business_report, write_presentation_outline
from .utils import apply_visual_style, ensure_directories


def run_pipeline() -> None:
    ensure_directories()
    apply_visual_style()

    datasets = parse_datasets(load_datasets())
    train = datasets["train"]
    test = datasets["test"]
    unlinked = datasets["unlinked"]
    variable_definitions = datasets["variable_definitions"]

    dataset_summary(train, test, unlinked)
    modelling_unit_summary(train)
    leakage_and_id_summary()

    labelled_preview = train.loc[train["IsDefaulted"].notna()].copy()
    labelled_preview["IsDefaulted"] = labelled_preview["IsDefaulted"].astype(int)
    variable_summary(labelled_preview, variable_definitions)

    history = build_history_frame(train, test, unlinked)
    history_features = compute_history_features(history)
    model_df, test_df = build_modelling_frames(train, test, history_features)

    numeric_features, categorical_features, _ = feature_lists(model_df)

    missing = missingness_table(model_df)
    build_missingness_plot(missing)
    build_target_distribution_plot(model_df["IsDefaulted"])
    build_numeric_distribution_plots(model_df, numeric_features)
    build_categorical_distribution_plots(model_df, categorical_features)
    build_time_trend_plots(model_df)
    run_relationship_tests(model_df, numeric_features, categorical_features)
    build_relationship_plots(model_df, numeric_features, categorical_features)

    train_df, valid_df = time_based_split(model_df)
    clipped_train, clipped_valid, clipped_test, _ = clip_outliers(
        train_df, valid_df, test_df, numeric_features
    )

    runs, y_valid = train_models(clipped_train, clipped_valid, numeric_features, categorical_features)
    write_classification_reports(runs, y_valid)
    metrics_df = build_model_comparison_plots(runs, y_valid)

    best_run = max(runs, key=lambda run: (run.metrics["f1"], run.metrics["roc_auc"]))
    plot_threshold_tradeoff(best_run.name, y_valid, best_run.y_valid_proba)
    extract_feature_effects(best_run)

    full_clipped_model, _, full_clipped_test, _ = clip_outliers(
        model_df, clipped_valid, test_df, numeric_features
    )
    refit_best_model(best_run, full_clipped_model, full_clipped_test, numeric_features, categorical_features)

    write_business_report(best_run, metrics_df)
    write_presentation_outline(best_run, metrics_df)


if __name__ == "__main__":
    run_pipeline()

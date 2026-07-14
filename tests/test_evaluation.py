from __future__ import annotations

import math

import polars as pl

from just_rna.evaluation import evaluate_predictions


def test_regression_metrics_match_exact_values_and_groups() -> None:
    predictions = pl.DataFrame(
        {
            "prediction": [2.0, 2.0, 4.0, 4.0],
            "ground_truth": [1.0, 2.0, 3.0, 4.0],
            "tissue": ["a", "a", "b", "b"],
        }
    )

    report = evaluate_predictions(predictions, group_by=("tissue",))

    assert report.overall.samples == 4
    assert math.isclose(report.overall.mae, 0.5)
    assert math.isclose(report.overall.rmse, math.sqrt(0.5))
    assert math.isclose(report.overall.median_absolute_error, 0.5)
    assert math.isclose(report.overall.mean_error, 0.5)
    assert math.isclose(report.overall.r_squared, 0.6)
    assert report.overall.pearson is not None
    assert math.isclose(report.overall.pearson, 0.8944271909999159)
    assert report.overall.spearman is not None
    assert math.isclose(report.overall.spearman, 0.8944271909999159)
    assert len(report.groups) == 2
    assert {group.metrics.samples for group in report.groups} == {2}


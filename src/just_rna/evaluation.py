"""Typed regression evaluation for clock predictions."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from scipy.stats import spearmanr

from just_rna.exceptions import JustRnaError
from just_rna.models import MetadataItem
from just_rna.models import MetadataValue


class EvaluationError(JustRnaError, ValueError):
    """Predictions cannot be evaluated against the requested ground truth."""


class RegressionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int
    mae: float
    rmse: float
    median_absolute_error: float
    mean_error: float
    r_squared: float | None
    pearson: float | None
    spearman: float | None


class GroupMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group: tuple[MetadataItem, ...]
    metrics: RegressionMetrics


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_column: str
    ground_truth_column: str
    overall: RegressionMetrics
    groups: tuple[GroupMetrics, ...] = ()

    def to_polars(self) -> pl.DataFrame:
        rows = [_metrics_row("overall", (), self.overall)]
        rows.extend(
            _metrics_row("group", group.group, group.metrics)
            for group in self.groups
        )
        return pl.DataFrame(rows, infer_schema_length=None)

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_csv(path)
        return path


def evaluate_predictions(
    predictions: pl.DataFrame,
    *,
    prediction_column: str = "prediction",
    ground_truth_column: str = "ground_truth",
    group_by: tuple[str, ...] = (),
) -> EvaluationReport:
    """Compute regression metrics from finite prediction/ground-truth pairs."""

    required = {prediction_column, ground_truth_column, *group_by}
    missing = required.difference(predictions.columns)
    if missing:
        raise EvaluationError(f"Missing evaluation columns: {sorted(missing)}")
    evaluation = predictions.with_columns(
        pl.col(prediction_column).cast(pl.Float64, strict=True),
        pl.col(ground_truth_column).cast(pl.Float64, strict=True),
    ).filter(
        pl.col(prediction_column).is_finite()
        & pl.col(ground_truth_column).is_finite()
    )
    if evaluation.is_empty():
        raise EvaluationError("No finite prediction/ground-truth pairs")

    groups = tuple(
        _evaluate_group(evaluation, values, group_by, prediction_column, ground_truth_column)
        for values in evaluation.select(group_by).unique(maintain_order=True).iter_rows()
    ) if group_by else ()
    return EvaluationReport(
        prediction_column=prediction_column,
        ground_truth_column=ground_truth_column,
        overall=_regression_metrics(
            evaluation.get_column(prediction_column).to_numpy(),
            evaluation.get_column(ground_truth_column).to_numpy(),
        ),
        groups=groups,
    )


def _evaluate_group(
    evaluation: pl.DataFrame,
    values: tuple[object, ...],
    group_by: tuple[str, ...],
    prediction_column: str,
    ground_truth_column: str,
) -> GroupMetrics:
    condition = pl.lit(True)
    for column, value in zip(group_by, values, strict=True):
        condition &= pl.col(column).eq_missing(value)
    frame = evaluation.filter(condition)
    return GroupMetrics(
        group=tuple(
            MetadataItem(key=column, value=cast(MetadataValue, value))
            for column, value in zip(group_by, values, strict=True)
        ),
        metrics=_regression_metrics(
            frame.get_column(prediction_column).to_numpy(),
            frame.get_column(ground_truth_column).to_numpy(),
        ),
    )


def _regression_metrics(
    predictions: NDArray[np.float64],
    ground_truth: NDArray[np.float64],
) -> RegressionMetrics:
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(ground_truth, dtype=np.float64)
    errors = predicted - observed
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)
    centered = observed - np.mean(observed)
    total_sum_squares = float(np.sum(np.square(centered)))
    residual_sum_squares = float(np.sum(squared_errors))
    has_correlation = len(observed) >= 2 and bool(np.std(observed) > 0.0)
    prediction_varies = bool(np.std(predicted) > 0.0)
    return RegressionMetrics(
        samples=len(observed),
        mae=float(np.mean(absolute_errors)),
        rmse=float(np.sqrt(np.mean(squared_errors))),
        median_absolute_error=float(np.median(absolute_errors)),
        mean_error=float(np.mean(errors)),
        r_squared=(
            None
            if total_sum_squares == 0.0
            else 1.0 - residual_sum_squares / total_sum_squares
        ),
        pearson=(
            float(np.corrcoef(observed, predicted)[0, 1])
            if has_correlation and prediction_varies
            else None
        ),
        spearman=(
            float(cast(float, spearmanr(observed, predicted).statistic))
            if has_correlation and prediction_varies
            else None
        ),
    )


def _metrics_row(
    scope: str,
    group: tuple[MetadataItem, ...],
    metrics: RegressionMetrics,
) -> dict[str, MetadataValue]:
    return {
        "scope": scope,
        **{item.key: item.value for item in group},
        "samples": metrics.samples,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "median_absolute_error": metrics.median_absolute_error,
        "mean_error": metrics.mean_error,
        "r_squared": metrics.r_squared,
        "pearson": metrics.pearson,
        "spearman": metrics.spearman,
    }


__all__ = [
    "EvaluationError",
    "EvaluationReport",
    "GroupMetrics",
    "RegressionMetrics",
    "evaluate_predictions",
]


"""Explicit end-to-end workflow boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from just_rna.assets import download_asset
from just_rna.clocks.base import run_clock
from just_rna.clocks.tage.clock import TAgeClock
from just_rna.clocks.tage.clock import TAgeConfig
from just_rna.clocks.tage.prediction import load_model
from just_rna.config import load_settings
from just_rna.data import Species
from just_rna.datasets.gse225576 import MouseTissue
from just_rna.datasets.gse225576 import load_gse225576
from just_rna.evaluation import EvaluationReport
from just_rna.evaluation import evaluate_predictions
from just_rna.models import ClockResult
from just_rna.models import PredictionProvenance


@dataclass(frozen=True, slots=True)
class Gse225576PredictionRun:
    results: tuple[ClockResult, ...]
    output: Path
    metrics: EvaluationReport
    metrics_output: Path


def predict_gse225576_tage(
    dataset_directory: Path,
    output: Path,
    *,
    tissues: tuple[MouseTissue, ...] = tuple(MouseTissue),
    cache_directory: Path | None = None,
    dotenv_path: Path | None = None,
    metrics_output: Path | None = None,
) -> Gse225576PredictionRun:
    """Download real age-annotated cohorts, run tAge, and save predictions."""

    settings = load_settings(dotenv_path)
    asset_cache = settings.cache_directory if cache_directory is None else cache_directory
    cohorts = load_gse225576(dataset_directory, tissues=tissues)
    clock = TAgeClock()
    model_path = download_asset(clock.spec.asset_id, cache_dir=asset_cache)
    model = load_model(model_path)
    config = TAgeConfig(
        species=Species.MOUSE,
        control_group_column="age_months",
        control_group_label="6",
    )
    results = tuple(
        run_clock(
            clock,
            cohort.dataset,
            config,
            model,
            PredictionProvenance.create(
                model_id=clock.spec.id,
                source=cohort.dataset.source,
                asset_path=model_path,
            ),
        )
        for cohort in cohorts
    )
    combined = pl.concat(
        (result.to_polars() for result in results),
        how="diagonal_relaxed",
    ).with_columns(
        (pl.col("age_months") - 6).cast(pl.Float64).alias("ground_truth"),
        pl.lit("months_since_6_month_control").alias("ground_truth_unit"),
    )
    evaluation = evaluate_predictions(combined, group_by=("tissue",))
    resolved_metrics_output = (
        output.with_name(f"{output.stem}_metrics.csv")
        if metrics_output is None
        else metrics_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        combined.write_parquet(output)
    elif output.suffix == ".csv":
        combined.write_csv(output)
    else:
        raise ValueError("Output must end in .csv or .parquet")
    evaluation.write_csv(resolved_metrics_output)
    return Gse225576PredictionRun(
        results=results,
        output=output,
        metrics=evaluation,
        metrics_output=resolved_metrics_output,
    )


__all__ = ["Gse225576PredictionRun", "predict_gse225576_tage"]


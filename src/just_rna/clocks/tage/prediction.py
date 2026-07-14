from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast
import warnings

import joblib  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from just_rna.data import Species

PREDICTION_SPECIES_MULTIPLIERS: dict[Species, float] = {
    Species.HUMAN: 122.5,
    Species.MOUSE: 48.0,
    Species.RAT: 50.4,
    Species.MONKEY: 39.0,
}


class TAgePredictor(Protocol):
    def predict(self, values: object, **kwargs: object) -> object: ...


def _patch_simple_imputers(model: object) -> None:
    if not isinstance(model, Pipeline):
        return
    for _, step in model.steps:
        if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
            statistics = cast(
                NDArray[np.float64] | None,
                getattr(step, "statistics_", None),
            )
            fill_dtype: np.dtype[np.float64] | type[np.float64] = (
                statistics.dtype if statistics is not None else np.float64
            )
            setattr(step, "_fill_dtype", fill_dtype)


def _feature_names(model: object) -> tuple[str, ...]:
    raw_features = getattr(model, "feature_names_in_", None)
    if raw_features is None:
        raw_features = getattr(model, "feature_names", None)
    if raw_features is None:
        raise AttributeError(
            "Clock model has neither feature_names_in_ nor feature_names"
        )
    return tuple(str(feature) for feature in cast(Sequence[object], raw_features))


def load_model(model_path: str | Path) -> tuple[TAgePredictor, tuple[str, ...]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = joblib.load(path)  # pyright: ignore[reportUnknownMemberType]
    _patch_simple_imputers(model)
    return cast(TAgePredictor, model), _feature_names(model)


def align_model_features(
    expression: pl.DataFrame,
    features: Sequence[str],
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    if gene_column not in expression.columns:
        raise ValueError(f"Gene column {gene_column!r} is missing")
    sample_columns = [column for column in expression.columns if column != gene_column]
    genes = expression.get_column(gene_column).cast(pl.String).to_list()
    positions = {gene: index for index, gene in enumerate(genes)}
    values = expression.select(sample_columns).cast(pl.Float64, strict=True).to_numpy()
    aligned = np.full((len(sample_columns), len(features)), np.nan, dtype=np.float64)
    for feature_index, feature in enumerate(features):
        source_index = positions.get(feature)
        if source_index is not None:
            aligned[:, feature_index] = values[source_index, :]
    return pl.DataFrame(
        {
            feature: np.asarray(aligned[:, index], dtype=np.float64)
            for index, feature in enumerate(features)
        }
    )


def predict_tage(
    model_path: str | Path,
    expression: pl.DataFrame,
    *,
    species: Species | str,
    metadata: pl.DataFrame | None = None,
    return_std: bool = False,
    prefix: str = "",
    gene_column: str = "gene",
) -> pl.DataFrame:
    model, features = load_model(model_path)
    return predict_with_model(
        model,
        features,
        expression,
        species=species,
        metadata=metadata,
        return_std=return_std,
        prefix=prefix,
        gene_column=gene_column,
    )


def predict_with_model(
    model: TAgePredictor,
    features: Sequence[str],
    expression: pl.DataFrame,
    *,
    species: Species | str,
    metadata: pl.DataFrame | None = None,
    return_std: bool = False,
    prefix: str = "",
    gene_column: str = "gene",
) -> pl.DataFrame:
    """Apply an already-loaded estimator without performing I/O."""

    model_input = align_model_features(
        expression,
        features,
        gene_column=gene_column,
    )
    prediction_result = (
        model.predict(model_input, return_std=True)
        if return_std
        else model.predict(model_input)
    )
    standard_deviation: NDArray[np.float64] | None = None
    if return_std:
        prediction, raw_standard_deviation = cast(tuple[object, object], prediction_result)
        standard_deviation = np.asarray(raw_standard_deviation, dtype=np.float64)
    else:
        prediction = prediction_result
    ages = np.asarray(prediction, dtype=np.float64)
    sample_columns = [
        column for column in expression.columns if column != gene_column
    ]
    output = (
        pl.DataFrame({"sample": sample_columns})
        if metadata is None
        else metadata.clone()
    )
    if output.height != len(sample_columns):
        raise ValueError("Metadata rows must match expression sample columns")
    multiplier = PREDICTION_SPECIES_MULTIPLIERS[Species(species)]
    output = output.with_columns(
        pl.Series(f"{prefix}tAge", ages * multiplier, dtype=pl.Float64)
    )
    if standard_deviation is not None:
        output = output.with_columns(
            pl.Series(
                f"{prefix}tAge_std",
                standard_deviation * multiplier,
                dtype=pl.Float64,
            )
        )
    return output

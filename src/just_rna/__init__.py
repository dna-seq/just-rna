"""Type-safe transcriptomic aging-clock API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from just_rna.data import ExpressionDataset
from just_rna.data import ExpressionKind
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import Species
from just_rna.data import read_expression
from just_rna.models import ClockId
from just_rna.models import ClockResult
from just_rna.models import SamplePrediction
from just_rna.quantification import read_salmon
from just_rna.quantification import read_transcript_quantifications

if TYPE_CHECKING:
    from just_rna.clocks.tage.clock import TAgeConfig


def predict(
    dataset: ExpressionDataset,
    *,
    species: Species,
    clock: ClockId = ClockId.TAGE,
    model: str | None = None,
    config: TAgeConfig | None = None,
    cache_directory: Path | None = None,
) -> ClockResult:
    """Lazy public facade; model/download dependencies load only when called."""

    from just_rna.prediction import predict as predict_impl

    return predict_impl(
        dataset,
        species=species,
        clock=clock,
        model=model,
        config=config,
        cache_directory=cache_directory,
    )


def predict_one(
    dataset: ExpressionDataset,
    *,
    species: Species,
    clock: ClockId = ClockId.TAGE,
    model: str | None = None,
    config: TAgeConfig | None = None,
    cache_directory: Path | None = None,
) -> SamplePrediction:
    from just_rna.prediction import predict_one as predict_one_impl

    return predict_one_impl(
        dataset,
        species=species,
        clock=clock,
        model=model,
        config=config,
        cache_directory=cache_directory,
    )


def main() -> None:
    from just_rna.cli import main as cli_main

    cli_main()


__all__ = [
    "ClockId",
    "ClockResult",
    "ExpressionDataset",
    "ExpressionKind",
    "ExpressionScale",
    "GeneIdType",
    "SamplePrediction",
    "Species",
    "main",
    "predict",
    "predict_one",
    "read_expression",
    "read_salmon",
    "read_transcript_quantifications",
]

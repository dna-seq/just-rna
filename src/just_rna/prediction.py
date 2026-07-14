"""Public prediction facade and effect boundary."""

from __future__ import annotations

from pathlib import Path

from just_rna.assets import download_asset
from just_rna.clocks.base import run_clock
from just_rna.clocks.tage.clock import DEFAULT_TAGE_MODEL_ID
from just_rna.clocks.tage.clock import TAgeConfig
from just_rna.clocks.tage.prediction import load_model
from just_rna.data import ExpressionDataset
from just_rna.data import Species
from just_rna.exceptions import UnknownModelError
from just_rna.models import ClockId
from just_rna.models import ClockResult
from just_rna.models import PredictionProvenance
from just_rna.models import SamplePrediction
from just_rna.registry import get_clock


def predict(
    dataset: ExpressionDataset,
    *,
    species: Species,
    clock: ClockId = ClockId.TAGE,
    model: str | None = None,
    config: TAgeConfig | None = None,
    cache_directory: Path | None = None,
) -> ClockResult:
    """Resolve assets at the boundary, then execute the pure clock core."""

    implementation = get_clock(clock)
    model_id = DEFAULT_TAGE_MODEL_ID if model is None else model
    if model_id != implementation.spec.id:
        raise UnknownModelError(
            f"Unknown {clock.value} model {model_id!r}; "
            f"available model: {implementation.spec.id!r}"
        )
    run_config = TAgeConfig(species=species) if config is None else config
    if run_config.species is not species:
        raise ValueError(
            "The explicit species must match config.species"
        )
    model_path = download_asset(
        implementation.spec.asset_id,
        cache_dir=cache_directory,
    )
    loaded_model = load_model(model_path)
    provenance = PredictionProvenance.create(
        model_id=implementation.spec.id,
        source=dataset.source,
        asset_path=model_path,
    )
    return run_clock(
        implementation,
        dataset,
        run_config,
        loaded_model,
        provenance,
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
    return predict(
        dataset,
        species=species,
        clock=clock,
        model=model,
        config=config,
        cache_directory=cache_directory,
    ).one()


__all__ = ["predict", "predict_one"]


"""Typed tAge clock implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from typing import cast

import polars as pl

from just_rna.clocks.base import BaseClock
from just_rna.clocks.base import PreparedInput
from just_rna.clocks.tage.prediction import PREDICTION_SPECIES_MULTIPLIERS
from just_rna.clocks.tage.prediction import TAgePredictor
from just_rna.clocks.tage.prediction import predict_with_model
from just_rna.clocks.tage.preprocessing import align_to_gene_list
from just_rna.clocks.tage.preprocessing import control_median_subtract
from just_rna.clocks.tage.preprocessing import filter_genes
from just_rna.clocks.tage.preprocessing import load_gene_list
from just_rna.clocks.tage.preprocessing import log10_transform
from just_rna.clocks.tage.preprocessing import map_genes
from just_rna.clocks.tage.preprocessing import normalize_counts
from just_rna.clocks.tage.preprocessing import scale_columns
from just_rna.clocks.tage.preprocessing import yugene
from just_rna.data import ExpressionDataset
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import Species
from just_rna.models import Age
from just_rna.models import ClockDetails
from just_rna.models import ClockId
from just_rna.models import ClockModelSpec
from just_rna.models import ClockRunConfig
from just_rna.models import ClockWarning
from just_rna.models import MetadataItem
from just_rna.models import MetadataValue
from just_rna.models import PredictionTarget
from just_rna.models import PreprocessingReport
from just_rna.models import SamplePrediction
from just_rna.models import TAgeDetails
from just_rna.models import TimeUnit

DEFAULT_TAGE_MODEL_ID = "EN_Chronoage_Multispecies_Multitissue_scaleddiff"
DEFAULT_TAGE_ASSET_ID = (
    "tage:zenodo:EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl"
)


class TAgeTransform(StrEnum):
    SCALED_DIFF = "scaled_diff"
    YUGENE_DIFF = "yugene_diff"


class TAgeConfig(ClockRunConfig):
    transform: TAgeTransform = TAgeTransform.SCALED_DIFF
    count_threshold: float = 10.0
    percent_threshold: float = 20.0
    control_group_column: str | None = None
    control_group_label: str | None = None


@dataclass(frozen=True, slots=True)
class TAgePrepared:
    expression: pl.DataFrame


LoadedTAgeModel = tuple[TAgePredictor, tuple[str, ...]]


DEFAULT_TAGE_SPEC = ClockModelSpec(
    id=DEFAULT_TAGE_MODEL_ID,
    clock=ClockId.TAGE,
    target=PredictionTarget.CHRONOLOGICAL_AGE,
    accepted_scales=frozenset(
        {
            ExpressionScale.RAW_COUNTS,
            ExpressionScale.ESTIMATED_COUNTS,
            ExpressionScale.NORMALIZED_COUNTS,
        }
    ),
    accepted_gene_ids=frozenset({GeneIdType.SYMBOL, GeneIdType.ENSEMBL}),
    supported_species=frozenset(Species),
    asset_id=DEFAULT_TAGE_ASSET_ID,
)


class TAgeClock(
    BaseClock[TAgeConfig, TAgePrepared, LoadedTAgeModel],
):
    def __init__(self, spec: ClockModelSpec = DEFAULT_TAGE_SPEC) -> None:
        self._spec = spec

    @property
    def spec(self) -> ClockModelSpec:
        return self._spec

    def validate_input(
        self,
        dataset: ExpressionDataset,
        config: TAgeConfig,
    ) -> tuple[ClockWarning, ...]:
        if (config.control_group_column is None) != (
            config.control_group_label is None
        ):
            raise ValueError(
                "control_group_column and control_group_label must be provided together"
            )
        if config.control_group_column is None:
            raise ValueError(
                f"The {config.transform.value} model requires an explicit control "
                "group column and label"
            )
        if dataset.metadata is None:
            raise ValueError("Metadata is required for control subtraction")
        if config.control_group_column not in dataset.metadata.columns:
            raise ValueError(
                f"Metadata column {config.control_group_column!r} is missing"
            )
        labels = (
            dataset.metadata.get_column(config.control_group_column)
            .cast(pl.String)
            .to_list()
        )
        if config.control_group_label in labels:
            return _scale_warnings(dataset)
        return _scale_warnings(dataset) + (
            ClockWarning(
                code="control-group-not-found",
                message=(
                    f"Control label {config.control_group_label!r} was absent; "
                    "all samples were used as the reference group"
                ),
            ),
        )

    def preprocess(
        self,
        dataset: ExpressionDataset,
        config: TAgeConfig,
    ) -> PreparedInput[TAgePrepared]:
        input_genes = dataset.expression.height
        filtered = filter_genes(
            dataset.expression,
            gene_column=dataset.gene_column,
            count_threshold=config.count_threshold,
            percent_threshold=config.percent_threshold,
        )
        mapping_type = cast(
            Literal["Gene.Symbol", "Ensembl"],
            dataset.gene_id_type.value,
        )
        mapped = map_genes(
            filtered,
            config.species,
            mapping_type,
            gene_column=dataset.gene_column,
        )
        normalized = (
            normalize_counts(
                mapped,
                method="RLE",
                gene_column=dataset.gene_column,
            )
            if dataset.scale
            in {
                ExpressionScale.RAW_COUNTS,
                ExpressionScale.ESTIMATED_COUNTS,
            }
            else mapped
        )
        logged = log10_transform(normalized, gene_column=dataset.gene_column)
        scaled = scale_columns(logged, gene_column=dataset.gene_column)
        transformed = (
            scaled
            if config.transform is TAgeTransform.SCALED_DIFF
            else yugene(scaled, gene_column=dataset.gene_column)
        )
        genes_before_alignment = frozenset(
            transformed.get_column(dataset.gene_column).cast(pl.String).to_list()
        )
        gene_list = load_gene_list()
        aligned = align_to_gene_list(
            transformed,
            gene_list,
            gene_column=dataset.gene_column,
        )
        adjusted = control_median_subtract(
            aligned,
            dataset.metadata,
            control_group_column=config.control_group_column,
            control_group_label=config.control_group_label,
            gene_column=dataset.gene_column,
        )
        missing = tuple(gene for gene in gene_list if gene not in genes_before_alignment)
        report = PreprocessingReport(
            input_genes=input_genes,
            retained_genes=filtered.height,
            mapped_genes=mapped.height,
            model_features=len(gene_list),
            missing_model_features=missing,
        )
        warnings = (
            (
                ClockWarning(
                    code="missing-model-features",
                    message=(
                        f"{len(missing)} model-alignment genes were absent and "
                        "represented as NaN for model imputation"
                    ),
                ),
            )
            if missing
            else ()
        )
        return PreparedInput(
            value=TAgePrepared(expression=adjusted),
            report=report,
            warnings=warnings,
        )

    def predict_preprocessed(
        self,
        prepared: TAgePrepared,
        model: LoadedTAgeModel,
        dataset: ExpressionDataset,
        config: TAgeConfig,
    ) -> tuple[tuple[SamplePrediction, ...], ClockDetails]:
        estimator, features = model
        raw = predict_with_model(
            estimator,
            features,
            prepared.expression,
            species=config.species,
            return_std=config.return_uncertainty,
            gene_column=dataset.gene_column,
        )
        prediction_values = raw.get_column("tAge").to_list()
        uncertainty_values = (
            raw.get_column("tAge_std").to_list()
            if "tAge_std" in raw.columns
            else [None] * raw.height
        )
        unit = _species_time_unit(config.species)
        metadata_by_sample = _metadata_by_sample(dataset)
        samples = tuple(
            SamplePrediction(
                sample=sample,
                prediction=float(prediction_values[index]),
                uncertainty=_optional_float(uncertainty_values[index]),
                clock=self.spec.clock,
                model=self.spec.id,
                target=self.spec.target,
                unit=unit,
                species=config.species,
                chronological_age=_chronological_age(
                    metadata_by_sample.get(sample, ()),
                ),
                metadata=metadata_by_sample.get(sample, ()),
            )
            for index, sample in enumerate(dataset.sample_ids)
        )
        details: ClockDetails = TAgeDetails(
            transform=config.transform.value,
            species_multiplier=PREDICTION_SPECIES_MULTIPLIERS[config.species],
        )
        return samples, details


def _species_time_unit(species: Species) -> TimeUnit:
    return (
        TimeUnit.MONTHS
        if species in {Species.MOUSE, Species.RAT}
        else TimeUnit.YEARS
    )


def _scale_warnings(dataset: ExpressionDataset) -> tuple[ClockWarning, ...]:
    if dataset.scale is ExpressionScale.ESTIMATED_COUNTS:
        return (
            ClockWarning(
                code="estimated-counts",
                message=(
                    "Input contains fractional estimated counts; they are treated "
                    "as count-like values and RLE-normalized."
                ),
            ),
        )
    if dataset.scale is not ExpressionScale.NORMALIZED_COUNTS:
        return ()
    return (
        ClockWarning(
            code="pre-normalized-counts",
            message=(
                "Input is declared as normalized counts; tAge RLE normalization "
                "was skipped. Predictions may not be directly comparable with "
                "the published raw-count pipeline."
            ),
        ),
    )


def _optional_float(value: object | None) -> float | None:
    return None if value is None else float(cast(float, value))


def _metadata_by_sample(
    dataset: ExpressionDataset,
) -> dict[str, tuple[MetadataItem, ...]]:
    if dataset.metadata is None:
        return {}
    result: dict[str, tuple[MetadataItem, ...]] = {}
    for row in dataset.metadata.iter_rows(named=True):
        sample = str(row[dataset.sample_column])
        result[sample] = tuple(
            MetadataItem(key=key, value=cast(MetadataValue, value))
            for key, value in row.items()
            if key != dataset.sample_column
        )
    return result


def _chronological_age(metadata: tuple[MetadataItem, ...]) -> Age | None:
    by_key = {item.key: item.value for item in metadata}
    for key, unit in (
        ("age_months", TimeUnit.MONTHS),
        ("age_years", TimeUnit.YEARS),
        ("age_days", TimeUnit.DAYS),
    ):
        value = by_key.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return Age(value=float(value), unit=unit)
    return None


__all__ = [
    "DEFAULT_TAGE_ASSET_ID",
    "DEFAULT_TAGE_MODEL_ID",
    "DEFAULT_TAGE_SPEC",
    "LoadedTAgeModel",
    "TAgeClock",
    "TAgeConfig",
    "TAgePrepared",
    "TAgeTransform",
]


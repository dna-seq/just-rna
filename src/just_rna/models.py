"""Immutable public values shared by transcriptomic clocks."""

from __future__ import annotations

from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import Self

import polars as pl
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import Species

MetadataValue = str | int | float | bool | None
ModelId = Annotated[str, Field(min_length=1)]


class ClockId(StrEnum):
    TAGE = "tage"


class PredictionTarget(StrEnum):
    CHRONOLOGICAL_AGE = "chronological_age"
    NORMALIZED_AGE = "normalized_age"
    MORTALITY_RISK = "mortality_risk"


class TimeUnit(StrEnum):
    YEARS = "years"
    MONTHS = "months"
    DAYS = "days"
    UNITLESS = "unitless"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Age(ImmutableModel):
    value: float
    unit: TimeUnit


class MetadataItem(ImmutableModel):
    key: str
    value: MetadataValue


class ClockWarning(ImmutableModel):
    code: str
    message: str


class ClockModelSpec(ImmutableModel):
    id: ModelId
    clock: ClockId
    target: PredictionTarget
    accepted_scales: frozenset[ExpressionScale]
    accepted_gene_ids: frozenset[GeneIdType]
    supported_species: frozenset[Species]
    asset_id: str


class ClockRunConfig(ImmutableModel):
    species: Species
    return_uncertainty: bool = False


class PreprocessingReport(ImmutableModel):
    input_genes: int
    retained_genes: int
    mapped_genes: int
    model_features: int
    missing_model_features: tuple[str, ...] = ()


class PredictionProvenance(ImmutableModel):
    package_version: str
    model_id: ModelId
    source: Path | None = None
    asset_path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        source: Path | None = None,
        asset_path: Path | None = None,
    ) -> Self:
        try:
            package_version = version("just-rna")
        except PackageNotFoundError:
            package_version = "uninstalled"
        return cls(
            package_version=package_version,
            model_id=model_id,
            source=source,
            asset_path=asset_path,
        )


class GenericClockDetails(ImmutableModel):
    kind: Literal["generic"] = "generic"


class TAgeDetails(ImmutableModel):
    kind: Literal["tage"] = "tage"
    transform: Literal["scaled_diff", "yugene_diff"]
    species_multiplier: float


ClockDetails = Annotated[
    GenericClockDetails | TAgeDetails,
    Field(discriminator="kind"),
]


class SamplePrediction(ImmutableModel):
    sample: str
    prediction: float
    uncertainty: float | None = None
    clock: ClockId
    model: ModelId
    target: PredictionTarget
    unit: TimeUnit
    species: Species
    chronological_age: Age | None = None
    metadata: tuple[MetadataItem, ...] = ()

    def metadata_dict(self) -> dict[str, MetadataValue]:
        return {item.key: item.value for item in self.metadata}


class ClockResult(ImmutableModel):
    spec: ClockModelSpec
    species: Species
    samples: tuple[SamplePrediction, ...]
    preprocessing: PreprocessingReport
    provenance: PredictionProvenance
    details: ClockDetails
    warnings: tuple[ClockWarning, ...] = ()

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if not self.samples:
            raise ValueError("ClockResult must contain at least one prediction")
        identifiers = tuple(item.sample for item in self.samples)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Prediction sample identifiers must be unique")
        if any(item.clock is not self.spec.clock for item in self.samples):
            raise ValueError("Sample clock does not match result specification")
        if any(item.model != self.spec.id for item in self.samples):
            raise ValueError("Sample model does not match result specification")
        return self

    def to_samples(self) -> list[SamplePrediction]:
        return list(self.samples)

    def by_sample(self) -> dict[str, SamplePrediction]:
        return {item.sample: item for item in self.samples}

    def one(self) -> SamplePrediction:
        if len(self.samples) != 1:
            raise ValueError(
                f"Expected one prediction, received {len(self.samples)}"
            )
        return self.samples[0]

    def to_polars(self) -> pl.DataFrame:
        rows: list[dict[str, MetadataValue]] = []
        for item in self.samples:
            row = item.metadata_dict()
            row.update(
                {
                    "sample": item.sample,
                    "prediction": item.prediction,
                    "uncertainty": item.uncertainty,
                    "clock": item.clock.value,
                    "model": item.model,
                    "target": item.target.value,
                    "unit": item.unit.value,
                    "species": item.species.value,
                    "ground_truth": (
                        None
                        if item.chronological_age is None
                        else item.chronological_age.value
                    ),
                    "ground_truth_unit": (
                        None
                        if item.chronological_age is None
                        else item.chronological_age.unit.value
                    ),
                }
            )
            rows.append(row)
        return pl.DataFrame(rows, infer_schema_length=None)

    def write_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_csv(path)
        return path

    def write_parquet(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_parquet(path)
        return path


__all__ = [
    "Age",
    "ClockDetails",
    "ClockId",
    "ClockModelSpec",
    "ClockResult",
    "ClockRunConfig",
    "ClockWarning",
    "GenericClockDetails",
    "MetadataItem",
    "MetadataValue",
    "ModelId",
    "PredictionProvenance",
    "PredictionTarget",
    "PreprocessingReport",
    "SamplePrediction",
    "TAgeDetails",
    "TimeUnit",
]


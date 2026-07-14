"""Pure, generic clock orchestration."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic
from typing import TypeVar

from just_rna.data import ExpressionDataset
from just_rna.exceptions import IncompatibleExpressionScaleError
from just_rna.models import ClockDetails
from just_rna.models import ClockModelSpec
from just_rna.models import ClockResult
from just_rna.models import ClockRunConfig
from just_rna.models import ClockWarning
from just_rna.models import PredictionProvenance
from just_rna.models import PreprocessingReport
from just_rna.models import SamplePrediction

ConfigT = TypeVar("ConfigT", bound=ClockRunConfig)
PreparedT = TypeVar("PreparedT")
ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class PreparedInput(Generic[PreparedT]):
    value: PreparedT
    report: PreprocessingReport
    warnings: tuple[ClockWarning, ...] = ()


class BaseClock(ABC, Generic[ConfigT, PreparedT, ModelT]):
    """A clock whose scientific core receives all dependencies explicitly."""

    @property
    @abstractmethod
    def spec(self) -> ClockModelSpec: ...

    @abstractmethod
    def validate_input(
        self,
        dataset: ExpressionDataset,
        config: ConfigT,
    ) -> tuple[ClockWarning, ...]: ...

    @abstractmethod
    def preprocess(
        self,
        dataset: ExpressionDataset,
        config: ConfigT,
    ) -> PreparedInput[PreparedT]: ...

    @abstractmethod
    def predict_preprocessed(
        self,
        prepared: PreparedT,
        model: ModelT,
        dataset: ExpressionDataset,
        config: ConfigT,
    ) -> tuple[tuple[SamplePrediction, ...], ClockDetails]: ...


def run_clock(
    clock: BaseClock[ConfigT, PreparedT, ModelT],
    dataset: ExpressionDataset,
    config: ConfigT,
    model: ModelT,
    provenance: PredictionProvenance,
) -> ClockResult:
    """Run a clock without file, environment, network, or model-loading effects."""

    if dataset.scale not in clock.spec.accepted_scales:
        accepted = ", ".join(sorted(scale.value for scale in clock.spec.accepted_scales))
        raise IncompatibleExpressionScaleError(
            f"{clock.spec.clock.value} model {clock.spec.id!r} accepts {accepted}; "
            f"received {dataset.scale.value}"
        )
    if dataset.gene_id_type not in clock.spec.accepted_gene_ids:
        accepted = ", ".join(
            sorted(gene_id.value for gene_id in clock.spec.accepted_gene_ids)
        )
        raise IncompatibleExpressionScaleError(
            f"{clock.spec.clock.value} model {clock.spec.id!r} accepts gene IDs "
            f"{accepted}; received {dataset.gene_id_type.value}"
        )
    if config.species not in clock.spec.supported_species:
        raise IncompatibleExpressionScaleError(
            f"{clock.spec.id!r} does not support species {config.species.value}"
        )

    validation_warnings = clock.validate_input(dataset, config)
    prepared = clock.preprocess(dataset, config)
    samples, details = clock.predict_preprocessed(
        prepared.value,
        model,
        dataset,
        config,
    )
    return ClockResult(
        spec=clock.spec,
        species=config.species,
        samples=samples,
        preprocessing=prepared.report,
        provenance=provenance,
        details=details,
        warnings=validation_warnings + prepared.warnings,
    )


__all__ = ["BaseClock", "PreparedInput", "run_clock"]


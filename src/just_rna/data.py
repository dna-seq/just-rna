"""Typed, validated transcriptomic input values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl

from just_rna.exceptions import InvalidExpressionError
from just_rna.exceptions import SampleAlignmentError


class Species(StrEnum):
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    MONKEY = "monkey"


class GeneIdType(StrEnum):
    SYMBOL = "Gene.Symbol"
    ENSEMBL = "Ensembl"
    ENTREZ = "Entrez"


class ExpressionScale(StrEnum):
    RAW_COUNTS = "raw_counts"
    ESTIMATED_COUNTS = "estimated_counts"
    NORMALIZED_COUNTS = "normalized_counts"
    TPM = "tpm"
    CPM = "cpm"
    FPKM = "fpkm"


class ExpressionKind(StrEnum):
    SOURCE = "source"
    MODEL_READY = "model_ready"


@dataclass(frozen=True, slots=True)
class ExpressionDataset:
    """An immutable descriptor around a gene-by-sample Polars matrix."""

    expression: pl.DataFrame
    gene_id_type: GeneIdType
    scale: ExpressionScale
    metadata: pl.DataFrame | None = None
    gene_column: str = "gene"
    sample_column: str = "sample"
    kind: ExpressionKind = ExpressionKind.SOURCE
    source: Path | None = None

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(
            column for column in self.expression.columns if column != self.gene_column
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", self.expression.clone())
        if self.metadata is not None:
            object.__setattr__(self, "metadata", self.metadata.clone())
        if self.gene_column not in self.expression.columns:
            raise InvalidExpressionError(
                f"Gene column {self.gene_column!r} is missing"
            )
        if self.expression.height == 0:
            raise InvalidExpressionError("Expression matrix cannot be empty")
        samples = self.sample_ids
        if not samples:
            raise InvalidExpressionError(
                "Expression matrix must contain at least one sample column"
            )
        genes = self.expression.get_column(self.gene_column)
        if genes.null_count() > 0:
            raise InvalidExpressionError("Gene identifiers cannot be null")
        gene_values = genes.cast(pl.String, strict=True)
        if gene_values.n_unique() != gene_values.len():
            raise InvalidExpressionError("Gene identifiers must be unique")

        try:
            values = (
                self.expression.select(samples)
                .cast({sample: pl.Float64 for sample in samples}, strict=True)
                .to_numpy()
            )
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise InvalidExpressionError(
                "All expression sample columns must be numeric"
            ) from error
        numeric = np.asarray(values, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise InvalidExpressionError("Expression values must be finite")
        if np.any(numeric < 0.0):
            raise InvalidExpressionError("Expression values cannot be negative")
        if self.scale is ExpressionScale.RAW_COUNTS and not np.allclose(
            numeric,
            np.rint(numeric),
            rtol=0.0,
            atol=1e-9,
        ):
            raise InvalidExpressionError("Raw counts must be integer-valued")

        if self.metadata is not None:
            self._validate_metadata(samples)

    def _validate_metadata(self, samples: tuple[str, ...]) -> None:
        assert self.metadata is not None
        if self.sample_column not in self.metadata.columns:
            raise SampleAlignmentError(
                f"Metadata sample column {self.sample_column!r} is missing"
            )
        metadata_samples = tuple(
            self.metadata.get_column(self.sample_column)
            .cast(pl.String, strict=True)
            .to_list()
        )
        if len(metadata_samples) != len(set(metadata_samples)):
            raise SampleAlignmentError("Metadata sample identifiers must be unique")
        if metadata_samples != samples:
            raise SampleAlignmentError(
                "Metadata sample identifiers and order must exactly match "
                "expression sample columns"
            )


def read_expression(
    source: Path | pl.DataFrame,
    *,
    scale: ExpressionScale,
    gene_id_type: GeneIdType,
    metadata: Path | pl.DataFrame | None = None,
    gene_column: str = "gene",
    sample_column: str = "sample",
) -> ExpressionDataset:
    """Materialize and validate expression input at the I/O boundary."""

    expression_path = source if isinstance(source, Path) else None
    expression = _read_frame(source)
    metadata_frame = None if metadata is None else _read_frame(metadata)
    if metadata_frame is not None and sample_column in metadata_frame.columns:
        samples = [
            column for column in expression.columns if column != gene_column
        ]
        metadata_frame = metadata_frame.with_columns(
            pl.col(sample_column).cast(pl.String, strict=True)
        )
        metadata_samples = metadata_frame.get_column(sample_column).to_list()
        if len(metadata_samples) != len(set(metadata_samples)):
            raise SampleAlignmentError("Metadata sample identifiers must be unique")
        if set(metadata_samples) != set(samples):
            raise SampleAlignmentError(
                "Metadata and expression must contain the same sample identifiers"
            )
        metadata_frame = (
            pl.DataFrame(
                {
                    sample_column: samples,
                    "_sample_order": range(len(samples)),
                }
            )
            .join(
                metadata_frame,
                on=sample_column,
                how="left",
                validate="1:1",
            )
            .sort("_sample_order")
            .drop("_sample_order")
        )
    return ExpressionDataset(
        expression=expression,
        metadata=metadata_frame,
        gene_column=gene_column,
        sample_column=sample_column,
        gene_id_type=gene_id_type,
        scale=scale,
        source=expression_path,
    )


def _read_frame(source: Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source.clone()
    suffixes = source.suffixes
    logical_suffix = suffixes[-2] if suffixes and suffixes[-1] == ".gz" else source.suffix
    if logical_suffix == ".parquet":
        return pl.read_parquet(source)
    if logical_suffix in {".tsv", ".txt"}:
        return pl.read_csv(source, separator="\t")
    if logical_suffix == ".csv":
        return pl.read_csv(source)
    raise InvalidExpressionError(
        f"Unsupported table format for {source}; expected CSV, TSV, TXT, or Parquet"
    )


__all__ = [
    "ExpressionDataset",
    "ExpressionKind",
    "ExpressionScale",
    "GeneIdType",
    "Species",
    "read_expression",
]


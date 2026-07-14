"""Import transcript-level quantifications as gene-level clock input."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import polars as pl

from just_rna.data import ExpressionDataset
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import read_expression
from just_rna.exceptions import InvalidExpressionError


class QuantificationError(InvalidExpressionError):
    """Transcript quantifications cannot be mapped to genes."""


def read_salmon(
    quantifications: Mapping[str, Path],
    transcript_to_gene: Path | pl.DataFrame,
    *,
    gene_id_type: GeneIdType,
    metadata: Path | pl.DataFrame | None = None,
    transcript_column: str = "transcript",
    gene_column: str = "gene",
    strip_versions: bool = True,
) -> ExpressionDataset:
    """Read Salmon ``quant.sf`` files and aggregate ``NumReads`` by gene."""

    return read_transcript_quantifications(
        quantifications,
        transcript_to_gene,
        gene_id_type=gene_id_type,
        metadata=metadata,
        mapping_transcript_column=transcript_column,
        mapping_gene_column=gene_column,
        quantification_transcript_column="Name",
        count_column="NumReads",
        strip_versions=strip_versions,
    )


def read_transcript_quantifications(
    quantifications: Mapping[str, Path],
    transcript_to_gene: Path | pl.DataFrame,
    *,
    gene_id_type: GeneIdType,
    metadata: Path | pl.DataFrame | None = None,
    mapping_transcript_column: str = "transcript",
    mapping_gene_column: str = "gene",
    quantification_transcript_column: str,
    count_column: str,
    strip_versions: bool = True,
) -> ExpressionDataset:
    """Aggregate Salmon/Kallisto-style estimated transcript counts by gene."""

    if not quantifications:
        raise QuantificationError("At least one quantification is required")
    mapping = _read_mapping(transcript_to_gene)
    required_mapping = {mapping_transcript_column, mapping_gene_column}
    missing_mapping = required_mapping.difference(mapping.columns)
    if missing_mapping:
        raise QuantificationError(
            f"Transcript-to-gene mapping is missing columns: {sorted(missing_mapping)}"
        )
    mapping = mapping.select(
        pl.col(mapping_transcript_column).cast(pl.String).alias("_transcript"),
        pl.col(mapping_gene_column).cast(pl.String).alias("gene"),
    ).drop_nulls()
    if strip_versions:
        expressions = [
            pl.col("_transcript").str.replace(r"\.\d+$", ""),
        ]
        if gene_id_type is GeneIdType.ENSEMBL:
            expressions.append(pl.col("gene").str.replace(r"\.\d+$", ""))
        mapping = mapping.with_columns(expressions)
    mapping = mapping.unique()
    ambiguous = (
        mapping.group_by("_transcript")
        .agg(pl.col("gene").n_unique().alias("_genes"))
        .filter(pl.col("_genes") > 1)
    )
    if not ambiguous.is_empty():
        raise QuantificationError(
            f"{ambiguous.height} transcripts map to multiple genes"
        )
    mapping = mapping.unique(subset="_transcript", keep="first")

    samples = tuple(
        _aggregate_sample(
            sample,
            source,
            mapping,
            quantification_transcript_column=quantification_transcript_column,
            count_column=count_column,
            strip_versions=strip_versions,
        )
        for sample, source in quantifications.items()
    )
    expression = samples[0]
    for sample in samples[1:]:
        expression = expression.join(
            sample,
            on="gene",
            how="full",
            coalesce=True,
            validate="1:1",
        )
    expression = expression.fill_null(0.0).sort("gene")
    return read_expression(
        expression,
        scale=ExpressionScale.ESTIMATED_COUNTS,
        gene_id_type=gene_id_type,
        metadata=metadata,
    )


def _aggregate_sample(
    sample: str,
    source: Path,
    mapping: pl.DataFrame,
    *,
    quantification_transcript_column: str,
    count_column: str,
    strip_versions: bool,
) -> pl.DataFrame:
    path = source / "quant.sf" if source.is_dir() else source
    if not path.is_file():
        raise QuantificationError(f"Quantification file does not exist: {path}")
    quantification = pl.read_csv(path, separator="\t")
    required = {quantification_transcript_column, count_column}
    missing = required.difference(quantification.columns)
    if missing:
        raise QuantificationError(
            f"{path} is missing quantification columns: {sorted(missing)}"
        )
    quantification = quantification.select(
        pl.col(quantification_transcript_column)
        .cast(pl.String)
        .alias("_transcript"),
        pl.col(count_column).cast(pl.Float64, strict=True).alias("_count"),
    )
    if strip_versions:
        quantification = quantification.with_columns(
            pl.col("_transcript").str.replace(r"\.\d+$", "")
        )
    mapped = quantification.join(
        mapping,
        on="_transcript",
        how="inner",
        validate="m:1",
    )
    if mapped.is_empty():
        raise QuantificationError(f"No transcripts from {path} mapped to genes")
    return mapped.group_by("gene").agg(pl.col("_count").sum().alias(sample))


def _read_mapping(source: Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source.clone()
    separator = "," if source.suffix == ".csv" else "\t"
    return pl.read_csv(source, separator=separator)


__all__ = [
    "QuantificationError",
    "read_salmon",
    "read_transcript_quantifications",
]


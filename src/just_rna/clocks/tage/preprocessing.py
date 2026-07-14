from __future__ import annotations

from importlib.resources import as_file
from importlib.resources import files
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import rankdata

from just_rna.data import Species

GeneMappingType = Literal["Gene.Symbol", "Ensembl"]
NormalizationMethod = Literal["RLE", "TMM"]
FloatArray = NDArray[np.float64]

def _read_resource_csv(name: str) -> pl.DataFrame:
    resource = files("just_rna.clocks.tage").joinpath("resources", name)
    with as_file(resource) as path:
        return pl.read_csv(
            path,
            infer_schema_length=10000,
            null_values=["NA", ""],
        )


def _sample_columns(expression: pl.DataFrame, gene_column: str) -> list[str]:
    if gene_column not in expression.columns:
        raise ValueError(f"Gene column {gene_column!r} is missing")
    sample_columns = [column for column in expression.columns if column != gene_column]
    if not sample_columns:
        raise ValueError("Expression matrix must contain at least one sample")
    return sample_columns


def _to_numpy(expression: pl.DataFrame, gene_column: str) -> tuple[list[str], FloatArray]:
    sample_columns = _sample_columns(expression, gene_column)
    values = (
        expression.select(sample_columns)
        .cast({column: pl.Float64 for column in sample_columns}, strict=True)
        .to_numpy()
    )
    return sample_columns, np.asarray(values, dtype=np.float64)


def _from_numpy(
    genes: pl.Series,
    sample_columns: list[str],
    values: FloatArray,
    gene_column: str,
) -> pl.DataFrame:
    columns: dict[str, pl.Series | FloatArray] = {gene_column: genes.cast(pl.String)}
    columns.update(
        {
            sample: np.asarray(values[:, index], dtype=np.float64)
            for index, sample in enumerate(sample_columns)
        }
    )
    return pl.DataFrame(columns)


def filter_genes(
    expression: pl.DataFrame,
    *,
    gene_column: str = "gene",
    count_threshold: float = 10.0,
    percent_threshold: float = 20.0,
) -> pl.DataFrame:
    if not 0.0 <= percent_threshold <= 100.0:
        raise ValueError("percent_threshold must be between 0 and 100")
    sample_columns, values = _to_numpy(expression, gene_column)
    required = len(sample_columns) * percent_threshold / 100.0
    keep = np.sum(values >= count_threshold, axis=1) >= required
    return expression.filter(pl.Series("_keep", keep))


def _read_gene_table(species: Species) -> pl.DataFrame:
    return _read_resource_csv(f"Gene_table_{species.value}.csv")


def _mapping_table(
    species: Species,
    gene_mapping_type: GeneMappingType,
) -> pl.DataFrame:
    gene_table = _read_gene_table(species)
    if gene_mapping_type not in gene_table.columns:
        raise ValueError(f"Unknown gene mapping type: {gene_mapping_type}")
    if species is Species.MONKEY:
        monkey_to_ensembl = gene_table.select(
            pl.col(gene_mapping_type).cast(pl.String).alias("_source"),
            pl.col("Ensembl").cast(pl.String).alias("_monkey_ensembl"),
        )
        orthologs = _read_resource_csv("Orthologs_monkey_to_mouse_5.0.csv").select(
            pl.col("Ensembl.macaca").cast(pl.String).alias("_monkey_ensembl"),
            pl.col("Entrez.mouse").cast(pl.String).alias("_mapped"),
        )
        return (
            monkey_to_ensembl.join(
                orthologs.drop_nulls("_mapped"),
                on="_monkey_ensembl",
                how="inner",
                maintain_order="left",
            )
            .select("_source", "_mapped")
            .unique(subset="_source", keep="first", maintain_order=True)
        )
    return (
        gene_table.select(
            pl.col(gene_mapping_type).cast(pl.String).alias("_source"),
            pl.col("Entrez").cast(pl.String).alias("_mapped"),
        )
        .drop_nulls("_mapped")
        .unique(subset="_source", keep="first", maintain_order=True)
    )


def _map_to_mouse_orthologs(mapped: pl.DataFrame, species: Species) -> pl.DataFrame:
    key = f"Entrez.{species.value.capitalize()}"
    orthologs = (
        _read_resource_csv("Table_of_orthologs.csv")
        .select(
            pl.col(key).cast(pl.String).alias("_mapped"),
            pl.col("Entrez.Mouse").cast(pl.String).alias("_mouse"),
        )
        .drop_nulls("_mouse")
        .unique(subset="_mapped", keep="first", maintain_order=True)
    )
    return (
        mapped.join(orthologs, on="_mapped", how="inner", maintain_order="left")
        .unique(subset="_mouse", keep="first", maintain_order=True)
        .drop("_mapped")
        .rename({"_mouse": "_mapped"})
    )


def map_genes(
    expression: pl.DataFrame,
    species: Species | str,
    gene_mapping_type: GeneMappingType,
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    resolved_species = Species(species)
    sample_columns = _sample_columns(expression, gene_column)
    numeric = expression.with_columns(
        pl.col(column).cast(pl.Float64, strict=True) for column in sample_columns
    )
    mapped = (
        numeric.with_columns(pl.col(gene_column).cast(pl.String).alias("_source"))
        .join(
            _mapping_table(resolved_species, gene_mapping_type),
            on="_source",
            how="inner",
            maintain_order="left",
        )
        .group_by("_mapped", maintain_order=True)
        .agg(pl.col(column).sum() for column in sample_columns)
        .sort(pl.col("_mapped").cast(pl.Int64))
    )
    if resolved_species not in {Species.MOUSE, Species.MONKEY}:
        mapped = _map_to_mouse_orthologs(mapped, resolved_species)
    return mapped.rename({"_mapped": gene_column}).select(gene_column, *sample_columns)


def _validate_counts(values: FloatArray) -> FloatArray:
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalization requires finite counts")
    if np.any(values < 0.0):
        raise ValueError("Normalization requires non-negative counts")
    library_sizes = np.sum(values, axis=0, dtype=np.float64)
    if np.any(library_sizes <= 0.0):
        raise ValueError("Normalization requires positive library sizes")
    return library_sizes


def rle_factors(values: FloatArray) -> FloatArray:
    library_sizes = _validate_counts(values)
    positive_rows = np.all(values > 0.0, axis=1)
    if not np.any(positive_rows):
        raise ValueError("RLE requires at least one gene positive in every sample")
    geometric_means = np.exp(np.mean(np.log(values[positive_rows]), axis=1))
    median_ratios = np.median(
        values[positive_rows] / geometric_means[:, np.newaxis],
        axis=0,
    )
    factors = median_ratios / library_sizes
    if np.any(factors <= 0.0):
        raise ValueError("RLE produced a non-positive normalization factor")
    return factors / np.exp(np.mean(np.log(factors)))


def _tmm_pair_factor(
    observed: FloatArray,
    reference: FloatArray,
    observed_size: float,
    reference_size: float,
    *,
    logratio_trim: float,
    sum_trim: float,
) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(
            (observed / observed_size) / (reference / reference_size)
        )
        absolute_expression = (
            np.log2(observed / observed_size)
            + np.log2(reference / reference_size)
        ) / 2.0
        variance = (
            (observed_size - observed) / observed_size / observed
            + (reference_size - reference) / reference_size / reference
        )
    finite = (
        np.isfinite(log_ratio)
        & np.isfinite(absolute_expression)
        & (absolute_expression > -1.0e10)
    )
    if not np.any(finite):
        return 1.0
    log_ratio = log_ratio[finite]
    absolute_expression = absolute_expression[finite]
    variance = variance[finite]
    if np.max(np.abs(log_ratio)) < 1.0e-6:
        return 1.0
    count = log_ratio.size
    lower_log = np.floor(count * logratio_trim) + 1.0
    upper_log = count + 1.0 - lower_log
    lower_sum = np.floor(count * sum_trim) + 1.0
    upper_sum = count + 1.0 - lower_sum
    keep = (
        (rankdata(log_ratio, method="average") >= lower_log)
        & (rankdata(log_ratio, method="average") <= upper_log)
        & (rankdata(absolute_expression, method="average") >= lower_sum)
        & (rankdata(absolute_expression, method="average") <= upper_sum)
    )
    if not np.any(keep):
        return 1.0
    weighted_mean = np.nansum(log_ratio[keep] / variance[keep]) / np.nansum(
        1.0 / variance[keep]
    )
    if np.isnan(weighted_mean):
        weighted_mean = 0.0
    return float(2.0**weighted_mean)


def tmm_factors(
    values: FloatArray,
    *,
    reference_column: int | None = None,
    logratio_trim: float = 0.30,
    sum_trim: float = 0.05,
) -> FloatArray:
    library_sizes = _validate_counts(values)
    if reference_column is None:
        upper_quartiles = np.array(
            [
                np.quantile(column, 0.75, method="linear")
                for column in values.T
            ],
            dtype=np.float64,
        ) / library_sizes
        if np.median(upper_quartiles) < 1.0e-20:
            root_sums = np.sum(np.sqrt(values), axis=0)
            reference_column = int(np.argmax(root_sums))
        else:
            reference_column = int(
                np.argmin(np.abs(upper_quartiles - np.mean(upper_quartiles)))
            )
    if not 0 <= reference_column < values.shape[1]:
        raise IndexError("reference_column is outside the expression matrix")
    factors = np.array(
        [
            _tmm_pair_factor(
                values[:, index],
                values[:, reference_column],
                float(library_sizes[index]),
                float(library_sizes[reference_column]),
                logratio_trim=logratio_trim,
                sum_trim=sum_trim,
            )
            for index in range(values.shape[1])
        ],
        dtype=np.float64,
    )
    return factors / np.exp(np.mean(np.log(factors)))


def normalize_counts(
    expression: pl.DataFrame,
    *,
    method: NormalizationMethod = "RLE",
    gene_column: str = "gene",
    scale: float = 1.0e7,
) -> pl.DataFrame:
    sample_columns, values = _to_numpy(expression, gene_column)
    library_sizes = _validate_counts(values)
    if method == "RLE":
        factors = rle_factors(values)
    elif method == "TMM":
        factors = tmm_factors(values)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    normalized = values / (library_sizes * factors)[np.newaxis, :] * scale
    return _from_numpy(
        expression.get_column(gene_column),
        sample_columns,
        normalized,
        gene_column,
    )


def log10_transform(
    expression: pl.DataFrame,
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    sample_columns, values = _to_numpy(expression, gene_column)
    transformed = np.log10(values + 1.0)
    return _from_numpy(
        expression.get_column(gene_column),
        sample_columns,
        transformed,
        gene_column,
    )


def scale_columns(
    expression: pl.DataFrame,
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    sample_columns, values = _to_numpy(expression, gene_column)
    scaled = (values - np.mean(values, axis=0)) / np.std(values, axis=0, ddof=1)
    return _from_numpy(
        expression.get_column(gene_column),
        sample_columns,
        scaled,
        gene_column,
    )


def yugene(
    expression: pl.DataFrame,
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    sample_columns, values = _to_numpy(expression, gene_column)
    shifted = values - np.nanmin(values, axis=0)
    result = np.empty_like(shifted)
    for column_index in range(shifted.shape[1]):
        column = shifted[:, column_index]
        order = np.argsort(-column, kind="stable")
        sorted_values = column[order]
        total = np.sum(sorted_values)
        if total == 0.0:
            result[:, column_index] = 1.0
            continue
        cumulative = np.cumsum(sorted_values) / total
        tied = sorted_values[1:] == sorted_values[:-1]
        for index in np.flatnonzero(tied) + 1:
            cumulative[index] = cumulative[index - 1]
        result[order, column_index] = 1.0 - cumulative
    return _from_numpy(
        expression.get_column(gene_column),
        sample_columns,
        result,
        gene_column,
    )


def load_gene_list() -> tuple[str, ...]:
    resource = files("just_rna.clocks.tage").joinpath(
        "resources",
        "Gene_list_all_4.6.txt",
    )
    return tuple(
        line.strip()
        for line in resource.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def align_to_gene_list(
    expression: pl.DataFrame,
    gene_list: tuple[str, ...] | list[str] | None = None,
    *,
    gene_column: str = "gene",
) -> pl.DataFrame:
    target = load_gene_list() if gene_list is None else tuple(map(str, gene_list))
    requested = pl.DataFrame(
        {
            gene_column: target,
            "_order": np.arange(len(target), dtype=np.int64),
        }
    )
    aligned = (
        requested.join(
            expression.with_columns(pl.col(gene_column).cast(pl.String)),
            on=gene_column,
            how="left",
            maintain_order="left",
        )
        .sort("_order")
        .drop("_order")
    )
    sample_columns = [column for column in aligned.columns if column != gene_column]
    return aligned.with_columns(
        pl.col(column).cast(pl.Float64).fill_null(float("nan"))
        for column in sample_columns
    )


def control_median_subtract(
    expression: pl.DataFrame,
    metadata: pl.DataFrame | None = None,
    *,
    control_group_column: str | None = None,
    control_group_label: str | None = None,
    gene_column: str = "gene",
) -> pl.DataFrame:
    if control_group_column is None or control_group_label is None:
        return expression.clone()
    if metadata is None:
        raise ValueError("metadata is required for control subtraction")
    if control_group_column not in metadata.columns:
        raise ValueError(f"Metadata column {control_group_column!r} is missing")
    sample_columns, values = _to_numpy(expression, gene_column)
    if metadata.height != len(sample_columns):
        raise ValueError("Metadata rows must match expression sample columns")
    labels = metadata.get_column(control_group_column).cast(pl.String).to_list()
    control_indices = [
        index for index, label in enumerate(labels) if label == control_group_label
    ]
    reference = values[:, control_indices] if control_indices else values
    medians = np.full(values.shape[0], np.nan, dtype=np.float64)
    rows_with_values = np.any(~np.isnan(reference), axis=1)
    medians[rows_with_values] = np.nanmedian(
        reference[rows_with_values],
        axis=1,
    )
    adjusted = values - medians[:, np.newaxis]
    return _from_numpy(
        expression.get_column(gene_column),
        sample_columns,
        adjusted,
        gene_column,
    )



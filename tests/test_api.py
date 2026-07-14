from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from just_rna.data import ExpressionDataset
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import read_expression
from just_rna.exceptions import InvalidExpressionError
from just_rna.exceptions import SampleAlignmentError


def test_expression_dataset_validates_counts_and_sample_alignment() -> None:
    expression = pl.DataFrame({"gene": ["Xkr4"], "sample_a": [100]})
    metadata = pl.DataFrame({"sample": ["different"]})

    with pytest.raises(SampleAlignmentError):
        ExpressionDataset(
            expression=expression,
            metadata=metadata,
            gene_id_type=GeneIdType.SYMBOL,
            scale=ExpressionScale.RAW_COUNTS,
        )
    with pytest.raises(InvalidExpressionError):
        ExpressionDataset(
            expression=pl.DataFrame({"gene": ["Xkr4"], "sample_a": [1.5]}),
            gene_id_type=GeneIdType.SYMBOL,
            scale=ExpressionScale.RAW_COUNTS,
        )


def test_read_expression_uses_pathlib_and_reorders_metadata(tmp_path: Path) -> None:
    expression_path = tmp_path / "expression.tsv"
    metadata_path = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            "gene": ["Xkr4"],
            "sample_a": [100],
            "sample_b": [120],
        }
    ).write_csv(expression_path, separator="\t")
    pl.DataFrame(
        {
            "sample": ["sample_b", "sample_a"],
            "batch": [2, 1],
        }
    ).write_csv(metadata_path)

    dataset = read_expression(
        expression_path,
        scale=ExpressionScale.RAW_COUNTS,
        gene_id_type=GeneIdType.SYMBOL,
        metadata=metadata_path,
    )

    assert dataset.source == expression_path
    assert dataset.sample_ids == ("sample_a", "sample_b")
    assert dataset.metadata is not None
    assert dataset.metadata.get_column("batch").to_list() == [1, 2]


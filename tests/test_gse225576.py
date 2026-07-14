from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from just_rna.datasets.gse225576 import MouseTissue
from just_rna.datasets.gse225576 import normalize_gse225576_metadata
from just_rna.datasets.gse225576 import parse_gse225576_expression


def test_curated_parser_uses_geo_age_metadata(tmp_path: Path) -> None:
    matrix_path = tmp_path / "GSE225576_Brain.txt.gz"
    with gzip.open(matrix_path, "wt", encoding="utf-8") as stream:
        stream.write("Gene_ID\t6M1\t30M4\n")
        stream.write("Xkr4\t10.5\t20.5\n")

    expression = parse_gse225576_expression(matrix_path, MouseTissue.BRAIN)

    assert expression.columns == ["gene", "brain_6mo_1", "brain_30mo_4"]
    rows = [
        {
            "sample_name": f"{tissue.value.lower()}_{age}mo_{replicate}",
            "sample_geo_accession": f"GSM{index:07d}",
            "tissue": tissue.value,
            "age": f"{age}mo",
        }
        for index, (tissue, age, replicate) in enumerate(
            (
                (tissue, age, replicate)
                for tissue in MouseTissue
                for age in (6, 15, 24, 30)
                for replicate in range(1, 5)
            ),
            start=1,
        )
    ]
    metadata = normalize_gse225576_metadata(pl.DataFrame(rows))

    assert metadata.height == 128
    assert set(metadata.get_column("age_months").to_list()) == {6, 15, 24, 30}
    assert metadata.filter(pl.col("sample") == "brain_30mo_4").row(
        0,
        named=True,
    )["age_months"] == 30


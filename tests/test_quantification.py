from __future__ import annotations

from pathlib import Path

import polars as pl

from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.quantification import read_salmon


def test_salmon_estimated_counts_are_aggregated_by_gene(tmp_path: Path) -> None:
    first = tmp_path / "mouse_6m_1"
    second = tmp_path / "mouse_24m_1"
    first.mkdir()
    second.mkdir()
    _write_quant(
        first / "quant.sf",
        transcripts=["ENSMUST1.1", "ENSMUST2.2", "ENSMUST3.1"],
        counts=[1.5, 2.25, 4.0],
    )
    _write_quant(
        second / "quant.sf",
        transcripts=["ENSMUST1.1", "ENSMUST2.2", "ENSMUST3.1"],
        counts=[3.0, 1.0, 8.5],
    )
    mapping = pl.DataFrame(
        {
            "transcript": ["ENSMUST1", "ENSMUST2", "ENSMUST3"],
            "gene": ["ENSMUSG1.4", "ENSMUSG1.4", "ENSMUSG2.7"],
        }
    )

    dataset = read_salmon(
        {
            "mouse_6m_1": first,
            "mouse_24m_1": second,
        },
        mapping,
        gene_id_type=GeneIdType.ENSEMBL,
    )

    assert dataset.scale is ExpressionScale.ESTIMATED_COUNTS
    assert dataset.expression.to_dict(as_series=False) == {
        "gene": ["ENSMUSG1", "ENSMUSG2"],
        "mouse_6m_1": [3.75, 4.0],
        "mouse_24m_1": [4.0, 8.5],
    }


def _write_quant(
    path: Path,
    *,
    transcripts: list[str],
    counts: list[float],
) -> None:
    pl.DataFrame(
        {
            "Name": transcripts,
            "Length": [1000] * len(transcripts),
            "EffectiveLength": [800.0] * len(transcripts),
            "TPM": [1.0] * len(transcripts),
            "NumReads": counts,
        }
    ).write_csv(path, separator="\t")


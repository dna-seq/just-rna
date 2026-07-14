from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from just_rna.clocks.tage import align_model_features
from just_rna.clocks.tage import align_to_gene_list
from just_rna.clocks.tage import control_median_subtract
from just_rna.clocks.tage import filter_genes
from just_rna.clocks.tage import map_genes
from just_rna.clocks.tage import normalize_counts
from just_rna.clocks.tage import predict_tage
from just_rna.clocks.tage import rle_factors
from just_rna.clocks.tage import scale_columns
from just_rna.clocks.tage import tmm_factors
from just_rna.clocks.tage import yugene


class SmallClock:
    feature_names = ("20", "10")

    def predict(
        self,
        values: pl.DataFrame,
        *,
        return_std: bool = False,
    ) -> np.ndarray[Any, np.dtype[np.float64]] | tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        prediction = (
            values.get_column("20").to_numpy()
            + 2.0 * values.get_column("10").to_numpy()
        )
        if return_std:
            return prediction, np.full(values.height, 0.25)
        return prediction


def test_filter_genes_uses_at_least_threshold_and_percentage() -> None:
    expression = pl.DataFrame(
        {
            "gene": ["keep", "half", "drop", "nan"],
            "a": [10.0, 10.0, 9.0, float("nan")],
            "b": [10.0, 0.0, 9.0, 10.0],
        }
    )

    result = filter_genes(
        expression,
        count_threshold=10.0,
        percent_threshold=50.0,
    )

    assert result.get_column("gene").to_list() == ["keep", "half", "nan"]


def test_rle_and_tmm_exact_small_vectors() -> None:
    counts = np.array(
        [
            [100.0, 100.0],
            [100.0, 200.0],
            [100.0, 700.0],
        ]
    )
    np.testing.assert_allclose(
        rle_factors(counts),
        np.array([1.2909944487358056, 0.7745966692414834]),
        rtol=1e-14,
    )

    proportional = np.array(
        [
            [10.0, 20.0, 40.0],
            [30.0, 60.0, 120.0],
            [60.0, 120.0, 240.0],
        ]
    )
    np.testing.assert_allclose(tmm_factors(proportional), np.ones(3), atol=1e-15)
    nontrivial = np.array(
        [
            [10.0, 20.0, 5.0],
            [20.0, 10.0, 50.0],
            [30.0, 60.0, 20.0],
            [40.0, 20.0, 80.0],
            [50.0, 100.0, 25.0],
            [60.0, 30.0, 120.0],
        ]
    )
    np.testing.assert_allclose(
        tmm_factors(nontrivial, reference_column=0),
        np.array(
            [1.0365361661213797, 0.7928418866003347, 1.216827325760962]
        ),
        rtol=1e-14,
    )

    normalized = normalize_counts(
        pl.DataFrame(
            {
                "gene": ["a", "b", "c"],
                "s1": proportional[:, 0],
                "s2": proportional[:, 1],
                "s3": proportional[:, 2],
            }
        ),
        method="RLE",
    )
    expected = np.array([1.0e6, 3.0e6, 6.0e6])
    for sample in ("s1", "s2", "s3"):
        np.testing.assert_allclose(normalized.get_column(sample).to_numpy(), expected)


def test_scale_and_yugene_match_r_small_vectors() -> None:
    scaled = scale_columns(
        pl.DataFrame(
            {
                "gene": ["a", "b", "c"],
                "sample": [1.0, 2.0, 3.0],
            }
        )
    )
    np.testing.assert_allclose(
        scaled.get_column("sample").to_numpy(),
        np.array([-1.0, 0.0, 1.0]),
        atol=0.0,
    )

    transformed = yugene(
        pl.DataFrame(
            {
                "gene": ["a", "b", "c", "d"],
                "tied": [3.0, 3.0, 1.0, 0.0],
                "zero": [5.0, 5.0, 5.0, 5.0],
            }
        )
    )
    np.testing.assert_allclose(
        transformed.get_column("tied").to_numpy(),
        np.array([4.0 / 7.0, 4.0 / 7.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        transformed.get_column("zero").to_numpy(),
        np.ones(4),
    )


def test_alignment_uses_nan_and_control_subtraction_uses_control_median() -> None:
    expression = pl.DataFrame(
        {
            "gene": ["20", "10"],
            "control_a": [2.0, 1.0],
            "treated": [8.0, 7.0],
            "control_b": [4.0, 5.0],
        }
    )
    aligned = align_to_gene_list(expression, ["10", "missing", "20"])
    assert aligned.get_column("gene").to_list() == ["10", "missing", "20"]
    assert np.isnan(aligned.get_column("control_a")[1])

    adjusted = control_median_subtract(
        aligned,
        pl.DataFrame({"group": ["control", "treated", "control"]}),
        control_group_column="group",
        control_group_label="control",
    )
    np.testing.assert_allclose(
        adjusted.filter(pl.col("gene") == "10").select(
            "control_a", "treated", "control_b"
        ).to_numpy(),
        np.array([[-2.0, 4.0, 2.0]]),
    )
    assert np.isnan(adjusted.filter(pl.col("gene") == "missing")["treated"][0])


def test_mouse_mapping_sums_duplicate_targets() -> None:
    expression = pl.DataFrame(
        {
            "gene": ["Xkr4", "Xkr4", "Sox17", "not-a-gene"],
            "sample": [1.0, 2.0, 4.0, 8.0],
        }
    )

    mapped = map_genes(expression, "mouse", "Gene.Symbol")

    assert dict(mapped.iter_rows()) == {"497097": 3.0, "20671": 4.0}


def test_human_mapping_converts_entrez_to_mouse_orthologs() -> None:
    expression = pl.DataFrame(
        {
            "gene": ["A1BG", "A2M"],
            "sample": [2.0, 3.0],
        }
    )

    mapped = map_genes(expression, "human", "Gene.Symbol")

    assert dict(mapped.iter_rows()) == {"117586": 2.0, "232345": 3.0}


def test_prediction_aligns_features_and_applies_species_multiplier(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "small-clock.joblib"
    joblib.dump(SmallClock(), model_path)
    expression = pl.DataFrame(
        {
            "gene": ["10", "unused", "20"],
            "sample_a": [1.0, 100.0, 3.0],
            "sample_b": [2.0, 200.0, 4.0],
        }
    )

    aligned = align_model_features(expression, ["20", "10"])
    assert aligned.columns == ["20", "10"]
    np.testing.assert_array_equal(aligned.to_numpy(), np.array([[3.0, 1.0], [4.0, 2.0]]))

    result = predict_tage(
        model_path,
        expression,
        species="mouse",
        metadata=pl.DataFrame({"batch": ["x", "y"]}),
        return_std=True,
        prefix="EN_",
    )

    assert result.columns == ["batch", "EN_tAge", "EN_tAge_std"]
    np.testing.assert_array_equal(
        result.get_column("EN_tAge").to_numpy(),
        np.array([240.0, 384.0]),
    )
    np.testing.assert_array_equal(
        result.get_column("EN_tAge_std").to_numpy(),
        np.array([12.0, 12.0]),
    )

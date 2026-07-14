from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from just_rna.datasets.gse225576 import download_gse225576
from just_rna.models import ClockResult
from just_rna.workflows import predict_gse225576_tage

REFERENCE_PREDICTIONS = {
    "aorta_6mo_1": 0.837505549663179,
    "brain_15mo_2": 5.899412860413333,
    "heart_24mo_3": 17.15511309161728,
    "kidney_30mo_4": 17.975665592749788,
    "liver_6mo_4": 1.2695259775943413,
    "lung_15mo_3": 7.490942594312914,
    "muscle_24mo_2": 18.19322007457984,
    "skin_30mo_1": 7.627810215304814,
}


@pytest.mark.integration
def test_real_gse225576_predictions_match_reference_and_age_signal(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "gse225576"
    output = tmp_path / "mouse_aging_atlas_predictions.csv"
    cache_directory = tmp_path / "assets"

    run = predict_gse225576_tage(
        dataset_directory,
        output,
        cache_directory=cache_directory,
    )
    saved = pl.read_csv(run.output)
    in_memory = pl.concat(
        (result.to_polars() for result in run.results),
        how="diagonal_relaxed",
    ).with_columns(
        (pl.col("age_months") - 6).cast(pl.Float64).alias("ground_truth"),
        pl.lit("months_since_6_month_control").alias("ground_truth_unit"),
    )
    saved_metrics = pl.read_csv(run.metrics_output)

    assert len(run.results) == 8
    assert saved.height == 128
    assert saved.get_column("sample").n_unique() == 128
    assert_frame_equal(
        saved.sort("sample"),
        in_memory.sort("sample"),
        check_column_order=False,
        check_dtypes=False,
    )
    assert set(saved.get_column("age_months").to_list()) == {6, 15, 24, 30}
    assert (
        saved.get_column("ground_truth")
        .eq(saved.get_column("age_months") - 6)
        .all()
    )
    assert (
        saved.filter(pl.col("age_source") == "geo_characteristic").height == 112
    )
    assert saved.filter(pl.col("age_source") == "geo_sample_name").height == 16

    reference = saved.filter(
        pl.col("sample").is_in(REFERENCE_PREDICTIONS)
    ).sort("sample")
    expected_samples = sorted(REFERENCE_PREDICTIONS)
    assert reference.get_column("sample").to_list() == expected_samples
    np.testing.assert_allclose(
        reference.get_column("prediction").to_numpy(),
        np.array([REFERENCE_PREDICTIONS[sample] for sample in expected_samples]),
        rtol=1e-10,
        atol=1e-10,
    )

    age_means = (
        saved.group_by("age_months")
        .agg(pl.col("prediction").mean())
        .sort("age_months")
        .get_column("prediction")
        .to_numpy()
    )
    assert abs(age_means[0]) < 1.0
    assert np.all(np.diff(age_means) > 0.0)
    tissue_correlations = saved.group_by("tissue").agg(
        pl.corr("age_months", "prediction").alias("correlation")
    )
    assert tissue_correlations.get_column("correlation").min() > 0.8
    assert saved_metrics.height == 9
    assert run.metrics.overall.samples == 128
    assert np.isclose(run.metrics.overall.mae, 4.510186822397359)
    assert np.isclose(run.metrics.overall.rmse, 6.123404091437703)
    assert np.isclose(run.metrics.overall.r_squared, 0.5465326963926109)
    assert np.isclose(run.metrics.overall.pearson, 0.8754033045506261)

    assert all(
        "pre-normalized-counts" in {warning.code for warning in result.warnings}
        for result in run.results
    )
    assert all(
        ClockResult.model_validate_json(result.model_dump_json()) == result
        for result in run.results
    )

    first_download = download_gse225576(dataset_directory)
    mtimes = {path: path.stat().st_mtime_ns for path in first_download.files}
    second_download = download_gse225576(dataset_directory)
    assert second_download.files == first_download.files
    assert {path: path.stat().st_mtime_ns for path in second_download.files} == mtimes


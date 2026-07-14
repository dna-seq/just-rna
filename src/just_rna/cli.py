"""Typer command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from just_rna.assets import download_asset
from just_rna.assets import list_assets
from just_rna.clocks.tage.clock import TAgeConfig
from just_rna.config import load_settings
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.data import Species
from just_rna.data import read_expression
from just_rna.datasets.geo import download_geo
from just_rna.datasets.gse225576 import download_gse225576
from just_rna.prediction import predict
from just_rna.workflows import predict_gse225576_tage

DEFAULT_MOUSE_AGING_ATLAS_INPUT = Path("data/input/mouse-aging-atlas")
DEFAULT_MOUSE_AGING_ATLAS_OUTPUT = Path(
    "data/output/mouse_aging_atlas_tage_predictions.parquet"
)
DEFAULT_MOUSE_AGING_ATLAS_CSV = Path(
    "data/output/mouse_aging_atlas_tage_predictions.csv"
)

app = typer.Typer(
    name="just-rna",
    help="Typed transcriptomic aging-clock predictions.",
    no_args_is_help=True,
)
assets_app = typer.Typer(help="Inspect and download clock assets.")
datasets_app = typer.Typer(help="Download public expression datasets.")
app.add_typer(assets_app, name="assets")
app.add_typer(datasets_app, name="datasets")


@app.command("predict")
def predict_command(
    expression: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    species: Annotated[Species, typer.Option(case_sensitive=False)],
    gene_id: Annotated[
        GeneIdType,
        typer.Option("--gene-id", case_sensitive=True),
    ] = GeneIdType.SYMBOL,
    scale: Annotated[
        ExpressionScale,
        typer.Option(case_sensitive=False),
    ] = ExpressionScale.RAW_COUNTS,
    metadata: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    gene_column: Annotated[str, typer.Option()] = "gene",
    sample_column: Annotated[str, typer.Option()] = "sample",
    control_group_column: Annotated[str | None, typer.Option()] = None,
    control_group_label: Annotated[str | None, typer.Option()] = None,
    return_uncertainty: Annotated[bool, typer.Option()] = False,
    dotenv_path: Annotated[
        Path | None,
        typer.Option("--dotenv", exists=True, dir_okay=False),
    ] = None,
) -> None:
    settings = load_settings(dotenv_path)
    dataset = read_expression(
        expression,
        scale=scale,
        gene_id_type=gene_id,
        metadata=metadata,
        gene_column=gene_column,
        sample_column=sample_column,
    )
    config = TAgeConfig(
        species=species,
        return_uncertainty=return_uncertainty,
        control_group_column=control_group_column,
        control_group_label=control_group_label,
    )
    result = predict(
        dataset,
        species=species,
        config=config,
        cache_directory=settings.cache_directory,
    )
    if output.suffix == ".parquet":
        result.write_parquet(output)
    elif output.suffix == ".csv":
        result.write_csv(output)
    else:
        raise typer.BadParameter("Output must end in .csv or .parquet")
    typer.echo(output)


@assets_app.command("list")
def assets_list(
    provider: Annotated[str | None, typer.Option()] = None,
    include_dynamic: Annotated[bool, typer.Option()] = False,
) -> None:
    for asset in list_assets(provider, include_dynamic=include_dynamic):
        typer.echo(f"{asset.id}\t{asset.kind.value}\t{asset.size}\t{asset.filename}")


@assets_app.command("download")
def assets_download(
    asset_id: Annotated[str, typer.Argument()],
    allow_large: Annotated[bool, typer.Option()] = False,
    dotenv_path: Annotated[
        Path | None,
        typer.Option("--dotenv", exists=True, dir_okay=False),
    ] = None,
) -> None:
    settings = load_settings(dotenv_path)
    path = download_asset(
        asset_id,
        cache_dir=settings.cache_directory,
        allow_large=allow_large,
    )
    typer.echo(path)


@datasets_app.command("geo")
def datasets_geo(
    accession: Annotated[str, typer.Argument()],
    destination: Annotated[Path, typer.Option("--destination", "-d")],
    metadata_only: Annotated[bool, typer.Option()] = False,
) -> None:
    project = download_geo(
        accession,
        destination,
        metadata_only=metadata_only,
    )
    typer.echo(
        f"{project.accession}: {project.metadata.height} metadata rows, "
        f"{len(project.files)} files"
    )


@datasets_app.command("atlas")
def datasets_mouse_aging_atlas(
    destination: Annotated[Path, typer.Option("--destination", "-d")] = (
        DEFAULT_MOUSE_AGING_ATLAS_INPUT
    ),
) -> None:
    downloaded = download_gse225576(destination)
    typer.echo(
        f"{len(downloaded.files)} matrices; metadata: {downloaded.metadata_path}"
    )


@app.command("atlas")
def predict_mouse_aging_atlas_command(
    input_directory: Annotated[
        Path,
        typer.Option("--input-directory", "-i"),
    ] = DEFAULT_MOUSE_AGING_ATLAS_INPUT,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    csv: Annotated[
        bool,
        typer.Option("--csv", help="Write predictions as CSV instead of Parquet."),
    ] = False,
    dotenv_path: Annotated[
        Path | None,
        typer.Option("--dotenv", exists=True, dir_okay=False),
    ] = None,
) -> None:
    resolved_output = _atlas_output(output, csv=csv)
    run = predict_gse225576_tage(
        input_directory,
        resolved_output,
        dotenv_path=dotenv_path,
    )
    typer.echo(f"predictions: {run.output}")
    typer.echo(f"metrics: {run.metrics_output}")


def _atlas_output(output: Path | None, *, csv: bool) -> Path:
    if output is None:
        return (
            DEFAULT_MOUSE_AGING_ATLAS_CSV
            if csv
            else DEFAULT_MOUSE_AGING_ATLAS_OUTPUT
        )
    return output.with_suffix(".csv") if csv else output


def main() -> None:
    app()


def age_main() -> None:
    """Standalone ``age`` executable."""

    typer.run(predict_mouse_aging_atlas_command)


def predict_main() -> None:
    """Standalone ``predict`` executable."""

    typer.run(predict_command)


def geo_main() -> None:
    """Standalone ``geo`` executable."""

    typer.run(datasets_geo)


def models_main() -> None:
    """Standalone ``models`` asset manager."""

    assets_app()


__all__ = [
    "app",
    "age_main",
    "geo_main",
    "main",
    "models_main",
    "predict_main",
]


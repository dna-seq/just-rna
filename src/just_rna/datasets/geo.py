"""Typed GEOfetch boundary for GEO and SRA projects."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from typing import cast

import polars as pl

from just_rna.exceptions import JustRnaError

_ACCESSION_RE = re.compile(r"^(?:GSE|SRP)\d+$", re.IGNORECASE)


class GeoDataSource(StrEnum):
    SAMPLES = "samples"
    SERIES = "series"
    ALL = "all"


class GeoDownloadError(JustRnaError):
    """GEOfetch did not return a usable project."""


@dataclass(frozen=True, slots=True)
class GeoProject:
    accession: str
    metadata: pl.DataFrame
    files: tuple[Path, ...]
    destination: Path


class _PandasLike(Protocol):
    def to_dict(self, orient: str) -> dict[str, list[object]]: ...


class _PeppyProject(Protocol):
    sample_table: _PandasLike


def download_geo(
    accession: str,
    destination: Path,
    *,
    processed: bool = True,
    metadata_only: bool = False,
    data_source: GeoDataSource = GeoDataSource.ALL,
    filename_filter: str | None = None,
    size_filter: str | None = None,
) -> GeoProject:
    """Fetch one project through GEOfetch.

    Importing GEOfetch and its pandas stack is delayed until this effectful
    function is called.
    """

    normalized = accession.upper()
    if _ACCESSION_RE.fullmatch(normalized) is None:
        raise GeoDownloadError(
            f"Expected a GSE or SRP accession, received {accession!r}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    from geofetch import Geofetcher  # pyright: ignore[reportMissingTypeStubs]

    raw_projects: Mapping[str, _PeppyProject] | None = None
    final_error: Exception | None = None
    for _attempt in range(2):
        fetcher = Geofetcher(
            name=normalized,
            metadata_root=str(destination),
            metadata_folder=str(destination),
            processed=processed,
            data_source=data_source.value,
            filter=filename_filter,
            filter_size=size_filter,
            geo_folder=str(destination),
            just_metadata=metadata_only,
            discard_soft=True,
            disable_progressbar=True,
        )
        try:
            raw_projects = cast(
                Mapping[str, _PeppyProject],
                fetcher.get_projects(  # pyright: ignore[reportUnknownMemberType]
                    normalized,
                    just_metadata=metadata_only,
                    discard_soft=True,
                ),
            )
            if raw_projects:
                break
            final_error = GeoDownloadError(
                f"GEOfetch returned no projects for {normalized}"
            )
            raw_projects = None
        except Exception as error:
            final_error = error
    if raw_projects is None:
        raise GeoDownloadError(
            f"GEOfetch failed twice for {normalized}"
        ) from final_error
    metadata_frames = [
        _project_metadata(project_name, project)
        for project_name, project in raw_projects.items()
    ]
    if not metadata_frames:
        raise GeoDownloadError(f"GEOfetch returned empty projects for {normalized}")
    metadata = (
        pl.concat(metadata_frames, how="diagonal_relaxed")
        if len(metadata_frames) > 1
        else metadata_frames[0]
    )
    files = tuple(
        sorted(
            path
            for path in destination.rglob("*")
            if path.is_file()
        )
    )
    return GeoProject(
        accession=normalized,
        metadata=metadata,
        files=files,
        destination=destination,
    )


def _project_metadata(
    project_name: str,
    project: _PeppyProject,
) -> pl.DataFrame:
    columns = project.sample_table.to_dict(orient="list")
    return pl.DataFrame(columns, strict=False).with_columns(
        pl.lit(project_name).alias("_geo_project")
    )


__all__ = [
    "GeoDataSource",
    "GeoDownloadError",
    "GeoProject",
    "download_geo",
]


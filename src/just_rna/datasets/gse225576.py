"""Curated GSE225576 mouse aging atlas adapter."""

from __future__ import annotations

import hashlib
import gzip
import re
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx
import polars as pl

from just_rna.data import ExpressionDataset
from just_rna.data import ExpressionScale
from just_rna.data import GeneIdType
from just_rna.datasets.geo import download_geo
from just_rna.exceptions import JustRnaError

GSE225576_ACCESSION = "GSE225576"
_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE225nnn/GSE225576/suppl"
)
_COLUMN_RE = re.compile(r"^(?P<age>\d+)M(?P<replicate>\d+)$")
_AGE_RE = re.compile(r"^(?P<age>\d+)mo$")


class MouseTissue(StrEnum):
    AORTA = "Aorta"
    BRAIN = "Brain"
    HEART = "Heart"
    KIDNEY = "Kidney"
    LIVER = "Liver"
    LUNG = "Lung"
    MUSCLE = "Muscle"
    SKIN = "Skin"


@dataclass(frozen=True, slots=True)
class GseFile:
    tissue: MouseTissue
    filename: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"{_BASE_URL}/{self.filename}"


@dataclass(frozen=True, slots=True)
class Gse225576Download:
    files: tuple[Path, ...]
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class Gse225576Cohort:
    tissue: MouseTissue
    dataset: ExpressionDataset


class Gse225576Error(JustRnaError):
    """The curated dataset did not match its pinned contract."""


GSE225576_FILES: tuple[GseFile, ...] = (
    GseFile(MouseTissue.AORTA, "GSE225576_Aorta.txt.gz", 1_753_610, "63470b990138ecb4304b313918241599"),
    GseFile(MouseTissue.BRAIN, "GSE225576_Brain.txt.gz", 1_840_744, "15145767e27cedc93de929cad1ead0ae"),
    GseFile(MouseTissue.HEART, "GSE225576_Heart.txt.gz", 1_570_265, "a139c2a0c882f2b84f526248b4671b0b"),
    GseFile(MouseTissue.KIDNEY, "GSE225576_Kidney.txt.gz", 1_738_356, "488fc1cd631ff91bb7a5de57b4629395"),
    GseFile(MouseTissue.LIVER, "GSE225576_Liver.txt.gz", 1_541_746, "2c77673fb9cb7e66917b3d87f841c1f9"),
    GseFile(MouseTissue.LUNG, "GSE225576_Lung.txt.gz", 1_885_384, "0bcd7941d3c738c17bfef2dc91aae83d"),
    GseFile(MouseTissue.MUSCLE, "GSE225576_Muscle.txt.gz", 1_617_447, "d06dc02c19992da5c606aad8deb35266"),
    GseFile(MouseTissue.SKIN, "GSE225576_Skin.txt.gz", 1_816_802, "e15165894827ea0c3dea857ee0dd6dbb"),
)


def download_gse225576(
    destination: Path,
    *,
    tissues: tuple[MouseTissue, ...] = tuple(MouseTissue),
    client: httpx.Client | None = None,
) -> Gse225576Download:
    """Download selected pinned matrices and GEO-derived age metadata."""

    destination.mkdir(parents=True, exist_ok=True)
    selected = frozenset(tissues)
    files = tuple(
        _download_file(item, destination, client=client)
        for item in GSE225576_FILES
        if item.tissue in selected
    )
    metadata_path = destination / "GSE225576_metadata.parquet"
    if not metadata_path.is_file():
        project = download_geo(
            GSE225576_ACCESSION,
            destination / "geofetch",
            processed=False,
            metadata_only=True,
        )
        metadata = normalize_gse225576_metadata(project.metadata)
        metadata.write_parquet(metadata_path)
    return Gse225576Download(files=files, metadata_path=metadata_path)


def load_gse225576(
    destination: Path,
    *,
    tissues: tuple[MouseTissue, ...] = tuple(MouseTissue),
) -> tuple[Gse225576Cohort, ...]:
    """Download, parse, and validate selected tissue cohorts."""

    downloaded = download_gse225576(destination, tissues=tissues)
    metadata = pl.read_parquet(downloaded.metadata_path)
    path_by_name = {path.name: path for path in downloaded.files}
    cohorts: list[Gse225576Cohort] = []
    for tissue in tissues:
        manifest = next(item for item in GSE225576_FILES if item.tissue is tissue)
        path = path_by_name[manifest.filename]
        expression = parse_gse225576_expression(path, tissue)
        samples = tuple(column for column in expression.columns if column != "gene")
        tissue_metadata = (
            metadata.filter(pl.col("tissue") == tissue.value)
            .filter(pl.col("sample").is_in(samples))
            .sort(
                pl.col("sample")
                .replace_strict(
                    {sample: index for index, sample in enumerate(samples)},
                    return_dtype=pl.Int64,
                )
            )
        )
        if tuple(tissue_metadata.get_column("sample").to_list()) != samples:
            raise Gse225576Error(
                f"GEO metadata does not align with {tissue.value} matrix columns"
            )
        cohorts.append(
            Gse225576Cohort(
                tissue=tissue,
                dataset=ExpressionDataset(
                    expression=expression,
                    metadata=tissue_metadata,
                    gene_id_type=GeneIdType.SYMBOL,
                    scale=ExpressionScale.NORMALIZED_COUNTS,
                    source=path,
                ),
            )
        )
    return tuple(cohorts)


def _download_file(
    item: GseFile,
    destination: Path,
    *,
    client: httpx.Client | None,
) -> Path:
    path = destination / item.filename
    if _verify(path, item):
        return path
    request = httpx.get if client is None else client.get
    response = request(item.url, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{item.filename}.",
        suffix=".part",
        dir=destination,
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as stream:
            stream.write(response.content)
            stream.flush()
        if len(response.content) != item.size:
            raise Gse225576Error(
                f"Unexpected size for {item.filename}: {len(response.content)}"
            )
        checksum = hashlib.md5(
            response.content,
            usedforsecurity=False,
        ).hexdigest()
        if checksum != item.md5:
            raise Gse225576Error(
                f"Checksum mismatch for {item.filename}: {checksum}"
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _verify(path: Path, item: GseFile) -> bool:
    if not path.is_file() or path.stat().st_size != item.size:
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item.md5


def parse_gse225576_expression(
    path: Path,
    tissue: MouseTissue,
) -> pl.DataFrame:
    with gzip.open(path, "rb") as compressed:
        prefix = compressed.read(64 * 1024)
    end_of_line = "\r" if b"\r" in prefix and b"\n" not in prefix else "\n"
    raw = pl.read_csv(path, separator="\t", eol_char=end_of_line)
    if "Gene_ID" not in raw.columns:
        raise Gse225576Error(f"{path.name} is missing Gene_ID")
    rename = {"Gene_ID": "gene"}
    for column in raw.columns:
        if column == "Gene_ID":
            continue
        match = _COLUMN_RE.fullmatch(column)
        if match is None:
            raise Gse225576Error(
                f"Unrecognized sample column {column!r} in {path.name}"
            )
        rename[column] = (
            f"{tissue.value.lower()}_{match.group('age')}mo_"
            f"{match.group('replicate')}"
        )
    return raw.rename(rename)


def normalize_gse225576_metadata(metadata: pl.DataFrame) -> pl.DataFrame:
    required = {"sample_name", "sample_geo_accession", "tissue", "age"}
    missing = required.difference(metadata.columns)
    if missing:
        raise Gse225576Error(
            f"GEO metadata is missing required fields: {sorted(missing)}"
        )
    normalized = metadata.select(
        pl.col("sample_name").cast(pl.String).alias("sample"),
        pl.col("sample_geo_accession").cast(pl.String).alias("geo_accession"),
        pl.col("tissue").cast(pl.String),
        pl.col("age").cast(pl.String).alias("geo_age"),
    ).with_columns(
        pl.col("geo_age")
        .str.extract(_AGE_RE.pattern, group_index=1)
        .cast(pl.Int64)
        .alias("_age_characteristic"),
        pl.col("sample")
        .str.extract(r"_(\d+)mo_", group_index=1)
        .cast(pl.Int64)
        .alias("_age_sample_name"),
        pl.col("sample")
        .str.extract(r"_(\d+)$", group_index=1)
        .cast(pl.Int64)
        .alias("replicate"),
    ).with_columns(
        pl.coalesce("_age_characteristic", "_age_sample_name").alias("age_months"),
        pl.when(pl.col("_age_characteristic").is_not_null())
        .then(pl.lit("geo_characteristic"))
        .otherwise(pl.lit("geo_sample_name"))
        .alias("age_source"),
    ).drop(
        "_age_characteristic",
        "_age_sample_name",
    )
    if any(normalized.null_count().row(0)):
        raise Gse225576Error("GEO age/tissue/sample metadata contains null values")
    if normalized.height != 128:
        raise Gse225576Error(
            f"Expected 128 GEO samples, received {normalized.height}"
        )
    return normalized


__all__ = [
    "GSE225576_ACCESSION",
    "GSE225576_FILES",
    "Gse225576Cohort",
    "Gse225576Download",
    "Gse225576Error",
    "GseFile",
    "MouseTissue",
    "download_gse225576",
    "load_gse225576",
    "normalize_gse225576_metadata",
    "parse_gse225576_expression",
]


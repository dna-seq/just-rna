"""Typed catalogs and verified, atomic downloads for clock assets."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TypedDict, cast

import httpx

from just_rna.config import CACHE_ENV_VAR
from just_rna.config import load_settings

TAGE_RECORD_ID = 18_763_485
TAGE_CATALOG_URL = f"https://zenodo.org/api/records/{TAGE_RECORD_ID}"
DEFAULT_TIMEOUT = httpx.Timeout(30.0, read=300.0)
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")


class AssetKind(StrEnum):
    MODEL = "model"
    EXPRESSION = "expression"
    METADATA = "metadata"
    SAMPLE = "sample"


class CacheState(StrEnum):
    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class License:
    name: str
    url: str
    summary: str


@dataclass(frozen=True, slots=True)
class Asset:
    """One downloadable file and its upstream integrity metadata."""

    id: str
    provider: str
    filename: str
    kind: AssetKind
    size: int
    md5: str
    url: str
    license: License
    description: str
    requires_large_download_consent: bool = False

    def __post_init__(self) -> None:
        if Path(self.filename).name != self.filename:
            raise ValueError(
                f"Asset filename must not contain directories: {self.filename}"
            )
        if self.size < 0:
            raise ValueError("Asset size cannot be negative")
        normalized = self.md5.removeprefix("md5:").lower()
        if _MD5_RE.fullmatch(normalized) is None:
            raise ValueError(f"Invalid MD5 checksum for {self.id}: {self.md5}")
        object.__setattr__(self, "md5", normalized)


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    """Typed static and/or remotely cataloged provider manifest."""

    id: str
    name: str
    license: License
    assets: tuple[Asset, ...] = ()
    catalog_url: str | None = None


@dataclass(frozen=True, slots=True)
class AssetStatus:
    asset: Asset
    path: Path
    state: CacheState

    @property
    def downloaded(self) -> bool:
        return self.state is CacheState.VALID


class ChecksumError(OSError):
    """Downloaded bytes did not match their published metadata."""


class LargeDownloadConsentError(PermissionError):
    """A multi-gigabyte Bayesian Ridge model needs explicit consent."""


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


class _ZenodoLinks(TypedDict):
    self: str


class _ZenodoFile(TypedDict):
    key: str
    size: int
    checksum: str
    links: _ZenodoLinks


class _ZenodoRecord(TypedDict):
    id: int
    files: list[_ZenodoFile]


TAGE_LICENSE = License(
    "MGB Open Access License 1.0",
    "https://github.com/Gladyshev-Lab/tAge/blob/main/LICENSE",
    (
        "Non-commercial, non-revenue-generating academic use only; "
        "commercial use requires a separate Mass General Brigham license."
    ),
)
PASTA_LICENSE = License(
    "MIT License",
    "https://github.com/jsalignon/pasta/blob/main/LICENSE",
    "Permissive MIT license; retain the copyright and license notice.",
)
_TAGE_COMMIT = "5d68f95b4a05b6c3b259d78a29c5d8186d4e531e"
_PASTA_COMMIT = "5d3f62c4772d017e625b6f2cc7188c4216f075f6"

TAGE_SAMPLE_ASSETS = (
    Asset(
        "tage:sample-expression",
        "tage",
        "Exprs_example.csv",
        AssetKind.EXPRESSION,
        4_665_228,
        "37bc258a745cee5ab922d9e6f9e3f526",
        (
            "https://raw.githubusercontent.com/Gladyshev-Lab/tAge/"
            f"{_TAGE_COMMIT}/inst/extdata/Exprs_example.csv"
        ),
        TAGE_LICENSE,
        "Real example expression matrix distributed with tAge.",
    ),
    Asset(
        "tage:sample-metadata",
        "tage",
        "Metadata_example.csv",
        AssetKind.METADATA,
        887,
        "5d5976f3c05d0f49dcda67a07d8c221d",
        (
            "https://raw.githubusercontent.com/Gladyshev-Lab/tAge/"
            f"{_TAGE_COMMIT}/inst/extdata/Metadata_example.csv"
        ),
        TAGE_LICENSE,
        "Metadata corresponding to the tAge example expression matrix.",
    ),
)
PASTA_SAMPLE_ASSETS = (
    Asset(
        "pasta:sample-gse103938",
        "pasta",
        "ES_GSE103938.rda",
        AssetKind.SAMPLE,
        1_025_048,
        "8547916ced94f5c35f8a00ed4f0451c7",
        (
            "https://raw.githubusercontent.com/jsalignon/pasta/"
            f"{_PASTA_COMMIT}/data/ES_GSE103938.rda"
        ),
        PASTA_LICENSE,
        "Feasible real GSE103938 ExpressionSet distributed with PASTA.",
    ),
)
PROVIDER_MANIFESTS: Mapping[str, ProviderManifest] = {
    "tage": ProviderManifest(
        "tage",
        "tAge",
        TAGE_LICENSE,
        TAGE_SAMPLE_ASSETS,
        TAGE_CATALOG_URL,
    ),
    "pasta": ProviderManifest(
        "pasta",
        "PASTA",
        PASTA_LICENSE,
        PASTA_SAMPLE_ASSETS,
    ),
}


def cache_directory() -> Path:
    """Return the cache root, honoring ``JUST_RNA_CACHE_DIR`` and ``.env``."""

    return load_settings().cache_directory


def get_cache_dir() -> Path:
    return cache_directory()


def md5sum(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path,
    expected_md5: str,
    expected_size: int | None = None,
) -> bool:
    if not path.is_file():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return md5sum(path) == expected_md5.removeprefix("md5:").lower()


def asset_path(asset: Asset, cache_dir: Path | None = None) -> Path:
    root = cache_directory() if cache_dir is None else cache_dir
    return root / asset.provider / asset.filename


def fetch_tage_catalog(
    *,
    client: httpx.Client | None = None,
    timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
) -> tuple[Asset, ...]:
    """Fetch filename, size, MD5, and content URL from live Zenodo metadata."""

    response = (
        httpx.get(TAGE_CATALOG_URL, timeout=timeout, follow_redirects=True)
        if client is None
        else client.get(TAGE_CATALOG_URL, timeout=timeout, follow_redirects=True)
    )
    response.raise_for_status()
    record = cast(_ZenodoRecord, response.json())
    if record.get("id") != TAGE_RECORD_ID:
        raise ValueError("Zenodo returned an unexpected tAge record")
    return tuple(_tage_asset(item) for item in record["files"])


def _tage_asset(item: _ZenodoFile) -> Asset:
    filename = item["key"]
    size = item["size"]
    checksum = item["checksum"]
    url = item["links"]["self"]
    if not url.startswith("https://"):
        raise ValueError(f"Zenodo returned an invalid URL for {filename}")
    kind = _tage_kind(filename)
    return Asset(
        id=f"tage:zenodo:{filename}",
        provider="tage",
        filename=filename,
        kind=kind,
        size=size,
        md5=checksum,
        url=url,
        license=TAGE_LICENSE,
        description=f"tAge Zenodo {kind.value}: {filename}",
        requires_large_download_consent=(
            kind is AssetKind.MODEL and filename.startswith("BR_")
        ),
    )


def _tage_kind(filename: str) -> AssetKind:
    if filename.startswith(("BR_", "EN_")):
        return AssetKind.MODEL
    if filename.startswith("Data_annotation_"):
        return AssetKind.METADATA
    if filename.startswith("Expression_data_"):
        return AssetKind.EXPRESSION
    return AssetKind.SAMPLE


def list_assets(
    provider: str | None = None,
    *,
    include_dynamic: bool = True,
    client: httpx.Client | None = None,
) -> tuple[Asset, ...]:
    if provider is not None and provider not in PROVIDER_MANIFESTS:
        raise KeyError(f"Unknown asset provider: {provider}")
    manifests = (
        (PROVIDER_MANIFESTS[provider],)
        if provider is not None
        else tuple(PROVIDER_MANIFESTS.values())
    )
    assets = tuple(asset for manifest in manifests for asset in manifest.assets)
    if include_dynamic and any(item.id == "tage" for item in manifests):
        assets += fetch_tage_catalog(client=client)
    return assets


def get_asset(
    asset_id: str,
    *,
    include_dynamic: bool = True,
    client: httpx.Client | None = None,
) -> Asset:
    """Resolve static IDs offline; query Zenodo only for its dynamic IDs."""

    static = list_assets(include_dynamic=False)
    match = next((item for item in static if item.id == asset_id), None)
    if match is not None:
        return match
    if include_dynamic and asset_id.startswith("tage:zenodo:"):
        dynamic = fetch_tage_catalog(client=client)
        match = next((item for item in dynamic if item.id == asset_id), None)
    if match is None:
        raise KeyError(f"Unknown asset: {asset_id}")
    return match


def asset_status(asset: Asset, cache_dir: Path | None = None) -> AssetStatus:
    path = asset_path(asset, cache_dir)
    if not path.exists():
        state = CacheState.MISSING
    elif verify_file(path, asset.md5, asset.size):
        state = CacheState.VALID
    else:
        state = CacheState.INVALID
    return AssetStatus(asset, path, state)


def status(
    asset_or_id: Asset | str,
    cache_dir: Path | None = None,
    *,
    client: httpx.Client | None = None,
) -> AssetStatus:
    asset = (
        get_asset(asset_or_id, client=client)
        if isinstance(asset_or_id, str)
        else asset_or_id
    )
    return asset_status(asset, cache_dir)


def download_asset(
    asset_or_id: Asset | str,
    cache_dir: Path | None = None,
    *,
    allow_large: bool = False,
    force: bool = False,
    client: httpx.Client | None = None,
    timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
) -> Path:
    """Stream one requested file, verify it, then atomically move it in place."""

    asset = (
        get_asset(asset_or_id, client=client)
        if isinstance(asset_or_id, str)
        else asset_or_id
    )
    destination = asset_path(asset, cache_dir)
    if not force and verify_file(destination, asset.md5, asset.size):
        return destination
    if asset.requires_large_download_consent and not allow_large:
        raise LargeDownloadConsentError(
            f"{asset.filename} is {asset.size / (1024**3):.2f} GiB; "
            "pass allow_large=True to download this Bayesian Ridge model"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    digest = hashlib.md5(usedforsecurity=False)
    try:
        if client is None:
            with httpx.stream(
                "GET",
                asset.url,
                timeout=timeout,
                follow_redirects=True,
            ) as response:
                size = _write_stream(response, temporary, digest)
        else:
            with client.stream(
                "GET",
                asset.url,
                timeout=timeout,
                follow_redirects=True,
            ) as response:
                size = _write_stream(response, temporary, digest)
        if size != asset.size or digest.hexdigest() != asset.md5:
            raise ChecksumError(
                f"Integrity check failed for {asset.id}: expected "
                f"{asset.size} bytes/{asset.md5}, received "
                f"{size} bytes/{digest.hexdigest()}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_stream(
    response: httpx.Response,
    destination: Path,
    digest: _Digest,
) -> int:
    response.raise_for_status()
    size = 0
    with destination.open("wb") as target:
        for chunk in response.iter_bytes():
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        target.flush()
        os.fsync(target.fileno())
    return size


def download(
    asset_or_id: Asset | str,
    cache_dir: Path | None = None,
    *,
    allow_large: bool = False,
    force: bool = False,
) -> Path:
    return download_asset(
        asset_or_id,
        cache_dir,
        allow_large=allow_large,
        force=force,
    )


__all__ = [
    "CACHE_ENV_VAR",
    "PROVIDER_MANIFESTS",
    "TAGE_SAMPLE_ASSETS",
    "Asset",
    "AssetKind",
    "AssetStatus",
    "CacheState",
    "ChecksumError",
    "LargeDownloadConsentError",
    "License",
    "ProviderManifest",
    "asset_path",
    "asset_status",
    "cache_directory",
    "download",
    "download_asset",
    "fetch_tage_catalog",
    "get_asset",
    "get_cache_dir",
    "list_assets",
    "md5sum",
    "status",
    "verify_file",
]

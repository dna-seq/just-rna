from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from just_rna.assets import (
    CACHE_ENV_VAR,
    PROVIDER_MANIFESTS,
    TAGE_SAMPLE_ASSETS,
    Asset,
    AssetKind,
    CacheState,
    LargeDownloadConsentError,
    asset_status,
    cache_directory,
    download_asset,
    fetch_tage_catalog,
    get_asset,
    list_assets,
    md5sum,
    verify_file,
)


def test_cache_directory_honors_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "cache"
    monkeypatch.setenv(CACHE_ENV_VAR, str(configured))
    assert cache_directory() == configured


def test_local_checksum_helpers(tmp_path: Path) -> None:
    content = b"just-rna checksum fixture\n"
    path = tmp_path / "fixture.bin"
    path.write_bytes(content)
    checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()

    assert md5sum(path) == checksum
    assert verify_file(path, checksum, len(content))
    assert verify_file(path, f"md5:{checksum}", len(content))
    assert not verify_file(path, "0" * 32, len(content))
    assert not verify_file(path, checksum, len(content) + 1)


def test_static_manifests_are_pinned_and_licensed() -> None:
    assets = list_assets(include_dynamic=False)
    assert {item.id for item in assets} == {
        "tage:sample-expression",
        "tage:sample-metadata",
        "pasta:sample-gse103938",
    }
    assert all("/main/" not in item.url for item in assets)
    assert PROVIDER_MANIFESTS["tage"].license.name == "MGB Open Access License 1.0"
    assert PROVIDER_MANIFESTS["pasta"].license.name == "MIT License"
    assert get_asset("pasta:sample-gse103938").provider == "pasta"


def test_status_detects_missing_valid_and_invalid(tmp_path: Path) -> None:
    content = b"verified"
    asset = _local_asset(content)
    missing = asset_status(asset, tmp_path)
    assert missing.state is CacheState.MISSING

    missing.path.parent.mkdir(parents=True)
    missing.path.write_bytes(content)
    assert asset_status(asset, tmp_path).state is CacheState.VALID
    missing.path.write_bytes(b"corrupt")
    assert asset_status(asset, tmp_path).state is CacheState.INVALID


def test_br_download_needs_explicit_consent(tmp_path: Path) -> None:
    asset = Asset(
        "tage:zenodo:BR_test.pkl",
        "tage",
        "BR_test.pkl",
        AssetKind.MODEL,
        2_000_000_000,
        "0" * 32,
        "https://zenodo.org/example",
        PROVIDER_MANIFESTS["tage"].license,
        "Large test model",
        True,
    )
    with pytest.raises(LargeDownloadConsentError):
        download_asset(asset, tmp_path)
    assert not (tmp_path / "tage" / "BR_test.pkl").exists()


@pytest.mark.integration
def test_live_tage_catalog_uses_published_metadata() -> None:
    by_name = {item.filename: item for item in fetch_tage_catalog()}
    model = by_name["EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl"]

    assert model.size == 584_873
    assert model.md5 == "7dc8a470f5db20a75d7052829efbd37a"
    assert model.url.endswith(
        "/EN_Chronoage_Multispecies_Multitissue_scaleddiff.pkl/content"
    )
    assert all(
        item.requires_large_download_consent
        for item in by_name.values()
        if item.filename.startswith("BR_")
    )


@pytest.mark.integration
def test_real_sample_download_is_verified_and_reused(tmp_path: Path) -> None:
    metadata = next(
        item for item in TAGE_SAMPLE_ASSETS if item.kind is AssetKind.METADATA
    )
    path = download_asset(metadata, tmp_path)
    modified = path.stat().st_mtime_ns

    assert download_asset(metadata, tmp_path) == path
    assert path.stat().st_mtime_ns == modified
    assert verify_file(path, metadata.md5, metadata.size)


def _local_asset(content: bytes) -> Asset:
    return Asset(
        "test:fixture",
        "test",
        "fixture.bin",
        AssetKind.SAMPLE,
        len(content),
        hashlib.md5(content, usedforsecurity=False).hexdigest(),
        "https://example.invalid/fixture.bin",
        PROVIDER_MANIFESTS["pasta"].license,
        "Local fixture",
    )

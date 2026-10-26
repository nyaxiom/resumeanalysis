"""Explicit immutable downloads and full offline asset validation for the reviewed model."""

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from resumeanalysis.jsonio import loads, write_json
from resumeanalysis.rerank.config import MODEL_ID, REVISION

FILES = {
    "README.md": (2278, "474736a65d6393a060119a8dc304563af67af4d8d86ccfee4a05dd0df107fc11"),
    "config.json": (891, "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77"),
    "model.safetensors": (
        470592698,
        "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4",
    ),
    "sentencepiece.bpe.model": (
        5069051,
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "special_tokens_map.json": (
        239,
        "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
    ),
    "tokenizer.json": (
        17082660,
        "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
    ),
    "tokenizer_config.json": (
        435,
        "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e",
    ),
}
MANIFEST = "resumeanalysis_reranker.json"
LICENSE_SOURCE = "https://raw.githubusercontent.com/UKPLab/sentence-transformers/33f9bd81610560e48ff5eac80e910496b49ae193/LICENSE"
LICENSE_SHA = "624b030fd62b8b989dd502d5bb6322cbc0dbb524ab9c9708f32efe33d7275735"


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def expected_manifest():
    return {
        "format_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "license_evidence": f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/README.md",
        "files": {
            name: {"bytes": size, "sha256": checksum} for name, (size, checksum) in FILES.items()
        },
        "license_text": {"file": "APACHE-2.0.txt", "sha256": LICENSE_SHA, "source": LICENSE_SOURCE},
        "score_interpretation": "Raw document-relevance logit, not a calibrated job-fit probability",
    }


def _check(path, size, checksum):
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"Missing regular reranker asset: {path.name}; run the explicit download command"
        )
    if path.stat().st_size != size or sha256_file(path) != checksum:
        raise ValueError(f"Reranker asset checksum/size mismatch: {path.name}")


def verify_assets(directory):
    """Validate the manifest AND every expected byte before any Transformers loader."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("Missing reranker manifest; run the explicit download command")
    manifest = loads(manifest_path.read_text())
    if manifest != expected_manifest():
        raise ValueError("Reranker model/revision/file manifest identity mismatch")
    for name, (size, checksum) in FILES.items():
        _check(directory / name, size, checksum)
    license_path = directory / "APACHE-2.0.txt"
    if (
        license_path.is_symlink()
        or not license_path.is_file()
        or sha256_file(license_path) != LICENSE_SHA
    ):
        raise ValueError("Reranker retained Apache license notice mismatch")
    return manifest


def _download(url, destination, size, checksum):
    if destination.exists():
        _check(destination, size, checksum)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".partial", delete=False
        ) as target:
            temporary = Path(target.name)
            received = 0
            with urllib.request.urlopen(url, timeout=120) as source:
                while chunk := source.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise ValueError("Reranker download exceeds pinned size")
                    target.write(chunk)
        _check(temporary, size, checksum)
        os.replace(temporary, destination)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def download_reranker(directory="artifacts/reranker"):
    """Only this explicitly invoked function performs network I/O."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / MANIFEST).exists():
        return verify_assets(directory)
    for name, (size, checksum) in FILES.items():
        _download(
            f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
            directory / name,
            size,
            checksum,
        )
    _download(LICENSE_SOURCE, directory / "APACHE-2.0.txt", 11338, LICENSE_SHA)
    write_json(directory / MANIFEST, expected_manifest())
    return verify_assets(directory)


def asset_identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

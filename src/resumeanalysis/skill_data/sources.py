"""Pinned downloads with original bytes and notices preserved outside Git."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path, PurePosixPath

from resumeanalysis.jsonio import loads, write_json

TEST_SHA = "8438eda34939a5877b984ea5daf9e6ff39b2d5c7e3d73cca0499d3de3784d24a"
SEALED_SHAS = {
    TEST_SHA,
    "9c9fe96da61d730a29d410ff3a534d5a6bb4bfafe1ecb17c9ad49d1ba31b401d",
    "2daa75ce244fe9401c8c3e5f3a7d196547a17a2dcb0f5829b82e4cf67c44bdfb",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def catalog(path):
    result = loads(Path(path).read_text())
    if result.get("schema_version") != 1:
        raise ValueError("Unknown catalog version")
    identifiers = set()
    for dataset in result["datasets"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset["id"]):
            raise ValueError("Dataset id must be a simple slug")
        if not re.fullmatch(r"[a-f0-9]{40}", dataset["revision"]):
            raise ValueError("Dataset revision must be a pinned 40-character commit")
        if dataset["id"] in identifiers:
            raise ValueError("Duplicate dataset id")
        identifiers.add(dataset["id"])
        if dataset["license"] not in {"MIT", "CC-BY-4.0"}:
            raise ValueError("Unreviewed license: download disabled")
        if not dataset["base_url"].startswith("https://"):
            raise ValueError("Dataset downloads require HTTPS")
        for asset in dataset["assets"]:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", asset["split"]):
                raise ValueError("Asset split must be a simple slug")
            path = PurePosixPath(asset["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe asset path")
            if asset["sha256"] in SEALED_SHAS and asset["role"] != "sealed_test":
                raise ValueError("SkillSpan-derived tests must remain sealed")
    return result


def asset_path(root, dataset, asset):
    root = Path(root).resolve()
    path = root / dataset["id"] / dataset["revision"] / asset["path"]
    if not path.resolve().is_relative_to(root):
        raise ValueError("Asset escapes raw directory")
    return path


def verify(path, asset):
    if Path(path).stat().st_size != asset["bytes"] or file_sha(path) != asset["sha256"]:
        raise ValueError(f"Source checksum/size mismatch: {path}")


def download(catalog_path, root):
    document = catalog(catalog_path)
    for dataset in document["datasets"]:
        for asset in dataset["assets"]:
            destination = asset_path(root, dataset, asset)
            if destination.exists():
                verify(destination, asset)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            if temporary.exists():
                raise FileExistsError(temporary)
            try:
                with urllib.request.urlopen(dataset["base_url"] + asset["path"], timeout=90) as src:
                    with temporary.open("xb") as target:
                        size = 0
                        while block := src.read(1024 * 1024):
                            size += len(block)
                            if size > asset["bytes"]:
                                raise ValueError("Download exceeds pinned asset size")
                            target.write(block)
                verify(temporary, asset)
                temporary.rename(destination)
            finally:
                temporary.unlink(missing_ok=True)
        # Derived metadata is separate from unchanged downloaded files.
        notice = Path(root) / dataset["id"] / "ATTRIBUTION.json"
        value = {k: v for k, v in dataset.items() if k != "assets"}
        if notice.exists() and loads(notice.read_text()) != value:
            raise ValueError("Refusing to replace changed attribution metadata")
        write_json(notice, value)
    return {"datasets": len(document["datasets"]), "catalog_sha256": file_sha(catalog_path)}


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

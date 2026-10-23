"""Explicit pinned downloads, retaining upstream terms and shard checksums."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from resumeanalysis.data.download import sha256_file
from resumeanalysis.jsonio import write_json
from resumeanalysis.qwen.config import QwenConfig


def download(config: QwenConfig) -> dict:
    info = HfApi().model_info(config.model_id, revision=config.revision, files_metadata=True)
    if info.card_data is None or info.card_data.get("license") != "apache-2.0":
        raise ValueError("Model license missing or changed; review before download")
    names = {s.rfilename for s in info.siblings}
    if "LICENSE" not in names:
        raise ValueError("Model has no LICENSE file; manual review required")
    destination = Path(config.model_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use an empty model destination; do not mix model revisions")
    snapshot_download(
        config.model_id,
        revision=info.sha,
        local_dir=destination,
        max_workers=2,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "LICENSE*",
            "NOTICE*",
            "README.md",
        ],
    )
    checksums = {}
    for item in info.siblings:
        path = destination / item.rfilename
        if path.is_file():
            digest = sha256_file(path)
            if item.lfs and digest != item.lfs.sha256:
                raise ValueError("Downloaded Qwen artifact checksum mismatch")
            checksums[item.rfilename] = digest
    if not any(name.endswith(".safetensors") for name in checksums):
        raise ValueError("No safetensors weights available")
    result = {
        "model_id": config.model_id,
        "revision": info.sha,
        "license": "apache-2.0",
        "sha256": checksums,
    }
    write_json(destination / "resumeanalysis_qwen.json", result)
    return result

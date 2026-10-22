"""Explicit downloads only; inference never accesses the network."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

E5_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
CANDIDATE_ID = "michaelozon/candidate-matching-synthetic"
CANDIDATE_REVISION = "178ab864dcad9910c5670d43e4bdbbb901a11f18"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_e5(destination: str | Path, revision: str = E5_REVISION) -> None:
    info = HfApi().model_info(E5_ID, revision=revision, files_metadata=True)
    if info.card_data is None or info.card_data.get("license") != "mit":
        raise ValueError("E5 license changed or is missing; review the model card first")
    destination = Path(destination)
    snapshot_download(
        E5_ID,
        revision=info.sha,
        local_dir=destination,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "README.md",
        ],
        max_workers=2,
    )
    weights = next(s for s in info.siblings if s.rfilename == "model.safetensors")
    digest = sha256_file(destination / "model.safetensors")
    if not weights.lfs or digest != weights.lfs.sha256:
        raise ValueError("E5 safetensors SHA256 does not match Hugging Face metadata")
    (destination / "resumeanalysis_model.json").write_text(
        json.dumps(
            {
                "model_id": E5_ID,
                "revision": info.sha,
                "license": "mit",
                "sha256": digest,
            },
            indent=2,
        )
    )


def download_candidates(destination: str | Path, revision: str = CANDIDATE_REVISION) -> None:
    info = HfApi().dataset_info(CANDIDATE_ID, revision=revision, files_metadata=True)
    if info.card_data is None or info.card_data.get("license") != "mit":
        raise ValueError("Dataset license missing or changed; inspect before use")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in [
        "resumes/train-00000-of-00001.parquet",
        "jobs/train-00000-of-00001.parquet",
        "matches/train-00000-of-00001.parquet",
        "README.md",
    ]:
        path = hf_hub_download(
            CANDIDATE_ID, name, repo_type="dataset", revision=info.sha, local_dir=destination
        )
        digest = sha256_file(path)
        metadata = next(s for s in info.siblings if s.rfilename == name)
        if metadata.lfs and digest != metadata.lfs.sha256:
            raise ValueError(f"Dataset SHA256 mismatch: {name}")
        checksums[name] = digest
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": CANDIDATE_ID,
                "revision": info.sha,
                "license": "mit",
                "label_provenance": "Qwen-generated profiles/jobs, heuristic skill-overlap matches",
                "sha256": checksums,
            },
            indent=2,
        )
    )

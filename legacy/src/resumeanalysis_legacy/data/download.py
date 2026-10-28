"""Explicit downloads only; inference never accesses the network."""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from resumeanalysis.assets import sha256_file

CANDIDATE_ID = "michaelozon/candidate-matching-synthetic"
CANDIDATE_REVISION = "178ab864dcad9910c5670d43e4bdbbb901a11f18"


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

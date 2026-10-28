"""Explicit pinned E5 download; inference remains offline."""

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from resumeanalysis.assets import sha256_file

E5_ID = "intfloat/multilingual-e5-small"
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


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

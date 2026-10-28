# Cross-encoder selection and asset provenance

The experimental reranking backend uses
[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
at revision `1427fd652930e4ba29e8149678df786c240d8825`. This is a multilingual
passage-ranking model with a publicly available safetensors checkpoint. Its
publisher lists English and Russian among 14 training languages. The model
produces a raw document-relevance logit; this is not a calibrated probability
that a candidate fits a job.

The initial source inspection was recorded on 2026-10-23 in
[rerank_model_evidence.json](rerank_model_evidence.json). The selected model was
downloaded and executed locally. Alternative-model metadata was inspected;
alternative weights were not downloaded or benchmarked.

## Candidate comparison

| Model | Reviewed revision | Architecture and weight bytes | License evidence | Decision |
| --- | --- | --- | --- | --- |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | `XLMRobertaForSequenceClassification`, 12 layers, width 384, one logit; 470,592,698 bytes | Explicit Apache-2.0 declaration in the pinned model card | Selected as the requested multilingual CPU experiment; hiring quality remains to be measured. |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | `XLMRobertaForSequenceClassification`, 24 layers, width 1,024; 2,271,071,852 bytes | Explicit Apache-2.0 declaration in the model card | Reasonable multilingual alternative with longer context, but approximately 4.83 times the weight bytes. Its runtime and hiring quality were not measured. |
| `cross-encoder/mmarco-mMiniLMv2-L6-H384-v1` | Unverified | Metadata request returned HTTP 401 | Unverified | Not assumed to exist or to be publicly usable. |

The selected architecture's positional embedding count is 514, corresponding
to an operational sequence limit of 512 tokens with the XLM-R tokenizer.
The alternative has 8,194 positional embeddings. Its longer context and larger
model do not establish better job matching or acceptable CPU performance.

## What the weights represent

The selected [pinned model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/1427fd652930e4ba29e8149678df786c240d8825/README.md)
states that the model was trained on Google-translated mMARCO, based on MS MARCO
passage ranking. It names
[`nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`](https://huggingface.co/nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large/tree/d828558d1a570cbbb5e62a8dbf85c8f18bf7982a)
as the base model. The final classifier is therefore a transfer-learning
candidate for resume–vacancy relevance, not a hiring model validated by the
publisher. Explicit skill denials, unknown qualifications, experience
requirements and user preferences need separate evidence checks.

The upstream model card is retained byte-for-byte in
[`LICENSES/references/reranker-model-card.txt`](../LICENSES/references/reranker-model-card.txt).
Its SHA-256 is
`474736a65d6393a060119a8dc304563af67af4d8d86ccfee4a05dd0df107fc11`.
The downloaded safetensors SHA-256 is
`5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4`.
All tokenizer, configuration and weight file sizes and digests are pinned in
the source asset manifest and evidence JSON.

## License scope and manual review

The final model's publisher explicitly declares Apache-2.0. The inspected model
tree does not contain a standalone `LICENSE` or `NOTICE`. The repository retains
the model card and a complete, unchanged copy of the Apache license from the
[pinned sentence-transformers project](https://github.com/UKPLab/sentence-transformers/blob/33f9bd81610560e48ff5eac80e910496b49ae193/LICENSE),
including its original copyright text. This copy is identified as a reference
license, not presented as a standalone file from the model tree.

When redistributing these weights, retain the license and attribution material,
identify modifications, and review Apache-2.0 section 4 requirements. The
project's MIT license does not replace the model's Apache-2.0 terms.

Several provenance questions remain for production or redistribution review:

- The inspected base-model card points to Microsoft's MiniLM/unilm work but
  does not itself establish a model-specific license grant. The final model's
  publisher independently declares Apache-2.0; downstream rights in all base
  assets have not been established by this audit.
- The inspected [mMARCO dataset card](https://huggingface.co/datasets/unicamp-dl/mmarco/tree/6d039c4638c0ba3e46a9cb7b498b145e7edc6230)
  does not establish an explicit dataset license. Original MS MARCO data is
  subject to separate terms. The original repository's MIT license covers its
  software and must not be substituted for dataset permissions. The terms
  endpoint returned HTTP 403 during the inspection.
- Rights or contractual restrictions associated with training data and its
  Google translation require manual review. No mMARCO dataset payload is
  downloaded or redistributed by this backend.

These are concrete limits of the inspected evidence. The final publisher's
license declaration is preserved without claiming that it resolves every
upstream data or base-model question.

## Offline runtime and long documents

Only `download-reranker` performs network I/O. The runtime verifies the exact
manifest and every expected file's size and SHA-256 before using Transformers.
It uses `local_files_only=True`, `trust_remote_code=False`, and safetensors, then
freezes the model and executes batches under `torch.inference_mode()` on CPU.
Missing or altered assets and unreviewed model/revision identifiers fail
explicitly; inference does not download replacements.

`RerankConfig` controls batch size, maximum sequence length, overlap and
long-document policy. Text pairs are jointly encoded with a balanced token
budget that transfers unused capacity to the longer side. The default
`window` policy covers both documents using the Cartesian product of their
overlapping windows and takes the maximum raw logit. The complete window count
must fit `max_windows`; otherwise inference fails before scoring rather than
silently omitting the tail. `error` rejects oversized pairs. Explicit
`truncate` uses the balanced prefixes and reports discarded token counts for
each input. Per-pair diagnostics expose coverage and aggregation.

Maximum-window aggregation is a documented heuristic: one locally relevant
passage can dominate a score. It requires validation on long resumes and job
descriptions. Increasing `max_windows` increases work multiplicatively when
both documents are long. These settings, the asset manifest, pair-packing
version and Transformers version are part of the cross-encoder identity.

Run the actual model smoke after explicitly downloading assets:

```bash
resumeanalysis download-reranker --output artifacts/reranker
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  RESUMEANALYSIS_RUN_RERANK_TESTS=1 \
  pytest -q tests/test_rerank_model.py
```

This opt-in test blocks socket connections and executes authored English,
Russian and unrelated-text pairs. It tests offline operation and a basic
sanity ordering; it does not establish job-domain accuracy. Full comparisons
and their limitations belong to the benchmark report and
[independent relevance evaluation protocol](rerank_relevance_data.md).

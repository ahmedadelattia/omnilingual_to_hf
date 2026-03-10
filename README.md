# omnilingual_to_hf

Convert all OmniASR models from [fairseq2](https://github.com/facebookresearch/fairseq2)
format to [HuggingFace](https://huggingface.co/) `Wav2Vec2ForCTC` checkpoints, and
optionally push them to the HuggingFace Hub.

---

## What this does

OmniASR is a family of multilingual CTC speech recognition models trained at four scales
(300M · 1B · 3B · 7B) across two generations (v1 W2V / v2 CTC).  They are distributed
through the `fairseq2` model hub as part of the `omnilingual_asr` package.

This repository provides a single self-contained script — `convert_all_omni_to_hf.py` —
that:

1. Loads each OmniASR variant from the fairseq2 hub.
2. **Auto-detects** all architecture parameters (layer count, hidden size, FFN width,
   attention heads, vocab size) directly from the model's state dict — no hardcoded
   lookup tables.
3. Writes a complete HuggingFace tokenizer bundle (`vocab.json`,
   `tokenizer_config.json`, `special_tokens_map.json`, `preprocessor_config.json`)
   from the fairseq2 SentencePiece vocabulary.
4. Converts all model weights using a verified fairseq2 → HF key mapping.
5. **Validates parity**: decodes the same audio file with both the original fairseq2
   model and the converted HF model, then compares logits, greedy token IDs, and
   decoded transcripts — all while both models are still in memory.
6. Saves the HF checkpoint locally.
7. Pushes to the HuggingFace Hub (optional).

### Models covered

| Tag | fairseq2 card | HF repo |
|-----|---------------|---------|
| v1 300M | `omniASR_W2V_300M` | `{user}/omniASR-W2V-300M` |
| v1 1B   | `omniASR_W2V_1B`   | `{user}/omniASR-W2V-1B`   |
| v1 3B   | `omniASR_W2V_3B`   | `{user}/omniASR-W2V-3B`   |
| v1 7B   | `omniASR_W2V_7B`   | `{user}/omniASR-W2V-7B`   |
| v2 300M | `omniASR_CTC_300M_v2` | `{user}/omniASR-CTC-300M-v2` |
| v2 1B   | `omniASR_CTC_1B_v2`   | `{user}/omniASR-CTC-1B-v2`   |
| v2 3B   | `omniASR_CTC_3B_v2`   | `{user}/omniASR-CTC-3B-v2`   |
| v2 7B   | `omniASR_CTC_7B_v2`   | `{user}/omniASR-CTC-7B-v2`   |

Both v1 and v2 use the wav2vec2-v2 feature extractor (per-layer layer norm on all 7
conv layers).  Architecture dimensions are detected at runtime from the state dict.

---

## Prerequisites

### 1 — omnilingual conda environment

The script must run inside the `omnilingual` conda environment where `fairseq2` and
`omnilingual_asr` are installed (these are not on PyPI):

```bash
conda activate omnilingual
pip install torch torchaudio transformers huggingface_hub
```

### 2 — fairseq2 asset directory

Point fairseq2 at the local asset store that contains the OmniASR model cards:

```bash
export FAIRSEQ2_ASSETS_DIR=/path/to/omnilingual/fairseq2_assets
```

---

## HuggingFace authentication

You need a HuggingFace account and a token with **write** permissions.
Generate one at https://huggingface.co/settings/tokens.

**Option A — environment variable (recommended for scripts):**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The script reads `HF_TOKEN` automatically and calls `huggingface_hub.login()`.

**Option B — interactive login (token is cached permanently):**

```bash
huggingface-cli login
```

Run once.  The token is stored at `~/.cache/huggingface/token` and picked up on all
future runs without needing the environment variable.

---

## Usage

### Validate with one model first

```bash
export HF_TOKEN=hf_xxxx
export FAIRSEQ2_ASSETS_DIR=/path/to/omnilingual/fairseq2_assets

/path/to/omnilingual/bin/python convert_all_omni_to_hf.py \
    --hf-user YOUR_HF_USERNAME \
    --models omniASR_CTC_300M_v2 \
    --smoke-test-audio /path/to/test.wav \
    --push-to-hub
```

### Convert and push all 8 models

```bash
/path/to/omnilingual/bin/python convert_all_omni_to_hf.py \
    --hf-user YOUR_HF_USERNAME \
    --push-to-hub
```

### Convert locally only (no push)

```bash
/path/to/omnilingual/bin/python convert_all_omni_to_hf.py \
    --output-root /path/to/output/
```

### Convert a specific subset

```bash
/path/to/omnilingual/bin/python convert_all_omni_to_hf.py \
    --models omniASR_W2V_300M omniASR_W2V_1B \
    --hf-user YOUR_HF_USERNAME \
    --push-to-hub
```

### All CLI flags

```
--hf-user           HuggingFace username or organisation (required with --push-to-hub)
--push-to-hub       Push each checkpoint to the HuggingFace Hub after saving locally
--models            Space-separated list of model tags to process (default: all 8)
--output-root       Root directory for local saves (default: converted_models/)
--device            Torch device — cpu or cuda (default: cpu)
--smoke-test-audio  Path to a 16 kHz WAV file for parity validation
```

---

## Parity validation

When `--smoke-test-audio` is provided, the script runs a four-part parity check
**before** releasing the fairseq2 model from memory, so both models are decoded
simultaneously on identical input:

| Check | What is compared | Pass condition |
|-------|-----------------|----------------|
| 1. Logit shape | Output tensor dimensions | Must match exactly |
| 2. Logit values | `torch.allclose(atol=1e-4, rtol=1e-3)` | Numerically identical |
| 3. Token IDs | Greedy-argmax CTC decode | Identical sequences |
| 4. Transcripts | Decoded strings via `vocab.json` | Identical text |

Checks 2–4 all run even if an earlier one fails, giving a complete picture of any
divergence.  Failures are logged as errors but do not abort the batch run.

> **Memory note:** the parity check holds both models in memory at once.  For 7B
> models on CPU this requires approximately 2 × model size in RAM.  Omit
> `--smoke-test-audio` if memory is constrained.

---

## Output structure

For each model the script writes:

```
converted_models/
└── omniASR-CTC-300M-v2/
    ├── config.json               ← Wav2Vec2Config (architecture)
    ├── model.safetensors         ← converted weights
    ├── vocab.json                ← 10,288-token multilingual vocabulary
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── preprocessor_config.json  ← 16 kHz, do_normalize=True
```

### Loading a converted model

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("your-username/omniASR-CTC-300M-v2")
model     = Wav2Vec2ForCTC.from_pretrained("your-username/omniASR-CTC-300M-v2")
model.eval()
```

---

## Key technical details

### Weight mapping (fairseq2 → HuggingFace)

| fairseq2 key | HuggingFace key |
|---|---|
| `encoder_frontend.feature_extractor.layers.{i}.*` | `wav2vec2.feature_extractor.conv_layers.{i}.*` |
| `encoder_frontend.model_dim_proj.*` | `wav2vec2.feature_projection.projection.*` |
| `encoder_frontend.pos_encoder.conv.weight_g` | `wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0` |
| `encoder_frontend.pos_encoder.conv.weight_v` | `wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1` |
| `encoder.layers.{i}.self_attn.{q,k,v}_proj.*` | `wav2vec2.encoder.layers.{i}.attention.{q,k,v}_proj.*` |
| `encoder.layers.{i}.ffn.inner_proj.*` | `wav2vec2.encoder.layers.{i}.feed_forward.intermediate_dense.*` |
| `final_proj.*` | `lm_head.*` |

The positional conv embedding uses PyTorch's `WeightNorm` parametrization in both
frameworks; the `weight_g` / `weight_v` split is preserved exactly.

### CTC blank token

fairseq2 maps the CTC blank to `<s>` at index 0 (the BOS token).  PyTorch's
`ctc_loss` uses `blank=0` by default.  The HF tokenizer config sets
`pad_token = "<s>"`, which causes `Wav2Vec2CTCTokenizer.batch_decode` to
strip blank frames via `skip_special_tokens=True`.

### Architecture auto-detection

All model-size-specific dimensions are read from the state dict at runtime:

```
hidden_size       = encoder.layer_norm.weight.shape[0]
intermediate_size = encoder.layers.0.ffn.inner_proj.weight.shape[0]
n_layers          = number of unique indices in encoder.layers.*
vocab_size        = final_proj.weight.shape[0]
num_attention_heads = hidden_size // 64   (head_dim=64 for all OmniASR sizes)
```

---

## Disclaimer — code generation and human contribution

This script was written with substantial assistance from
[Claude Code](https://claude.ai/claude-code) (Anthropic Claude Sonnet 4.6),
an AI coding assistant, during an interactive development session on 2026-03-10.

### The human's role

The human researcher was not a passive observer — they were the architect and
decision-maker throughout.  Specifically, the human:

- **Wrote and validated the original single-model conversion script**
  (`convert_omni_to_hf.py`) that established the entire technical foundation:
  the fairseq2 → HF key mapping, the CTC blank token convention, the tokenizer
  config structure, and the smoke-test pattern.  This script was confirmed working
  before any AI assistance was sought.
- **Ran `inspect_fairseq2_keys.py`** to map the exact fairseq2 state-dict key
  structure, which is the ground truth all weight mappings are built on.
- **Provided all domain knowledge** about the models: the eight fairseq2 card names,
  that v1 and v2 share the same feature-extractor topology, and that the standard
  naming differs between generations (`W2V` vs `CTC`).
- **Directed every design decision**: self-contained script, auto-detecting
  architecture, idempotent batch runs, and the parity validation feature.
- **Reviewed and approved** the implementation plan before any code was written.
- **Requested the parity validation feature** after reviewing the initial
  implementation, pushing the quality bar higher.

### The AI's role

The AI assistant (Claude Code) contributed:

- Extending the single-model script into a general batch pipeline covering all 8
  variants with architecture auto-detection.
- Writing the parity validation system (logit comparison, greedy CTC decode,
  transcript matching).
- Structuring, commenting, and documenting the code.
- Proposing and discussing implementation trade-offs (memory management, API
  fallback strategies for fairseq2 inference, idempotency design).

### In short

The core technical insight — that OmniASR fairseq2 checkpoints can be faithfully
converted to HuggingFace `Wav2Vec2ForCTC` — was established and validated by the
human researcher.  The AI accelerated the generalisation of that insight into a
production-quality batch tool.  Neither contribution stands without the other.

---

## License

This conversion tooling is provided as-is for research purposes.  The OmniASR model
weights themselves are subject to their own license terms as distributed by the
original authors.

"""Batch-convert all OmniASR models from fairseq2 format to HuggingFace Wav2Vec2ForCTC
and optionally push each converted checkpoint to the HuggingFace Hub.

Covers all 8 variants:
  v1 series  (omniASR_W2V_*)  : 300M · 1B · 3B · 7B
  v2 series  (omniASR_CTC_*_v2): 300M · 1B · 3B · 7B

The script is fully self-contained: all conversion logic lives here and does not
import from other files in this repository.

────────────────────────────────────────────────────────────────────────────────
ENVIRONMENT
────────────────────────────────────────────────────────────────────────────────
Must be run inside the `omnilingual` conda environment where both fairseq2 and
the omnilingual_asr package are installed:

  /home/scladmin/miniconda3/envs/omnilingual/bin/python

The fairseq2 asset store must be pointed at the local asset directory:

  FAIRSEQ2_ASSETS_DIR=/home/scladmin/Research/Projects/omnilingual/fairseq2_assets

────────────────────────────────────────────────────────────────────────────────
HUGGINGFACE AUTHENTICATION
────────────────────────────────────────────────────────────────────────────────
Two options (pick one):

  Option A — Environment variable (recommended for automated runs):
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    The script reads this variable and calls huggingface_hub.login() automatically.

  Option B — Interactive login (token is cached to disk permanently):
    huggingface-cli login
    After this, no environment variable is needed on future runs.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
# Convert a single model (useful for validation before the full batch):
FAIRSEQ2_ASSETS_DIR=.../fairseq2_assets \\
  /path/to/omnilingual/python scripts/convert_all_omni_to_hf.py \\
  --hf-user YOUR_HF_USERNAME \\
  --models omniASR_CTC_300M_v2 \\
  --smoke-test-audio path/to/test.wav \\
  --push-to-hub

# Convert and push all 8 models:
FAIRSEQ2_ASSETS_DIR=.../fairseq2_assets \\
  /path/to/omnilingual/python scripts/convert_all_omni_to_hf.py \\
  --hf-user YOUR_HF_USERNAME \\
  --push-to-hub
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model registry
#
# Each entry maps a short "tag" (used with --models on the CLI) to:
#   fairseq2_card : the model identifier used by fairseq2's hub loader
#   hf_repo_name  : the HF repo name suffix appended to --hf-user
#
# v1 models are identified by the "W2V" prefix in fairseq2 (the original
# wav2vec2 pretraining style); v2 models use the "CTC" prefix.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    # ── v1 series ─────────────────────────────────────────────────────────────
    "omniASR_CTC_300M": {
        "fairseq2_card": "omniASR_CTC_300M",
        "hf_repo_name":  "omniASR-CTC-300M",
        "model_type":    "ctc",
    },
        "omniASR_CTC_300M_v2": {
        "fairseq2_card": "omniASR_CTC_300M_v2",
        "hf_repo_name":  "omniASR-CTC-300M-v2",
        "model_type":    "ctc",
    },

    "omniASR_W2V_300M": {
        "fairseq2_card": "omniASR_W2V_300M",
        "hf_repo_name":  "omniASR-W2V-300M",
        "model_type":    "ssl",
    },
    "omniASR_CTC_1B": {
        "fairseq2_card": "omniASR_CTC_1B",
        "hf_repo_name":  "omniASR-CTC-1B",
        "model_type":    "ctc",
    },
        "omniASR_CTC_1B_v2": {
        "fairseq2_card": "omniASR_CTC_1B_v2",
        "hf_repo_name":  "omniASR-CTC-1B-v2",
        "model_type":    "ctc",
    },
        "omniASR_W2V_1B": {
        "fairseq2_card": "omniASR_W2V_1B",
        "hf_repo_name":  "omniASR-W2V-1B",
        "model_type":    "ssl",
    },
    "omniASR_CTC_3B": {
        "fairseq2_card": "omniASR_CTC_3B",
        "hf_repo_name":  "omniASR-CTC-3B",
        "model_type":    "ctc",
    },

    "omniASR_CTC_3B_v2": {
        "fairseq2_card": "omniASR_CTC_3B_v2",
        "hf_repo_name":  "omniASR-CTC-3B-v2",
        "model_type":    "ctc",
    },
    "omniASR_W2V_3B": {
        "fairseq2_card": "omniASR_W2V_3B",
        "hf_repo_name":  "omniASR-W2V-3B",
        "model_type":    "ssl",
    },
    "omniASR_CTC_7B": {
        "fairseq2_card": "omniASR_CTC_7B",
        "hf_repo_name":  "omniASR-CTC-7B",
        "model_type":    "ctc",
    },

    "omniASR_CTC_7B_v2": {
        "fairseq2_card": "omniASR_CTC_7B_v2",
        "hf_repo_name":  "omniASR-CTC-7B-v2",
        "model_type":    "ctc",
    },
    "omniASR_W2V_7B": {
        "fairseq2_card": "omniASR_W2V_7B",
        "hf_repo_name":  "omniASR-W2V-7B",
        "model_type":    "ssl",
    },


}

# ──────────────────────────────────────────────────────────────────────────────
# fairseq2 → HuggingFace key mappings
#
# Confirmed key structure (via inspect_fairseq2_keys.py):
#
#   Feature extractor (7 conv layers, all with layer norm):
#     encoder_frontend.feature_extractor.layers.{i}.conv.{weight,bias}
#     encoder_frontend.feature_extractor.layers.{i}.layer_norm.{weight,bias}
#
#   Feature projection:
#     encoder_frontend.post_extract_layer_norm.{weight,bias}
#     encoder_frontend.model_dim_proj.{weight,bias}
#
#   Positional conv embedding (weight-norm parametrized):
#     encoder_frontend.pos_encoder.conv.bias
#     encoder_frontend.pos_encoder.conv.weight_g  → parametrizations.weight.original0
#     encoder_frontend.pos_encoder.conv.weight_v  → parametrizations.weight.original1
#
#   Encoder transformer:
#     encoder.layer_norm.{weight,bias}
#     encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
#     encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
#     encoder.layers.{i}.self_attn.output_proj.{weight,bias}
#     encoder.layers.{i}.ffn_layer_norm.{weight,bias}
#     encoder.layers.{i}.ffn.inner_proj.{weight,bias}
#     encoder.layers.{i}.ffn.output_proj.{weight,bias}
#
#   CTC head:
#     final_proj.{weight,bias}  →  lm_head
#
#   HF-only (not in fairseq2, zeroed out):
#     wav2vec2.masked_spec_embed  (spectral masking embedding, only used during training)
# ──────────────────────────────────────────────────────────────────────────────

# Non-layer-indexed key mapping (one-to-one renaming)
_STATIC_MAP: Dict[str, str] = {
    # ── feature projection ────────────────────────────────────────────────────
    "encoder_frontend.post_extract_layer_norm.weight": "wav2vec2.feature_projection.layer_norm.weight",
    "encoder_frontend.post_extract_layer_norm.bias":   "wav2vec2.feature_projection.layer_norm.bias",
    "encoder_frontend.model_dim_proj.weight":          "wav2vec2.feature_projection.projection.weight",
    "encoder_frontend.model_dim_proj.bias":            "wav2vec2.feature_projection.projection.bias",
    # ── positional conv embedding (weight-norm parametrization) ───────────────
    # fairseq2 stores the decomposed weight-norm factors weight_g (scale) and
    # weight_v (direction); HF re-parametrizes them under torch's WeightNorm hook.
    "encoder_frontend.pos_encoder.conv.bias":    "wav2vec2.encoder.pos_conv_embed.conv.bias",
    "encoder_frontend.pos_encoder.conv.weight_g": "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    "encoder_frontend.pos_encoder.conv.weight_v": "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    # ── encoder top-level layer norm ──────────────────────────────────────────
    "encoder.layer_norm.weight": "wav2vec2.encoder.layer_norm.weight",
    "encoder.layer_norm.bias":   "wav2vec2.encoder.layer_norm.bias",
    # ── CTC projection head ───────────────────────────────────────────────────
    "final_proj.weight": "lm_head.weight",
    "final_proj.bias":   "lm_head.bias",
}

# Per-layer sub-key mapping applied inside encoder.layers.{i}
_LAYER_SUBKEY_MAP: Dict[str, str] = {
    # Self-attention projections
    "self_attn.q_proj.weight":    "attention.q_proj.weight",
    "self_attn.q_proj.bias":      "attention.q_proj.bias",
    "self_attn.k_proj.weight":    "attention.k_proj.weight",
    "self_attn.k_proj.bias":      "attention.k_proj.bias",
    "self_attn.v_proj.weight":    "attention.v_proj.weight",
    "self_attn.v_proj.bias":      "attention.v_proj.bias",
    "self_attn.output_proj.weight": "attention.out_proj.weight",
    "self_attn.output_proj.bias":   "attention.out_proj.bias",
    # Pre-attention layer norm
    "self_attn_layer_norm.weight":  "layer_norm.weight",
    "self_attn_layer_norm.bias":    "layer_norm.bias",
    # Feed-forward network
    "ffn.inner_proj.weight":  "feed_forward.intermediate_dense.weight",
    "ffn.inner_proj.bias":    "feed_forward.intermediate_dense.bias",
    "ffn.output_proj.weight": "feed_forward.output_dense.weight",
    "ffn.output_proj.bias":   "feed_forward.output_dense.bias",
    # Pre-FFN layer norm (called "final_layer_norm" in HF)
    "ffn_layer_norm.weight":  "final_layer_norm.weight",
    "ffn_layer_norm.bias":    "final_layer_norm.bias",
}


def build_key_mapping(n_layers: int) -> Dict[str, str]:
    """Build the complete fairseq2 → HF state-dict key mapping for a given depth.

    Args:
        n_layers: Number of transformer encoder layers.

    Returns:
        Dict mapping every fairseq2 state-dict key to its HF equivalent.
    """
    mapping = dict(_STATIC_MAP)

    # ── Feature extractor conv layers (7 layers, indices 0–6) ─────────────────
    for i in range(7):
        fs2_prefix = f"encoder_frontend.feature_extractor.layers.{i}"
        hf_prefix  = f"wav2vec2.feature_extractor.conv_layers.{i}"
        for suffix in ("conv.weight", "conv.bias", "layer_norm.weight", "layer_norm.bias"):
            mapping[f"{fs2_prefix}.{suffix}"] = f"{hf_prefix}.{suffix}"

    # ── Transformer encoder layers ────────────────────────────────────────────
    for i in range(n_layers):
        fs2_prefix = f"encoder.layers.{i}"
        hf_prefix  = f"wav2vec2.encoder.layers.{i}"
        for fs2_sub, hf_sub in _LAYER_SUBKEY_MAP.items():
            mapping[f"{fs2_prefix}.{fs2_sub}"] = f"{hf_prefix}.{hf_sub}"

    return mapping


def detect_arch(fs2_sd: Dict[str, Tensor]) -> Dict:
    """Auto-detect all architecture hyperparameters from the fairseq2 state dict.

    We avoid hard-coding per-size values so the same function works for all
    300M / 1B / 3B / 7B variants without any look-up tables.

    Derived values:
      n_layers          — number of unique indices in encoder.layers.*
      vocab_size        — row count of the CTC projection weight
      hidden_size       — width of the encoder's top-level layer norm
      intermediate_size — row count of the FFN inner (expansion) projection
      num_attention_heads — hidden_size ÷ 64  (all OmniASR models use head_dim=64)

    Args:
        fs2_sd: State dict from the loaded fairseq2 model.

    Returns:
        Dict with keys: n_layers, vocab_size, hidden_size, intermediate_size,
        num_attention_heads.
    """
    # Number of encoder layers: count unique integer indices in encoder.layers.{i}.*
    layer_indices = {
        int(k.split(".")[2])
        for k in fs2_sd
        if k.startswith("encoder.layers.")
    }
    n_layers = len(layer_indices)

    # Vocab size from the CTC head weight shape: (vocab_size, hidden_size)
    vocab_size = fs2_sd["final_proj.weight"].shape[0]

    # hidden_size from the encoder's top-level layer norm (shape: [hidden_size])
    hidden_size = fs2_sd["encoder.layer_norm.weight"].shape[0]

    # intermediate_size from the FFN's expansion layer: (intermediate_size, hidden_size)
    intermediate_size = fs2_sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]

    # Attention head count: OmniASR uses head_dim = 64 throughout all sizes
    num_attention_heads = hidden_size // 64

    arch = dict(
        n_layers=n_layers,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )
    logger.info(
        "Detected architecture: layers=%d  vocab=%d  hidden=%d  "
        "intermediate=%d  heads=%d",
        n_layers, vocab_size, hidden_size, intermediate_size, num_attention_heads,
    )
    return arch


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace config + weight conversion
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_config(
    n_layers: int,
    vocab_size: int,
    ctc_blank_token_id: int,
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: int,
):
    """Build a Wav2Vec2Config that matches the given OmniASR architecture.

    Both v1 and v2 series share the same feature extractor topology
    (7 conv layers with per-layer layer normalization) and positional
    conv embedding hyperparameters; only depth / width vary across sizes.

    Args:
        n_layers:             Number of transformer encoder layers.
        vocab_size:           CTC output vocabulary size.
        ctc_blank_token_id:   Token ID used as the CTC blank symbol.
        hidden_size:          Transformer hidden / model dimension.
        num_attention_heads:  Number of self-attention heads.
        intermediate_size:    FFN expansion width.

    Returns:
        Wav2Vec2Config ready to instantiate a Wav2Vec2ForCTC.
    """
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        # OmniASR uses pre-norm (LayerNorm before attention + FFN).
        # In HF this is `do_stable_layer_norm=True`.
        do_stable_layer_norm=True,
        # All 7 feature-extractor conv layers have individual layer norms
        # (wav2vec2 v2 style, as opposed to group norm in the original v1).
        feat_extract_norm="layer",
        # Feature extractor topology (identical for all OmniASR variants)
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        # fairseq2 feature extractor conv layers include a bias term
        conv_bias=True,
        # Positional conv embedding: kernel=128, groups=16 (OmniASR default)
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        vocab_size=vocab_size,
        # pad_token_id doubles as the CTC blank token in Wav2Vec2ForCTC
        pad_token_id=ctc_blank_token_id,
        ctc_loss_reduction="mean",
        add_adapter=False,
    )


def convert_fairseq2_to_hf(
    fs2_sd: Dict[str, Tensor],
    arch: Dict,
    ctc_blank_token_id: int,
):
    """Map a fairseq2 OmniASR state dict into a freshly initialised Wav2Vec2ForCTC.

    Steps:
      1. Build the full key mapping for this depth.
      2. Rename all fairseq2 keys to their HF equivalents.
      3. Initialise a HF model from a matching Wav2Vec2Config.
      4. Inject the renamed weights (strict=False to allow the HF-only
         `masked_spec_embed` to be zeroed out separately).
      5. Log any keys that were dropped or remain uninitialised.

    Args:
        fs2_sd:              fairseq2 model state dict.
        arch:                Architecture dict from detect_arch().
        ctc_blank_token_id:  Token ID for CTC blank (used in config.pad_token_id).

    Returns:
        Wav2Vec2ForCTC in eval mode with all fairseq2 weights loaded.
    """
    from transformers import Wav2Vec2ForCTC

    mapping = build_key_mapping(arch["n_layers"])
    config  = build_hf_config(
        n_layers=arch["n_layers"],
        vocab_size=arch["vocab_size"],
        ctc_blank_token_id=ctc_blank_token_id,
        hidden_size=arch["hidden_size"],
        num_attention_heads=arch["num_attention_heads"],
        intermediate_size=arch["intermediate_size"],
    )

    # Rename fairseq2 keys → HF keys; collect any unmapped keys for logging
    hf_sd: Dict[str, Tensor] = {}
    unmapped: List[str] = []

    for fs2_key, tensor in fs2_sd.items():
        if fs2_key in mapping:
            # Convert to float32 — inference precision; matches the fairseq2 load call
            hf_sd[mapping[fs2_key]] = tensor.float().clone()
        else:
            unmapped.append(fs2_key)

    if unmapped:
        logger.warning(
            "%d fairseq2 keys had no HF mapping and were dropped:\n  %s",
            len(unmapped),
            "\n  ".join(unmapped),
        )

    # Instantiate HF model skeleton from the matched config
    model = Wav2Vec2ForCTC(config)
    hf_expected = set(model.state_dict().keys())

    # `masked_spec_embed` exists in HF for masked spec augmentation during training
    # but has no fairseq2 equivalent — initialise to zeros (unused at inference).
    masked_key = "wav2vec2.masked_spec_embed"
    if masked_key in hf_expected and masked_key not in hf_sd:
        hf_sd[masked_key] = torch.zeros(config.hidden_size)

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)

    if missing:
        logger.warning("HF keys not loaded (missing from fairseq2 mapping): %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in mapped state dict (not in HF model): %s", unexpected)

    n_mapped   = len(hf_expected) - len(missing)
    n_total_hf = len(hf_expected)
    logger.info("Loaded %d / %d HF parameters from fairseq2 checkpoint.", n_mapped, n_total_hf)

    model.eval()
    return model


def detect_arch_ssl(fs2_sd: Dict[str, Tensor]) -> Dict:
    """Auto-detect architecture hyperparameters for a fairseq2 SSL Wav2Vec2 model.

    Same as detect_arch() but does not look for a CTC projection head
    (final_proj), since SSL pre-training models have no CTC output layer.

    Returns:
        Dict with keys: n_layers, hidden_size, intermediate_size,
        num_attention_heads.
    """
    layer_indices = {
        int(k.split(".")[2])
        for k in fs2_sd
        if k.startswith("encoder.layers.")
    }
    n_layers = len(layer_indices)
    hidden_size = fs2_sd["encoder.layer_norm.weight"].shape[0]
    intermediate_size = fs2_sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    num_attention_heads = hidden_size // 64

    arch = dict(
        n_layers=n_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )
    logger.info(
        "Detected SSL architecture: layers=%d  hidden=%d  intermediate=%d  heads=%d",
        n_layers, hidden_size, intermediate_size, num_attention_heads,
    )
    return arch


def build_hf_config_ssl(
    n_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    intermediate_size: int,
):
    """Build a Wav2Vec2Config for a SSL (feature-extractor) model.

    Identical to build_hf_config() except there is no vocab_size / CTC head.
    The resulting config is used to instantiate Wav2Vec2Model (no lm_head).
    """
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        hidden_size=hidden_size,
        num_hidden_layers=n_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        do_stable_layer_norm=True,
        feat_extract_norm="layer",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
    )


def convert_fairseq2_to_hf_ssl(
    fs2_sd: Dict[str, Tensor],
    arch: Dict,
):
    """Map a fairseq2 SSL Wav2Vec2 state dict into a HuggingFace Wav2Vec2Model.

    Only the encoder weights are transferred.  SSL-specific keys (final_proj,
    quantizer, project_q, etc.) that have no HF equivalent are silently dropped.

    Returns:
        Wav2Vec2Model in eval mode with all encoder weights loaded.
    """
    from transformers import Wav2Vec2Model

    # Build key mapping (includes final_proj → lm_head entries, but Wav2Vec2Model
    # has no lm_head so those will land in `unexpected` and be ignored).
    mapping = build_key_mapping(arch["n_layers"])
    config  = build_hf_config_ssl(
        n_layers=arch["n_layers"],
        hidden_size=arch["hidden_size"],
        num_attention_heads=arch["num_attention_heads"],
        intermediate_size=arch["intermediate_size"],
    )

    hf_sd: Dict[str, Tensor] = {}
    unmapped: List[str] = []

    for fs2_key, tensor in fs2_sd.items():
        if fs2_key in mapping:
            # Wav2Vec2Model state dict has no "wav2vec2." prefix (unlike
            # Wav2Vec2ForCTC which nests everything under self.wav2vec2).
            hf_key = mapping[fs2_key].removeprefix("wav2vec2.")
            hf_sd[hf_key] = tensor.float().clone()
        else:
            unmapped.append(fs2_key)

    if unmapped:
        logger.info(
            "%d fairseq2 SSL-only keys dropped (no HF encoder equivalent):\n  %s",
            len(unmapped), "\n  ".join(unmapped),
        )

    model = Wav2Vec2Model(config)
    hf_expected = set(model.state_dict().keys())

    # masked_spec_embed has no "wav2vec2." prefix in Wav2Vec2Model
    masked_key = "masked_spec_embed"
    if masked_key in hf_expected and masked_key not in hf_sd:
        hf_sd[masked_key] = torch.zeros(config.hidden_size)

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)

    if missing:
        logger.warning("HF keys not loaded (missing from fairseq2 mapping): %s", missing)
    if unexpected:
        logger.info(
            "Mapped keys not in Wav2Vec2Model (SSL head weights, expected): %s", unexpected
        )

    n_mapped   = len(hf_expected) - len(missing)
    n_total_hf = len(hf_expected)
    logger.info("Loaded %d / %d HF parameters from fairseq2 SSL checkpoint.", n_mapped, n_total_hf)

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer / preprocessor files
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_tokenizer(model_card: str, output_dir: str) -> int:
    """Extract the fairseq2 tokenizer vocab and write Wav2Vec2CTCTokenizer files.

    Writes four files to `output_dir`:
      vocab.json             — token string → integer ID mapping
      tokenizer_config.json  — tokenizer class + special token declarations
      special_tokens_map.json
      preprocessor_config.json  — Wav2Vec2FeatureExtractor settings

    The CTC blank token is <s> (index 0).  In fairseq2's wav2vec2_asr the
    ctc_loss blank defaults to 0, and skip_special_tokens=True in the decoder
    removes <s> from the output — matching standard CTC blank removal.

    Args:
        model_card:  fairseq2 model card identifier.
        output_dir:  Local directory to write the tokenizer files into.

    Returns:
        blank_id: The integer ID of the CTC blank token (always 0 for OmniASR).
    """
    from fairseq2.data.tokenizers.hub import load_tokenizer

    logger.info("Loading fairseq2 tokenizer: %s", model_card)
    tokenizer = load_tokenizer(model_card)

    # fairseq2 ≥0.6: tokenizer._model is a SentencePieceModel with
    # .vocabulary_size (int) and .index_to_token(i) → str
    spm = tokenizer._model
    vi  = tokenizer.vocab_info

    # Build the token → ID mapping by iterating the full vocabulary
    vocab: Dict[str, int] = {}
    for i in range(spm.vocabulary_size):
        vocab[spm.index_to_token(i)] = i

    # CTC blank = BOS = <s> at index 0.  PyTorch ctc_loss uses blank=0 by default.
    blank_id = vi.bos_idx  # always 0
    logger.info(
        "Vocab: size=%d  blank/bos=%d  eos=%d  unk=%d",
        spm.vocabulary_size, blank_id, vi.eos_idx, vi.unk_idx,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── vocab.json ────────────────────────────────────────────────────────────
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # ── tokenizer_config.json ─────────────────────────────────────────────────
    # pad_token is set to <s> (the CTC blank) so that Wav2Vec2CTCTokenizer's
    # `skip_special_tokens=True` path correctly removes blank frames.
    tokenizer_config = {
        "tokenizer_class": "Wav2Vec2CTCTokenizer",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<s>",          # CTC blank = <s> at index 0
        "word_delimiter_token": "▁", # SentencePiece word-boundary marker
        "do_lower_case": False,
        "replace_word_delimiter_char": " ",
    }
    with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    # ── special_tokens_map.json ───────────────────────────────────────────────
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<s>",  # CTC blank = <s> at index 0
    }
    with open(out / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2)

    # ── preprocessor_config.json ──────────────────────────────────────────────
    # Wav2Vec2FeatureExtractor settings — identical for all OmniASR variants
    # (all trained on 16 kHz audio with mean-variance normalisation)
    preprocessor_config = {
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1,
        "sampling_rate": 16000,
        "padding_value": 0.0,
        "do_normalize": True,   # instance (mean-variance) normalisation per utterance
        "return_attention_mask": False,
    }
    with open(out / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor_config, f, indent=2)

    logger.info(
        "Tokenizer files written to %s  (vocab_size=%d, blank_id=%d)",
        output_dir, spm.vocabulary_size, blank_id,
    )
    return blank_id


# ──────────────────────────────────────────────────────────────────────────────
# Parity validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_audio_16k(audio_path: str) -> Tensor:
    """Load an audio file, resample to 16 kHz, and return a 1-D float32 tensor."""
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform.squeeze(0)  # (T,)


def _normalize_waveform(waveform: Tensor) -> Tensor:
    """Apply zero-mean unit-variance normalisation (instance norm per utterance).

    Both HF Wav2Vec2FeatureExtractor (do_normalize=True) and fairseq2's
    wav2vec2 frontend apply this same normalisation to the raw waveform before
    the CNN feature extractor.  Applying it once here ensures both models
    receive exactly the same input.
    """
    mean = waveform.mean()
    std  = waveform.std()
    return (waveform - mean) / (std + 1e-7)


def _ctc_greedy_decode(logits: Tensor, blank_id: int = 0) -> List[int]:
    """CTC greedy decode: argmax → collapse repeats → remove blanks.

    Args:
        logits:   Shape (1, time, vocab_size) or (time, vocab_size).
        blank_id: Token index used as CTC blank.

    Returns:
        List of decoded (non-blank, non-repeated) token IDs.
    """
    token_ids = logits.squeeze(0).argmax(dim=-1).tolist()  # (time,)
    result = []
    prev = -1
    for t in token_ids:
        if t != blank_id and t != prev:
            result.append(t)
        prev = t
    return result


def _run_fairseq2_inference(fs2_model, waveform_norm: Tensor) -> Optional[Tensor]:
    """Forward-pass a normalised waveform through the fairseq2 omnilingual model.

    Uses the BatchLayout API that omnilingual's ASRInferencePipeline uses
    internally for Wav2Vec2AsrModel (CTC) inference.

    Args:
        fs2_model:      Loaded fairseq2 model (already in eval mode).
        waveform_norm:  Normalised waveform tensor of shape (T,).

    Returns:
        Logits tensor (1, T_out, vocab_size) or None on failure.
    """
    from fairseq2.nn.batch_layout import BatchLayout  # type: ignore[import]

    seqs     = waveform_norm.unsqueeze(0)          # (1, T)
    seq_lens = torch.tensor([waveform_norm.shape[0]])

    batch_layout = BatchLayout(seqs.shape, seq_lens=seq_lens, device=seqs.device)

    try:
        with torch.no_grad():
            logits, _bl_out = fs2_model(seqs, batch_layout)
        return logits  # (1, T_out, vocab_size)
    except Exception as exc:
        logger.warning("[parity] fairseq2 inference failed: %s", exc)
        return None


def _run_fairseq2_ssl_inference(fs2_model, waveform_norm: Tensor) -> Optional[Tensor]:
    """Extract encoder hidden states from a fairseq2 SSL Wav2Vec2Model.

    Mirrors the first three steps of Wav2Vec2AsrModel.forward() (feature
    extraction → feature processing → transformer encoder) but stops before
    the CTC final_proj, returning the raw encoder output of shape
    (1, T_out, hidden_size).

    Args:
        fs2_model:      Loaded fairseq2 SSL model (eval mode).
        waveform_norm:  Normalised waveform tensor of shape (T,).

    Returns:
        Encoder hidden states (1, T_out, hidden_size) or None on failure.
    """
    from fairseq2.nn.batch_layout import BatchLayout  # type: ignore[import]

    seqs     = waveform_norm.unsqueeze(0)          # (1, T)
    seq_lens = torch.tensor([waveform_norm.shape[0]])
    layout   = BatchLayout(seqs.shape, seq_lens=seq_lens, device=seqs.device)

    try:
        with torch.no_grad():
            # Step 1: CNN feature extractor + layer norm
            seqs, layout, _ = fs2_model.encoder_frontend.extract_features(seqs, layout)
            # Step 2: feature projection + positional encoding (no masking at inference)
            seqs, _ = fs2_model.encoder_frontend.process_features(seqs, layout, None)
            # Step 3: transformer encoder
            encoder_out = fs2_model.encoder(seqs, layout)
        return encoder_out.unsqueeze(0) if encoder_out.dim() == 2 else encoder_out
    except Exception as exc:
        logger.warning("[parity-ssl] fairseq2 SSL inference failed: %s", exc)
        return None


def verify_parity_ssl(
    fs2_model,
    hf_model,
    audio_path: str,
    tag: str,
) -> bool:
    """Validate that the converted Wav2Vec2Model produces equivalent encoder outputs.

    Runs the same normalised audio through both models and checks:
      1. Output shape matches.
      2. Full numerical closeness (torch.allclose, atol=1e-4).
      3. First-order statistics (per-dimension mean) match closely.
      4. Second-order statistics (per-dimension std) match closely.

    Checks 3 and 4 are always reported; they give a useful signal even when
    minor floating-point drift causes check 2 to fail.

    Args:
        fs2_model:  Loaded fairseq2 SSL model (eval mode).
        hf_model:   Converted HuggingFace Wav2Vec2Model (eval mode).
        audio_path: Path to a 16 kHz WAV file used as the test utterance.
        tag:        Short model tag for log messages.

    Returns:
        True if all checks pass, False otherwise.
    """
    try:
        logger.info("[%s] SSL parity check: loading audio from %s", tag, audio_path)
        waveform      = _load_audio_16k(audio_path)
        waveform_norm = _normalize_waveform(waveform)

        # ── fairseq2 encoder output ───────────────────────────────────────────
        fs2_out = _run_fairseq2_ssl_inference(fs2_model, waveform_norm)
        if fs2_out is None:
            logger.warning(
                "[%s] SSL parity check: fairseq2 inference failed — skipping.", tag
            )
            return False

        # ── HF encoder output ─────────────────────────────────────────────────
        with torch.no_grad():
            hf_out = hf_model(
                input_values=waveform_norm.unsqueeze(0)
            ).last_hidden_state  # (1, T_out, hidden_size)

        # ── Check 1: shape ────────────────────────────────────────────────────
        if fs2_out.shape != hf_out.shape:
            logger.error(
                "[%s] SSL PARITY FAIL — shape mismatch: fairseq2=%s  HF=%s",
                tag, tuple(fs2_out.shape), tuple(hf_out.shape),
            )
            return False
        logger.info("[%s]   embedding shape: %s  ✓", tag, tuple(hf_out.shape))

        # ── Check 2: full numerical closeness ─────────────────────────────────
        max_diff     = (fs2_out - hf_out).abs().max().item()
        values_match = torch.allclose(fs2_out, hf_out, atol=1e-4, rtol=1e-3)
        if values_match:
            logger.info(
                "[%s]   embedding values: max_abs_diff=%.2e  ✓", tag, max_diff
            )
        else:
            logger.error(
                "[%s] SSL PARITY FAIL — embedding mismatch: max_abs_diff=%.2e  "
                "(tolerance atol=1e-4, rtol=1e-3)",
                tag, max_diff,
            )

        # ── Check 3 & 4: first and second order statistics ────────────────────
        # Computed over the time dimension → shape (hidden_size,)
        fs2_mean = fs2_out.squeeze(0).mean(dim=0)   # (hidden_size,)
        hf_mean  = hf_out.squeeze(0).mean(dim=0)
        fs2_std  = fs2_out.squeeze(0).std(dim=0)
        hf_std   = hf_out.squeeze(0).std(dim=0)

        mean_max_diff = (fs2_mean - hf_mean).abs().max().item()
        std_max_diff  = (fs2_std  - hf_std ).abs().max().item()
        stats_match   = (mean_max_diff < 1e-3) and (std_max_diff < 1e-3)

        logger.info(
            "[%s]   mean max_abs_diff=%.2e  std max_abs_diff=%.2e  %s",
            tag, mean_max_diff, std_max_diff,
            "✓" if stats_match else "FAIL",
        )
        if not stats_match:
            logger.error(
                "[%s] SSL PARITY FAIL — statistics mismatch "
                "(mean diff=%.2e, std diff=%.2e)",
                tag, mean_max_diff, std_max_diff,
            )

        passed = values_match and stats_match
        if passed:
            logger.info("[%s] SSL parity check PASSED.", tag)
        else:
            logger.error("[%s] SSL parity check FAILED.", tag)
        return passed

    except Exception as exc:
        logger.warning(
            "[%s] SSL parity check raised an unexpected error (%s) — "
            "not blocking conversion.",
            tag, exc,
        )
        return False


def verify_parity(
    fs2_model,
    hf_model,
    fairseq2_card: str,
    audio_path: str,
    blank_id: int,
    local_dir: str,
    tag: str,
) -> bool:
    """Validate that the converted HF model produces identical output to fairseq2.

    Runs the same normalised audio through both models and checks three things:
      1. Logit tensors are numerically close (torch.allclose, atol=1e-4).
      2. Greedy-argmax token ID sequences are identical.
      3. Decoded transcripts are identical strings (using the vocab.json written
         by build_hf_tokenizer to convert token IDs → text).

    All three checks are run regardless of earlier failures so the log provides
    a complete picture of any divergence.  The overall result (pass/fail) is
    returned; failures are logged as errors but do not raise exceptions — the
    batch run continues.

    NOTE — memory: both models are held in memory simultaneously during this
    check.  For 7B models on CPU this requires ~2 × model_size RAM.  Omit
    --smoke-test-audio to skip the check if memory is a constraint.

    Args:
        fs2_model:      Loaded fairseq2 model (eval mode).
        hf_model:       Freshly converted Wav2Vec2ForCTC (eval mode, CPU).
        fairseq2_card:  fairseq2 card name (used in log messages only).
        audio_path:     Path to a 16 kHz WAV file used as the test utterance.
        blank_id:       CTC blank token ID (0 for all OmniASR models).
        local_dir:      Path to the local HF checkpoint directory; used to
                        locate vocab.json for transcript decoding.
        tag:            Short model tag for log messages.

    Returns:
        True if all three checks pass, False if any check fails.
    """
    try:
        import torchaudio  # noqa: F401 (used indirectly by _load_audio_16k)

        logger.info("[%s] Parity check: loading audio from %s", tag, audio_path)
        waveform      = _load_audio_16k(audio_path)
        waveform_norm = _normalize_waveform(waveform)

        # ── fairseq2 forward pass ─────────────────────────────────────────────
        fs2_logits = _run_fairseq2_inference(fs2_model, waveform_norm)
        if fs2_logits is None:
            logger.warning(
                "[%s] Parity check: could not determine fairseq2 inference API — "
                "skipping parity validation.",
                tag,
            )
            return False

        # ── HF forward pass ───────────────────────────────────────────────────
        # Feed the already-normalised waveform directly so both models receive
        # the exact same floating-point input values.
        hf_input = waveform_norm.unsqueeze(0)  # (1, T)
        with torch.no_grad():
            hf_logits = hf_model(input_values=hf_input).logits  # (1, T_out, vocab_size)

        # ── Check 1: logit shape ──────────────────────────────────────────────
        if fs2_logits.shape != hf_logits.shape:
            logger.error(
                "[%s] PARITY FAIL — logit shape mismatch: fairseq2=%s  HF=%s",
                tag, tuple(fs2_logits.shape), tuple(hf_logits.shape),
            )
            return False
        logger.info("[%s]   logit shape: %s  ✓", tag, tuple(hf_logits.shape))

        # ── Check 2: numerical closeness of raw logits ────────────────────────
        # atol=1e-4 / rtol=1e-3 tolerates minor float32 rounding differences
        # from the two frameworks' implementations of the same ops.
        max_diff     = (fs2_logits - hf_logits).abs().max().item()
        logits_match = torch.allclose(fs2_logits, hf_logits, atol=1e-4, rtol=1e-3)
        if logits_match:
            logger.info("[%s]   logit values: max_abs_diff=%.2e  ✓", tag, max_diff)
        else:
            logger.error(
                "[%s] PARITY FAIL — logit mismatch: max_abs_diff=%.2e  "
                "(tolerance atol=1e-4, rtol=1e-3)",
                tag, max_diff,
            )
            # Continue to token/transcript checks even on logit mismatch —
            # the top token (greedy argmax) may still agree even if logit
            # magnitudes differ, which helps narrow down the root cause.

        # ── Check 3: greedy token IDs ─────────────────────────────────────────
        fs2_ids = _ctc_greedy_decode(fs2_logits, blank_id)
        hf_ids  = _ctc_greedy_decode(hf_logits,  blank_id)
        ids_match = (fs2_ids == hf_ids)
        if ids_match:
            logger.info("[%s]   greedy token IDs: match (%d tokens)  ✓", tag, len(hf_ids))
        else:
            # Report first divergence position to aid debugging
            n_common = sum(1 for a, b in zip(fs2_ids, hf_ids) if a == b)
            logger.error(
                "[%s] PARITY FAIL — token ID mismatch: "
                "fairseq2=%d tokens  HF=%d tokens  first %d tokens agree",
                tag, len(fs2_ids), len(hf_ids), n_common,
            )

        # ── Check 4: decoded transcript ───────────────────────────────────────
        # Decode both ID sequences to strings using vocab.json written by
        # build_hf_tokenizer, so the same token→string mapping is used for both.
        def _ids_to_str(token_ids: List[int]) -> str:
            """Map token IDs to a readable string via vocab.json."""
            vocab_path = Path(local_dir) / "vocab.json"
            if vocab_path.exists():
                with open(vocab_path, encoding="utf-8") as vf:
                    vocab = json.load(vf)
                id_to_token = {v: k for k, v in vocab.items()}
                tokens = [id_to_token.get(i, f"<{i}>") for i in token_ids]
                return "".join(tokens).replace("▁", " ").strip()
            # Fallback: space-separated integer IDs (useful if vocab.json is missing)
            return " ".join(str(i) for i in token_ids)

        fs2_text   = _ids_to_str(fs2_ids)
        hf_text    = _ids_to_str(hf_ids)
        text_match = (fs2_text == hf_text)

        logger.info("[%s]   fairseq2 transcript: '%s'", tag, fs2_text)
        logger.info("[%s]   HF      transcript: '%s'", tag, hf_text)
        if text_match:
            logger.info("[%s]   transcript: match  ✓", tag)
        else:
            logger.error("[%s] PARITY FAIL — transcript mismatch (see above).", tag)

        # ── Overall verdict ───────────────────────────────────────────────────
        passed = logits_match and ids_match and text_match
        if passed:
            logger.info(
                "[%s] Parity check PASSED — fairseq2 and HF outputs are identical.", tag
            )
        else:
            logger.error(
                "[%s] Parity check FAILED — inspect the log above for details. "
                "A weight-mapping error is the most likely cause.",
                tag,
            )
        return passed

    except Exception as exc:
        logger.warning(
            "[%s] Parity check raised an unexpected error (%s) — "
            "not blocking conversion.",
            tag, exc,
        )
        return False


def write_model_card(
    local_dir: str,
    tag: str,
    fairseq2_card: str,
    hf_repo_id: str,
    arch: Dict,
    model_type: str,
    parity_verified: Optional[bool],
) -> None:
    """Write a README.md model card to `local_dir`.

    Args:
        local_dir:        Local checkpoint directory.
        tag:              Short model tag (e.g. ``omniASR_CTC_300M``).
        fairseq2_card:    Original fairseq2 model card name.
        hf_repo_id:       Full HuggingFace repo ID (``user/repo-name``).
        arch:             Architecture dict from detect_arch / detect_arch_ssl.
        model_type:       ``"ctc"`` or ``"ssl"``.
        parity_verified:  True = passed, False = failed, None = not tested.
    """
    is_ctc = model_type == "ctc"

    # ── YAML front-matter ─────────────────────────────────────────────────────
    if is_ctc:
        pipeline_tag = "automatic-speech-recognition"
        extra_tags   = "- automatic-speech-recognition"
    else:
        pipeline_tag = "feature-extraction"
        extra_tags   = "- feature-extraction"

    if parity_verified is True:
        verified_badge = "✅ Verified"
        verified_note  = (
            "Numerical parity against the original fairseq2 checkpoint has been "
            "confirmed: outputs match to within `atol=1e-4` on a held-out audio sample."
        )
    elif parity_verified is False:
        verified_badge = "⚠️ Verification failed"
        verified_note  = (
            "A parity check was run but reported a mismatch. "
            "Use with caution and open an issue if you observe unexpected behaviour."
        )
    else:
        verified_badge = "—"
        verified_note  = "No parity check was run during conversion."


    size_label = tag.split("_")[-1]  # "300M", "1B", etc.

    if is_ctc:
        hf_class      = "Wav2Vec2ForCTC"
        description   = (
            f"Wav2Vec2 CTC ASR model ({size_label}) converted from the "
            f"[OmniLingual](https://github.com/facebookresearch/omnilingual-asr) "
            f"fairseq2 checkpoint `{fairseq2_card}`.\n\n"
            "This model outputs CTC logits over a SentencePiece vocabulary and "
            "can transcribe speech in multiple languages."
        )
        usage_snippet = f"""\
```python
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch, torchaudio

processor = AutoProcessor.from_pretrained("{hf_repo_id}")
model     = Wav2Vec2ForCTC.from_pretrained("{hf_repo_id}")
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16_000:
    waveform = torchaudio.functional.resample(waveform, sr, 16_000)

inputs = processor(
    waveform.squeeze().numpy(), sampling_rate=16_000, return_tensors="pt"
)
with torch.no_grad():
    logits = model(**inputs).logits          # (1, T, vocab)

pred_ids   = torch.argmax(logits, dim=-1)
transcript = processor.decode(pred_ids[0])
print(transcript)
```"""
    else:
        hf_class      = "Wav2Vec2Model"
        description   = (
            f"Wav2Vec2 SSL encoder ({size_label}) converted from the "
            f"[OmniLingual](https://github.com/facebookresearch/omnilingual-asr) "
            f"fairseq2 checkpoint `{fairseq2_card}`.\n\n"
            "This is the **pre-trained encoder backbone without a CTC head**, "
            "suitable for feature extraction, probing, and fine-tuning on "
            "downstream speech tasks."
        )
        usage_snippet = f"""\
```python
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torch, torchaudio

extractor = Wav2Vec2FeatureExtractor.from_pretrained("{hf_repo_id}")
model     = Wav2Vec2Model.from_pretrained("{hf_repo_id}")
model.eval()

waveform, sr = torchaudio.load("audio.wav")
if sr != 16_000:
    waveform = torchaudio.functional.resample(waveform, sr, 16_000)

inputs = extractor(
    waveform.squeeze().numpy(), sampling_rate=16_000,
    return_tensors="pt", padding=True
)
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state  # (1, T, {arch["hidden_size"]})
```"""

    # ── Architecture table ────────────────────────────────────────────────────
    arch_rows = [
        f"| HF class             | `{hf_class}` |",
        f"| Encoder layers       | {arch['n_layers']} |",
        f"| Hidden size          | {arch['hidden_size']} |",
        f"| Attention heads      | {arch['num_attention_heads']} |",
        f"| FFN intermediate     | {arch['intermediate_size']} |",
    ]
    if is_ctc and "vocab_size" in arch:
        arch_rows.append(f"| Vocabulary size      | {arch['vocab_size']} |")
    arch_rows += [
        f"| Source framework     | fairseq2 |",
        f"| Source card          | `{fairseq2_card}` |",
        f"| Parity verification  | {verified_badge} |",
    ]
    arch_table = "\n".join(arch_rows)

    readme = f"""\
---
library_name: transformers
tags:
- speech
- audio
- wav2vec2
{extra_tags}
pipeline_tag: {pipeline_tag}
---

# {tag.replace("_", "-")}

{description}

## Model details

| Property             | Value |
|---|---|
{arch_table}


{verified_note}

## Usage

{usage_snippet}
"""

    card_path = Path(local_dir) / "README.md"
    card_path.write_text(readme, encoding="utf-8")
    logger.info("[%s] Model card written to %s", tag, card_path)


def _write_ssl_preprocessor_config(output_dir: str) -> None:
    """Write a minimal preprocessor_config.json for a SSL Wav2Vec2Model.

    HF pipelines and downstream users expect this file to know the sampling
    rate and normalisation settings expected by the model.
    """
    config = {
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "feature_size": 1,
        "sampling_rate": 16000,
        "padding_value": 0.0,
        "do_normalize": True,
        "return_attention_mask": False,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Per-model conversion pipeline
# ──────────────────────────────────────────────────────────────────────────────

def convert_and_push_one(
    tag: str,
    fairseq2_card: str,
    hf_repo_name: str,
    output_root: str,
    device: torch.device,
    push_to_hub: bool,
    hf_user: Optional[str],
    smoke_test_audio: Optional[str],
    model_type: str = "ctc",
) -> None:
    """Run the full conversion pipeline for a single OmniASR model.

    Steps:
      1. Check whether the model was already converted (idempotent).
      2. Load the fairseq2 model + state dict.
      3. Auto-detect architecture parameters.
      4. Write tokenizer / preprocessor files  (CTC only; skipped for SSL).
      5. Convert weights to HF format.
      6. Parity check — decode test audio with BOTH models and compare outputs.
         (Runs here, before the fairseq2 model is freed, so both are in memory.)
         (Skipped for SSL models — no CTC decoder.)
      7. Free the fairseq2 model from memory.
      8. Save the HF checkpoint locally.
      9. Push to the HuggingFace Hub (if requested).

    Args:
        tag:              Short model identifier (used in log messages).
        fairseq2_card:    fairseq2 hub model card name.
        hf_repo_name:     Local output sub-directory name and HF repo suffix.
        output_root:      Root directory under which to save converted models.
        device:           Torch device for loading fairseq2 weights.
        push_to_hub:      Whether to push the converted checkpoint to HF Hub.
        hf_user:          HuggingFace username / organisation (required for push).
        smoke_test_audio: Optional path to a WAV file for parity validation.
        model_type:       ``"ctc"`` for CTC ASR models, ``"ssl"`` for SSL
                          pre-training models (no CTC head, no tokenizer).
    """
    local_dir = str(Path(output_root) / hf_repo_name)

    # ── Idempotency check ─────────────────────────────────────────────────────
    # Skip conversion if a valid safetensors checkpoint already exists locally.
    safetensors_path = Path(local_dir) / "model.safetensors"
    if safetensors_path.exists():
        logger.info("[%s] Already converted at %s — skipping.", tag, local_dir)
        # Still attempt the push in case a previous run converted but did not push.
        if push_to_hub:
            _push_checkpoint(local_dir, hf_user, hf_repo_name, tag)
        return

    logger.info("=" * 72)
    logger.info("[%s] Starting conversion  (fairseq2 card: %s)", tag, fairseq2_card)
    logger.info("=" * 72)

    # ── Step 1: Load fairseq2 model ───────────────────────────────────────────
    from fairseq2.models.hub import load_model as fs2_load

    logger.info("[%s] Loading fairseq2 model …", tag)
    fs2_model = fs2_load(fairseq2_card, device=device, dtype=torch.float32)
    fs2_model.eval()
    fs2_sd = fs2_model.state_dict()
    logger.info("[%s] fairseq2 state dict: %d keys", tag, len(fs2_sd))

    # ── Step 2: Auto-detect architecture ─────────────────────────────────────
    if model_type == "ssl":
        arch = detect_arch_ssl(fs2_sd)
    else:
        arch = detect_arch(fs2_sd)

    # ── Step 3: Write tokenizer files (CTC only) ──────────────────────────────
    if model_type == "ssl":
        blank_id = None
        logger.info("[%s] SSL model — skipping tokenizer (no CTC head).", tag)
        # Write a minimal preprocessor_config so HF knows the sampling rate.
        _write_ssl_preprocessor_config(local_dir)
    else:
        blank_id = build_hf_tokenizer(fairseq2_card, local_dir)

    # ── Step 4: Convert weights ───────────────────────────────────────────────
    logger.info("[%s] Converting weights to HF format …", tag)
    if model_type == "ssl":
        hf_model = convert_fairseq2_to_hf_ssl(fs2_sd, arch)
    else:
        hf_model = convert_fairseq2_to_hf(fs2_sd, arch, ctc_blank_token_id=blank_id)

    # ── Step 5: Parity check ──────────────────────────────────────────────────
    # IMPORTANT: this must run BEFORE del fs2_model so both models are in memory.
    # For large models (3B / 7B) this requires ~2× model RAM simultaneously.
    # Skip by omitting --smoke-test-audio if memory is a constraint.
    # SSL models are skipped — they have no CTC decoder to compare.
    parity_verified: Optional[bool] = None
    if model_type == "ssl":
        if smoke_test_audio and os.path.exists(smoke_test_audio):
            parity_verified = verify_parity_ssl(
                fs2_model=fs2_model,
                hf_model=hf_model,
                audio_path=smoke_test_audio,
                tag=tag,
            )
        else:
            logger.info(
                "[%s] Pass --smoke-test-audio <path.wav> to run SSL embedding "
                "parity validation.",
                tag,
            )
    elif smoke_test_audio and os.path.exists(smoke_test_audio):
        parity_verified = verify_parity(
            fs2_model=fs2_model,
            hf_model=hf_model,
            fairseq2_card=fairseq2_card,
            audio_path=smoke_test_audio,
            blank_id=blank_id,
            local_dir=local_dir,
            tag=tag,
        )
    else:
        logger.info(
            "[%s] Pass --smoke-test-audio <path.wav> to run parity validation "
            "(compares fairseq2 vs HF transcriptions on the same audio).",
            tag,
        )

    # ── Step 6: Free fairseq2 model ───────────────────────────────────────────
    # Release memory before the (potentially large) save_pretrained call.
    del fs2_model, fs2_sd
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Step 7: Save locally ──────────────────────────────────────────────────
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(local_dir)
    logger.info("[%s] HF checkpoint saved to %s", tag, local_dir)

    # ── Step 7b: Write model card ─────────────────────────────────────────────
    hf_repo_id = f"{hf_user}/{hf_repo_name}" if hf_user else hf_repo_name
    write_model_card(
        local_dir=local_dir,
        tag=tag,
        fairseq2_card=fairseq2_card,
        hf_repo_id=hf_repo_id,
        arch=arch,
        model_type=model_type,
        parity_verified=parity_verified,
    )

    # ── Step 8: Push to HF Hub ────────────────────────────────────────────────
    if push_to_hub:
        _push_checkpoint(local_dir, hf_user, hf_repo_name, tag)

    logger.info("[%s] Done.", tag)


def _push_checkpoint(
    local_dir: str,
    hf_user: Optional[str],
    hf_repo_name: str,
    tag: str,
) -> None:
    """Push a locally saved HF checkpoint to the HuggingFace Hub.

    Pushes all files in `local_dir` (model weights + tokenizer / preprocessor
    configs) to the repository `{hf_user}/{hf_repo_name}`.

    The repository is created automatically if it does not yet exist.

    Args:
        local_dir:     Path to the saved HF checkpoint directory.
        hf_user:       HuggingFace username or organisation.
        hf_repo_name:  HF repository name (without the user/org prefix).
        tag:           Short model tag used in log messages.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from huggingface_hub import HfApi

    if not hf_user:
        logger.error(
            "[%s] --hf-user is required for pushing to the Hub. Skipping push.", tag
        )
        return

    hf_repo_id = f"{hf_user}/{hf_repo_name}"
    logger.info("[%s] Pushing to HF Hub: %s …", tag, hf_repo_id)

    try:
        # Use the HF API to upload the entire local directory as a single commit.
        # This handles model weights, config, and all tokenizer files in one shot.
        api = HfApi()
        api.create_repo(repo_id=hf_repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=hf_repo_id,
            commit_message=f"Upload {hf_repo_name} converted from fairseq2",
        )
        logger.info("[%s] Successfully pushed to https://huggingface.co/%s", tag, hf_repo_id)
    except Exception as exc:
        logger.error("[%s] Push failed: %s", tag, exc)


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace authentication
# ──────────────────────────────────────────────────────────────────────────────

def authenticate_hf() -> None:
    """Authenticate with HuggingFace Hub using a token from the environment.

    Checks for the HF_TOKEN environment variable.  If found, logs in
    programmatically.  If not found, relies on a previously cached token
    (set via `huggingface-cli login`).  No error is raised if neither is
    present — an unauthenticated push will simply fail with a 401 from HF.
    """
    from huggingface_hub import login as hf_login

    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("Authenticating with HuggingFace Hub via HF_TOKEN env variable.")
        hf_login(token=token, add_to_git_credential=False)
    else:
        logger.info(
            "HF_TOKEN not set. Relying on cached credentials "
            "(run `huggingface-cli login` to set them)."
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all OmniASR fairseq2 checkpoints to HuggingFace Wav2Vec2ForCTC "
            "and optionally push them to the HuggingFace Hub."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Validate with one model before the full batch:
  export HF_TOKEN=hf_xxxx
  FAIRSEQ2_ASSETS_DIR=.../fairseq2_assets \\
    /path/to/omnilingual/python scripts/convert_all_omni_to_hf.py \\
    --hf-user YOUR_USERNAME --models omniASR_CTC_300M_v2 \\
    --smoke-test-audio test.wav --push-to-hub

# Full batch (all 8 models):
  FAIRSEQ2_ASSETS_DIR=.../fairseq2_assets \\
    /path/to/omnilingual/python scripts/convert_all_omni_to_hf.py \\
    --hf-user YOUR_USERNAME --push-to-hub
""",
    )

    parser.add_argument(
        "--hf-user",
        default=None,
        help="HuggingFace username or organisation to push models to. "
             "Required when --push-to-hub is set.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push each converted checkpoint to the HuggingFace Hub after saving locally.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        metavar="TAG",
        help=(
            "Space-separated list of model tags to process "
            f"(default: all {len(MODEL_REGISTRY)}). "
            f"Available tags: {', '.join(MODEL_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="converted_models",
        help="Root directory under which converted checkpoints are saved "
             "(default: converted_models/).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for loading fairseq2 weights (default: cpu). "
             "Use 'cuda' if you have a GPU with enough VRAM for the 7B model.",
    )
    parser.add_argument(
        "--smoke-test-audio",
        default=None,
        metavar="PATH",
        help="Path to a 16 kHz WAV file to run a quick decode test after each conversion.",
    )

    args = parser.parse_args()

    # Validate model tags
    unknown = [t for t in args.models if t not in MODEL_REGISTRY]
    if unknown:
        parser.error(
            f"Unknown model tag(s): {unknown}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Check --push-to-hub requires --hf-user
    if args.push_to_hub and not args.hf_user:
        parser.error("--push-to-hub requires --hf-user to be set.")

    device = torch.device(args.device)

    # Authenticate with HF Hub (reads HF_TOKEN env var or uses cached credentials)
    if args.push_to_hub:
        authenticate_hf()

    # ── Main conversion loop ──────────────────────────────────────────────────
    n_models  = len(args.models)
    successes = []
    failures  = []

    for idx, tag in enumerate(args.models, start=1):
        entry = MODEL_REGISTRY[tag]
        logger.info("\n[%d/%d] Processing model: %s", idx, n_models, tag)

        try:
            convert_and_push_one(
                tag=tag,
                fairseq2_card=entry["fairseq2_card"],
                hf_repo_name=entry["hf_repo_name"],
                output_root=args.output_root,
                device=device,
                push_to_hub=args.push_to_hub,
                hf_user=args.hf_user,
                smoke_test_audio=args.smoke_test_audio,
                model_type=entry.get("model_type", "ctc"),
            )
            successes.append(tag)
        except Exception as exc:
            # Log the error but continue with remaining models so a single
            # failure (e.g. a missing model card) does not abort the entire batch.
            logger.error("[%s] Conversion failed: %s", tag, exc, exc_info=True)
            failures.append(tag)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("BATCH COMPLETE")
    logger.info("  Succeeded (%d): %s", len(successes), successes)
    if failures:
        logger.error("  Failed    (%d): %s", len(failures), failures)
    logger.info("=" * 72)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert a Chris GPT-2 training checkpoint (.pt) to a Hugging Face
GPT2LMHeadModel directory (config + tokenizer + SafeTensors weights).

Designed for the Chris GPT-2 implementation used to train model_19072.pt:
- GPT-2 124M style architecture
- nn.Linear for c_attn/c_proj/c_fc layers
- tied token embedding / lm_head
- padded vocab_size=50304
- GPT-2 BPE tokenizer (50257 real tokens)

The important conversion detail is that Hugging Face GPT-2 uses Conv1D
for four projection matrices. Chris GPT-2 uses nn.Linear, therefore these
weights must be transposed during conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch


# IMPORTANT: model_19072.pt pickled GPTConfig as __main__.GPTConfig.
# Keeping this class at module top-level lets torch.load(..., weights_only=False)
# resolve the trusted checkpoint's serialized config when this script is run.
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


TRANSPOSED_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)

MASK_SUFFIXES = (
    ".attn.bias",
    ".attn.masked_bias",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Chris GPT-2 .pt checkpoint to Hugging Face GPT2LMHeadModel"
    )
    p.add_argument("checkpoint", type=Path, help="Path to model_19072.pt")
    p.add_argument("output_dir", type=Path, help="Destination Hugging Face model directory")
    p.add_argument(
        "--tokenizer-source",
        default="openai-community/gpt2",
        help="Tokenizer to copy into output_dir (default: openai-community/gpt2)",
    )
    p.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Do not download/copy tokenizer files (not recommended for publishing or GGUF conversion)",
    )
    p.add_argument(
        "--pytorch-bin",
        action="store_true",
        help="Save pytorch_model.bin instead of model.safetensors",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    p.add_argument(
        "--skip-reload-check",
        action="store_true",
        help="Skip final from_pretrained() reload verification",
    )
    return p.parse_args()


def normalize_key(key: str) -> str:
    # Robustness for DDP / torch.compile checkpoints.
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def get_cfg_value(cfg: Any, name: str, fallback: int) -> int:
    if cfg is None:
        return fallback
    if isinstance(cfg, dict):
        return int(cfg.get(name, fallback))
    return int(getattr(cfg, name, fallback))


def serializable_config(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    out = {}
    for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
        if hasattr(cfg, name):
            out[name] = getattr(cfg, name)
    return out


def load_chris_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    # This checkpoint is the user's own trusted training artifact. weights_only=False
    # is needed because GPTConfig was serialized as a Python dataclass under __main__.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions without the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'model'")
    if not isinstance(checkpoint["model"], dict):
        raise TypeError("checkpoint['model'] is not a state_dict-like dictionary")
    return checkpoint


def build_hf_config(checkpoint: dict[str, Any]):
    try:
        from transformers import GPT2Config
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install transformers safetensors") from exc

    cfg = checkpoint.get("config")
    block_size = get_cfg_value(cfg, "block_size", 1024)
    vocab_size = get_cfg_value(cfg, "vocab_size", 50304)
    n_layer = get_cfg_value(cfg, "n_layer", 12)
    n_head = get_cfg_value(cfg, "n_head", 12)
    n_embd = get_cfg_value(cfg, "n_embd", 768)

    if n_embd % n_head != 0:
        raise ValueError(f"n_embd={n_embd} is not divisible by n_head={n_head}")

    # Matches Chris GPT-2:
    # - nn.GELU(approximate='tanh') == GPT-2 gelu_new formulation
    # - no dropout in the Chris implementation
    # - LayerNorm default eps = 1e-5
    hf_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        use_cache=True,
        tie_word_embeddings=True,
    )
    hf_cfg.architectures = ["GPT2LMHeadModel"]
    return hf_cfg


def convert_state_dict(source_raw: dict[str, torch.Tensor], hf_model) -> dict[str, torch.Tensor]:
    source: dict[str, torch.Tensor] = {}
    for raw_key, tensor in source_raw.items():
        key = normalize_key(raw_key)
        if key in source:
            raise KeyError(f"Duplicate normalized key: {key}")
        source[key] = tensor

    # Chris GPT-2 persisted the causal mask buffer; Transformers does not need it
    # in the exported checkpoint.
    source_params = {
        k: v for k, v in source.items()
        if not k.endswith(MASK_SUFFIXES)
    }

    target_state = hf_model.state_dict()
    target_keys = set(target_state.keys())
    source_keys = set(source_params.keys())

    # Some Transformers versions may expose attention mask buffers in state_dict;
    # leave those initialized by HF. Every actual parameter must still be present.
    target_nonmask = {k for k in target_keys if not k.endswith(MASK_SUFFIXES)}

    missing = sorted(target_nonmask - source_keys)
    extra = sorted(source_keys - target_nonmask)
    if missing or extra:
        details = []
        if missing:
            details.append("Missing Chris -> HF keys:\n  " + "\n  ".join(missing))
        if extra:
            details.append("Unexpected Chris keys:\n  " + "\n  ".join(extra))
        raise RuntimeError("State-dict key mismatch.\n" + "\n".join(details))

    converted = dict(target_state)

    for key, src in source_params.items():
        dst = target_state[key]
        value = src.detach().cpu()

        if key.endswith(TRANSPOSED_SUFFIXES):
            if value.ndim != 2:
                raise RuntimeError(f"Expected 2D tensor for transpose: {key} {tuple(value.shape)}")
            value = value.t().contiguous()

        if tuple(value.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: Chris {tuple(src.shape)} -> converted {tuple(value.shape)}, "
                f"HF expects {tuple(dst.shape)}"
            )

        # Preserve checkpoint precision. GPT-2 training checkpoint is expected FP32.
        converted[key] = value.to(dtype=dst.dtype)

    return converted


def verify_mapping(source_raw: dict[str, torch.Tensor], hf_state: dict[str, torch.Tensor]) -> None:
    """Exact tensor-level round-trip check (Chris -> HF -> Chris layout)."""
    checked = 0
    for raw_key, src in source_raw.items():
        key = normalize_key(raw_key)
        if key.endswith(MASK_SUFFIXES):
            continue
        if key not in hf_state:
            raise RuntimeError(f"Verification target missing key: {key}")

        restored = hf_state[key].detach().cpu()
        if key.endswith(TRANSPOSED_SUFFIXES):
            restored = restored.t().contiguous()

        src_cpu = src.detach().cpu().to(restored.dtype)
        if not torch.equal(src_cpu, restored):
            max_diff = (src_cpu.float() - restored.float()).abs().max().item()
            raise RuntimeError(f"Tensor verification failed for {key}; max abs diff={max_diff}")
        checked += 1

    print(f"[ok] exact tensor mapping verified for {checked} parameter tensors")


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()

    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {out}\nUse --overwrite to continue.")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] loading checkpoint: {args.checkpoint}")
    checkpoint = load_chris_checkpoint(args.checkpoint)

    hf_cfg = build_hf_config(checkpoint)
    print(
        "[2/6] config: "
        f"block={hf_cfg.n_ctx}, vocab={hf_cfg.vocab_size}, layers={hf_cfg.n_layer}, "
        f"heads={hf_cfg.n_head}, embd={hf_cfg.n_embd}"
    )

    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install transformers safetensors") from exc

    model = GPT2LMHeadModel(hf_cfg)
    converted = convert_state_dict(checkpoint["model"], model)

    incompatible = model.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected load_state_dict result: {incompatible}")

    verify_mapping(checkpoint["model"], model.state_dict())

    print(f"[3/6] saving Hugging Face model to: {out}")
    model.eval()
    model.save_pretrained(
        out,
        safe_serialization=not args.pytorch_bin,
        max_shard_size="2GB",
    )

    if not args.no_tokenizer:
        print(f"[4/6] saving tokenizer from: {args.tokenizer_source}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_source, use_fast=True)
        tokenizer.save_pretrained(out)
    else:
        print("[4/6] tokenizer skipped")

    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "source_step": checkpoint.get("step"),
        "source_val_loss": checkpoint.get("val_loss"),
        "source_config": serializable_config(checkpoint.get("config")),
        "hf_architecture": "GPT2LMHeadModel",
        "transposed_weight_suffixes": list(TRANSPOSED_SUFFIXES),
        "dropped_checkpoint_buffers": ["*.attn.bias"],
        "notes": [
            "Chris GPT-2 uses nn.Linear; Hugging Face GPT-2 uses Conv1D for the listed projections.",
            "vocab_size may be 50304 while the GPT-2 tokenizer has 50257 real tokens; the extra rows are padding/alignment slots from training.",
        ],
    }
    (out / "chris_conversion.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.skip_reload_check:
        print("[5/6] reload check skipped")
    else:
        print("[5/6] reloading with GPT2LMHeadModel.from_pretrained()")
        reloaded = GPT2LMHeadModel.from_pretrained(out, local_files_only=True)
        reloaded.eval()

        # Verify a deterministic tiny forward pass works. Token IDs are deliberately
        # within GPT-2's real vocabulary range.
        probe = torch.tensor([[15496, 11, 314, 716]], dtype=torch.long)  # valid GPT-2 ids
        with torch.inference_mode():
            logits = reloaded(probe).logits
        expected_shape = (1, probe.shape[1], hf_cfg.vocab_size)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"Reload forward shape {tuple(logits.shape)} != {expected_shape}")
        if not torch.isfinite(logits).all():
            raise RuntimeError("Reloaded model produced non-finite logits")
        print(f"[ok] reload + forward pass: logits {tuple(logits.shape)}")

    weight_file = "pytorch_model.bin" if args.pytorch_bin else "model.safetensors"
    print("[6/6] conversion complete")
    print(f"model:     {out / weight_file}")
    print(f"config:    {out / 'config.json'}")
    if not args.no_tokenizer:
        print(f"tokenizer: {out}")
    print(f"metadata:  {out / 'chris_conversion.json'}")


if __name__ == "__main__":
    main()

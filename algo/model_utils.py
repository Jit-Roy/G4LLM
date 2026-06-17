"""
Model utilities for ROME-based editing.

Handles the differences between GPT-2, GPT-J, LLaMA/LLaMA-2/LLaMA-3,
Mistral, Phi Qwen Family and similar causal-LM architectures.

Key abstractions
----------------
* ``get_module``  -- retrieve any sub-module by dotted name
* ``get_weight`` / ``set_weight``  -- read / patch a weight tensor in-place
* ``nethook``     -- context-manager to intercept & patch activations mid-forward
* ``ModelConfig`` -- per-model-family constants (which layer to target, etc.)
"""

from __future__ import annotations

import functools
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# Per-model configuration
# ------------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Describes the MLP structure of one model family.

    Attributes
    ----------
    mlp_module_tmp:
        Format string for the *output projection* weight path.
        ``{}`` is replaced by the layer index.
        This is the W that ROME edits.
    layer_module_tmp:
        Format string for the *layer* module path (used to hook the residual
        stream at a given depth).
    n_layers:
        Total transformer depth.
    edit_layers:
        Default candidate layers for editing (the "causal" layers per ROME).
    """

    mlp_module_tmp: str
    layer_module_tmp: str
    n_layers: int
    edit_layers: List[int] = field(default_factory=list)

    def mlp_path(self, layer: int) -> str:
        return self.mlp_module_tmp.format(layer)

    def layer_path(self, layer: int) -> str:
        return self.layer_module_tmp.format(layer)


# Registry keyed by strings that appear in model.config.model_type or model name
_CONFIGS: Dict[str, ModelConfig] = {
    "gpt2": ModelConfig(
        mlp_module_tmp="transformer.h.{}.mlp.c_proj",
        layer_module_tmp="transformer.h.{}",
        n_layers=12,
        edit_layers=list(range(6, 10)),
    ),
    "gpt2-medium": ModelConfig(
        mlp_module_tmp="transformer.h.{}.mlp.c_proj",
        layer_module_tmp="transformer.h.{}",
        n_layers=24,
        edit_layers=list(range(10, 16)),
    ),
    "gpt2-large": ModelConfig(
        mlp_module_tmp="transformer.h.{}.mlp.c_proj",
        layer_module_tmp="transformer.h.{}",
        n_layers=36,
        edit_layers=list(range(16, 22)),
    ),
    "gpt2-xl": ModelConfig(
        mlp_module_tmp="transformer.h.{}.mlp.c_proj",
        layer_module_tmp="transformer.h.{}",
        n_layers=48,
        edit_layers=list(range(20, 28)),
    ),
    "gptj": ModelConfig(
        mlp_module_tmp="transformer.h.{}.mlp.fc_out",
        layer_module_tmp="transformer.h.{}",
        n_layers=28,
        edit_layers=list(range(3, 8)),
    ),
    "llama": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        n_layers=32,
        edit_layers=list(range(4, 8)),
    ),
    "mistral": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        n_layers=32,
        edit_layers=list(range(4, 8)),
    ),
    "phi": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.fc2",
        layer_module_tmp="model.layers.{}",
        n_layers=32,
        edit_layers=list(range(4, 8)),
    ),
    # Qwen / Qwen2 / Qwen3 family (LLaMA-style MLP with down_proj)
    # Qwen3-0.6B: 28 layers, hidden=1024, intermediate=3072
    "qwen": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        n_layers=28,
        edit_layers=list(range(8, 14)),
    ),
    "qwen2": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        n_layers=28,
        edit_layers=list(range(8, 14)),
    ),
    "qwen3": ModelConfig(
        mlp_module_tmp="model.layers.{}.mlp.down_proj",
        layer_module_tmp="model.layers.{}",
        n_layers=28,
        edit_layers=list(range(8, 14)),
    ),
}


def get_model_config(model: nn.Module) -> ModelConfig:
    """Auto-detect model config from ``model.config``."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no .config attribute")

    model_type = getattr(cfg, "model_type", "").lower()
    model_name = getattr(cfg, "_name_or_path", "").lower()

    for key, mcfg in _CONFIGS.items():
        if key in model_type or key in model_name:
            return mcfg

    # Heuristic fallback for LLaMA-family models
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        return ModelConfig(
            mlp_module_tmp="model.layers.{}.mlp.down_proj",
            layer_module_tmp="model.layers.{}",
            n_layers=n,
            edit_layers=list(range(n // 4, n // 4 + 4)),
        )

    # Fallback for GPT-2-family models
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        return ModelConfig(
            mlp_module_tmp="transformer.h.{}.mlp.c_proj",
            layer_module_tmp="transformer.h.{}",
            n_layers=n,
            edit_layers=list(range(n // 2, n // 2 + 4)),
        )

    raise ValueError(
        f"Unknown model architecture (model_type={model_type!r}). "
        "Register a ModelConfig in model_utils._CONFIGS."
    )


# ------------------------------------------------------------------------------
# Module / weight access
# ------------------------------------------------------------------------------

def get_module(model: nn.Module, path: str) -> nn.Module:
    """
    Retrieve a sub-module by dotted path, e.g.
    ``get_module(model, 'transformer.h.7.mlp.c_proj')``.
    """
    parts = path.split(".")
    m = model
    for part in parts:
        if part.isdigit():
            m = m[int(part)]
        else:
            m = getattr(m, part)
    return m


def get_weight(model: nn.Module, path: str) -> torch.Tensor:
    """Return the ``.weight`` tensor of the module at *path*."""
    return get_module(model, path).weight


def set_weight(model: nn.Module, path: str, W: torch.Tensor) -> None:
    """Replace the ``.weight`` of the module at *path* with *W* (in-place)."""
    mod = get_module(model, path)
    with torch.no_grad():
        mod.weight.copy_(W)


# ------------------------------------------------------------------------------
# Activation hooks
# ------------------------------------------------------------------------------

@contextmanager
def nethook(
    model: nn.Module,
    patch_spec: Dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> Generator[Dict[str, List[torch.Tensor]], None, None]:
    """
    Context manager for intercepting module outputs mid-forward-pass.

    Parameters
    ----------
    patch_spec:
        Mapping from dotted module path -> callable.
        The callable receives the module's output tensor and must return
        a (possibly modified) tensor.  Use ``lambda x: x`` to read without
        patching.

    Yields
    ------
    collected:
        Dict mapping path -> list of captured tensors (one per forward call).

    Example
    -------
    >>> with nethook(model, {"transformer.h.7": lambda x: x}) as caps:
    ...     model(**inputs)
    >>> hidden = caps["transformer.h.7"][0]
    """
    handles = []
    collected: Dict[str, List[torch.Tensor]] = {k: [] for k in patch_spec}

    def make_hook(path: str, fn: Callable):
        def hook(module, input, output):
            # output may be a tuple; we patch/collect only the first element
            if isinstance(output, tuple):
                patched = fn(output[0])
                collected[path].append(patched.detach())
                return (patched,) + output[1:]
            else:
                patched = fn(output)
                collected[path].append(patched.detach())
                return patched
        return hook

    for path, fn in patch_spec.items():
        mod = get_module(model, path)
        h = mod.register_forward_hook(make_hook(path, fn))
        handles.append(h)

    try:
        yield collected
    finally:
        for h in handles:
            h.remove()


# ------------------------------------------------------------------------------
# Tokenisation helpers
# ------------------------------------------------------------------------------

def find_token_range(
    tokenizer,
    prompt: str,
    substring: str,
) -> Tuple[int, int]:
    """
    Find the token-index range [start, end) for *substring* inside *prompt*.

    Returns the range of the *last* occurrence (for multi-token subjects
    ROME cares about the last token position).
    """
    prompt_ids = tokenizer.encode(prompt)
    sub_ids = tokenizer.encode(substring, add_special_tokens=False)

    for i in range(len(prompt_ids) - len(sub_ids), -1, -1):
        if prompt_ids[i : i + len(sub_ids)] == sub_ids:
            return i, i + len(sub_ids)

    # Fallback: treat entire sequence as subject
    return 0, len(prompt_ids)


def get_last_subject_token(
    tokenizer,
    prompt: str,
    subject: str,
) -> int:
    """Index of the last token of *subject* in *prompt*."""
    _, end = find_token_range(tokenizer, prompt, subject)
    return end - 1
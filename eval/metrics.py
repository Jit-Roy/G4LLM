"""
Evaluation metrics for G4LLM edits.

Three standard metrics from the model-editing literature:

Efficacy (E)
    Does the model now output the new target for the edited prompt?
    E = P(target_new | edited prompt) / P(target_old | edited prompt)

Generalization (G)
    Does the edit transfer to paraphrased prompts?
    G = mean E over subject_aliases.

Specificity (S)
    Are unrelated facts unchanged?
    S = 1 − mean |P_new(tok | control) − P_old(tok | control)| over control prompts.

All metrics ∈ [0, 1], higher is better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..objects.edit_request import EditRequest

logger = logging.getLogger(__name__)

# Default control prompts used for specificity measurement
_DEFAULT_CONTROL_PROMPTS = [
    "The capital of France is",
    "The speed of light is approximately",
    "Water is composed of",
    "The president of the United States is",
    "Mount Everest is located in",
    "Albert Einstein was born in",
]


@dataclass
class EditMetrics:
    """Container for all three evaluation metrics."""

    efficacy: float
    generalization: float
    specificity: float

    def __repr__(self) -> str:
        return (
            f"EditMetrics("
            f"eff={self.efficacy:.3f}, "
            f"gen={self.generalization:.3f}, "
            f"spec={self.specificity:.3f})"
        )

    def passed(self, threshold: float = 0.5) -> bool:
        return (
            self.efficacy >= threshold
            and self.generalization >= threshold
            and self.specificity >= threshold
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core probability utilities
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def token_prob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    token: str,
    device: torch.device,
) -> float:
    """
    Return P(token | prompt) from the model's next-token distribution.
    """
    tok_id = tokenizer.encode(" " + token, add_special_tokens=False)[-1]
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1]
    probs = F.softmax(logits.float(), dim=-1)
    return probs[tok_id].item()


@torch.no_grad()
def top_token(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    """Return the most-likely next token for *prompt*."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1]
    tok_id = logits.argmax().item()
    return tokenizer.decode([tok_id])


@torch.no_grad()
def full_distribution(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    """Return the full probability vector over the vocabulary."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_efficacy(
    model: torch.nn.Module,
    tokenizer,
    request: EditRequest,
    device: torch.device,
) -> float:
    """
    Efficacy: probability ratio P(target_new) / P(target_old).

    Returns a value in [0, ∞); 1.0 means no change.  Values > 1 indicate
    the edit succeeded.  Clipped to [0, 1] for reporting convenience.
    """
    prompt = request.to_prompt()
    p_new = token_prob(model, tokenizer, prompt, request.target_new, device)

    if request.target_old:
        p_old = token_prob(model, tokenizer, prompt, request.target_old, device)
        denom = max(p_old, 1e-8)
        return min(p_new / denom, 1.0)

    # No target_old: use absolute probability (normalised heuristic)
    return float(p_new)


def compute_generalization(
    model: torch.nn.Module,
    tokenizer,
    request: EditRequest,
    device: torch.device,
) -> float:
    """
    Generalisation: average efficacy across subject paraphrases.

    Uses ``request.subject_aliases`` if provided, else returns efficacy
    on the canonical prompt.
    """
    prompts = []
    if request.subject_aliases:
        rel = request.relation
        prompts = [f"{alias} {rel}" for alias in request.subject_aliases]
    else:
        prompts = [request.to_prompt()]

    scores = []
    for p in prompts:
        alt_req = EditRequest(
            subject=request.subject,
            relation=request.relation,
            target_new=request.target_new,
            target_old=request.target_old,
            prompt=p,
        )
        scores.append(compute_efficacy(model, tokenizer, alt_req, device))

    return float(sum(scores) / len(scores))


def compute_specificity(
    model_before,
    model_after,
    tokenizer,
    device: torch.device,
    control_prompts: Optional[List[str]] = None,
) -> float:
    """
    Specificity: how much did unrelated predictions change?

    Requires access to the model *before* the edit (or a reference copy).
    Returns 1.0 if nothing changed on control prompts, 0.0 if completely
    different.

    Parameters
    ----------
    model_before:
        The original model (or a snapshot of it before the edit).
    model_after:
        The edited model.
    control_prompts:
        Sentences unrelated to the edit.  Defaults to a small built-in set.
    """
    if control_prompts is None:
        control_prompts = _DEFAULT_CONTROL_PROMPTS

    total_drift = 0.0
    for prompt in control_prompts:
        p_before = full_distribution(model_before, tokenizer, prompt, device)
        p_after = full_distribution(model_after, tokenizer, prompt, device)
        drift = (p_before - p_after).abs().mean().item()
        total_drift += drift

    mean_drift = total_drift / len(control_prompts)
    return float(max(0.0, 1.0 - mean_drift * 100))


def evaluate_edit(
    model_before,
    model_after,
    tokenizer,
    request: EditRequest,
    device: torch.device,
    control_prompts: Optional[List[str]] = None,
) -> EditMetrics:
    """
    Compute all three metrics for one edit.

    Parameters
    ----------
    model_before:
        Model state *before* the edit.
    model_after:
        Model state *after* the edit.
    """
    eff = compute_efficacy(model_after, tokenizer, request, device)
    gen = compute_generalization(model_after, tokenizer, request, device)
    spec = compute_specificity(
        model_before, model_after, tokenizer, device, control_prompts
    )
    return EditMetrics(efficacy=eff, generalization=gen, specificity=spec)
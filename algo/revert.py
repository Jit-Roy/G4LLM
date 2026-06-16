"""
Edit reversal for G4LLM.

Based on ideas from:
    "Tracing and Reversing Rank-One Model Edits" (2025)
    https://arxiv.org/abs/2505.20819

Two complementary reversal strategies are provided:

1. **Exact reversal** (``exact_revert``):
   Subtract the stored outer(u, v) from the weight matrix.
   This perfectly undoes the edit — valid as long as the weight hasn't
   been modified by any other edit since.

2. **Approximate reversal** (``trace_and_revert``):
   When subsequent edits may have altered the same matrix, use gradient-
   based tracing to find the edit's contribution and subtract it.
   Less precise but more robust to sequential editing.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from ..objects.commit import Commit
from ..objects.delta import WeightDelta
from .model_utils import get_weight, set_weight

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Exact reversal
# ──────────────────────────────────────────────────────────────────────────────

def exact_revert(model: torch.nn.Module, commit: Commit) -> None:
    """
    Exactly undo *commit* by subtracting each stored delta.

    This is valid when:
      (a) no other edit has modified the same weight matrix since *commit*, or
      (b) the same matrix has been edited by commutative (orthogonal) rank-1
          updates (which is approximately true in practice).

    Raises
    ------
    ValueError
        If a delta references a layer that cannot be found in the model.
    """
    logger.info("Exact revert of commit %s", commit.short_hash())
    for delta in commit.deltas:
        W = get_weight(model, delta.layer_name)
        W_new = delta.revert(W)
        set_weight(model, delta.layer_name, W_new)
        logger.debug("  Reverted %s  ‖Δ‖_F=%.4f", delta.layer_name, delta.frobenius_norm())


# ──────────────────────────────────────────────────────────────────────────────
# Approximate reversal (trace-and-revert)
# ──────────────────────────────────────────────────────────────────────────────

def trace_and_revert(
    model: torch.nn.Module,
    commit: Commit,
    tokenizer,
    n_steps: int = 30,
    lr: float = 5e-2,
) -> List[WeightDelta]:
    """
    Approximate reversal that is robust to subsequent edits.

    For each delta (u₀, v₀) stored in the commit:
        1. Compute the current "contamination": Δ_current = outer(u₀, v₀)
        2. Use gradient descent to find α ∈ [0, 1] such that
           W_effective = W - α · Δ_current best restores the original output
           on the subject's prompt.
        3. Apply W - α · Δ_current.

    Returns the *effective* deltas that were subtracted (useful for bookkeeping).
    """
    logger.info(
        "Trace-and-revert of commit %s (approx.)", commit.short_hash()
    )
    req = commit.edit_request
    device = next(model.parameters()).device

    if tokenizer is None:
        logger.warning(
            "No tokenizer supplied to trace_and_revert; falling back to exact_revert."
        )
        exact_revert(model, commit)
        return commit.deltas

    prompt = req.to_prompt()
    target_old = req.target_old or req.target_new  # best-effort
    target_id = tokenizer.encode(
        " " + target_old, add_special_tokens=False
    )[-1]

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    effective_deltas: List[WeightDelta] = []

    for delta in commit.deltas:
        W = get_weight(model, delta.layer_name).float()
        u0 = torch.tensor(delta.u, dtype=torch.float32, device=device)
        v0 = torch.tensor(delta.v, dtype=torch.float32, device=device)

        alpha = torch.tensor(1.0, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([alpha], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()

            # Temporarily apply W - alpha * outer(u0, v0)
            W_trial = W - alpha.clamp(0, 1) * torch.outer(u0, v0)

            # Patch the weight, run forward, restore
            with torch.no_grad():
                orig = get_weight(model, delta.layer_name).data.clone()
                set_weight(model, delta.layer_name, W_trial.detach())

            logits = model(**enc).logits[0, -1].float()

            with torch.no_grad():
                set_weight(model, delta.layer_name, orig)

            loss = -torch.log_softmax(logits, dim=-1)[target_id]
            loss.backward()
            optimizer.step()

        # Apply the traced reversal
        alpha_val = float(alpha.clamp(0, 1).item())
        u_eff = (alpha_val * delta.u).astype(np.float32)
        v_eff = delta.v.astype(np.float32)

        W_np = W.cpu().numpy()
        W_new = W_np - np.outer(u_eff, v_eff)
        set_weight(model, delta.layer_name, torch.tensor(W_new, dtype=W.dtype))

        eff = WeightDelta(layer_name=delta.layer_name, u=u_eff, v=v_eff)
        effective_deltas.append(eff)
        logger.debug(
            "  Layer %s  α=%.3f  ‖Δ‖_F=%.4f",
            delta.layer_name,
            alpha_val,
            eff.frobenius_norm(),
        )

    return effective_deltas


# ──────────────────────────────────────────────────────────────────────────────
# Revert a chain of commits (sequential undo)
# ──────────────────────────────────────────────────────────────────────────────

def revert_chain(
    model: torch.nn.Module,
    commits: List[Commit],
    method: str = "exact",
    tokenizer=None,
) -> None:
    """
    Revert a list of commits in *reverse* order (LIFO — like ``git revert``).

    Parameters
    ----------
    commits:
        Commits to undo, in *chronological* order.  They will be reverted
        newest-first.
    method:
        ``'exact'`` or ``'trace'``.
    """
    for commit in reversed(commits):
        if method == "exact":
            exact_revert(model, commit)
        else:
            trace_and_revert(model, commit, tokenizer)
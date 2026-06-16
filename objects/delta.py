"""
WeightDelta: a rank-1 update to a single weight matrix.

This is G4LLM's equivalent of a Git *diff hunk* — the minimal,
reversible change to model weights that implements one knowledge edit.

ROME represents each edit as:
    W_new = W_old + outer(u, v)

where
    u = (v* - W_old @ k*) / (k̂ · k*)        shape: [d_out]
    v = C⁻¹ @ k*                              shape: [d_in]
    k* = MLP intermediate activation for subject
    v* = optimised target value vector
    C  = empirical covariance of MLP activations
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class WeightDelta:
    """
    A reversible rank-1 update to the weight matrix of one layer.

    Parameters
    ----------
    layer_name:
        Dotted attribute path to the weight tensor in the model, e.g.
        ``'transformer.h.7.mlp.c_proj'``.
    u:
        Left (output) vector, shape ``[d_out]``.
    v:
        Right (input) vector, shape ``[d_in]``.
    """

    layer_name: str
    u: np.ndarray
    v: np.ndarray

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def apply(self, W: "torch.Tensor") -> "torch.Tensor":
        """Return *W + outer(u, v)* — the patched weight matrix."""
        import torch
        u_t = torch.tensor(self.u, dtype=W.dtype, device=W.device)
        v_t = torch.tensor(self.v, dtype=W.dtype, device=W.device)
        return W + torch.outer(u_t, v_t)

    def revert(self, W: "torch.Tensor") -> "torch.Tensor":
        """Return *W − outer(u, v)* — undo the patch exactly."""
        import torch
        u_t = torch.tensor(self.u, dtype=W.dtype, device=W.device)
        v_t = torch.tensor(self.v, dtype=W.dtype, device=W.device)
        return W - torch.outer(u_t, v_t)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def frobenius_norm(self) -> float:
        """‖outer(u, v)‖_F = ‖u‖ · ‖v‖"""
        return float(np.linalg.norm(self.u) * np.linalg.norm(self.v))

    def rank(self) -> int:
        """Always 1 for a ROME edit."""
        return 1

    def summary(self) -> str:
        return (
            f"WeightDelta(layer={self.layer_name!r}, "
            f"‖Δ‖_F={self.frobenius_norm():.4f}, "
            f"shape=[{len(self.u)}, {len(self.v)}])"
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "layer_name": self.layer_name,
            "u": self.u.tolist(),
            "v": self.v.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightDelta":
        return cls(
            layer_name=d["layer_name"],
            u=np.array(d["u"], dtype=np.float32),
            v=np.array(d["v"], dtype=np.float32),
        )

    def __repr__(self) -> str:
        return self.summary()
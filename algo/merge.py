"""
Task Arithmetic Merging with Magnitude Sparsification
======================================================

Implementation of the merging strategy described in:
    Ilharco et al., "Editing Models with Task Arithmetic"
    ICLR 2023.  https://arxiv.org/abs/2212.04089

With sparsification as used in MM4KE:
    "sparsification_method=SparsificationMethod.magnitude"
    -- similar to TIES but WITHOUT the sign election step.

How it works
------------
1.  Compute the *task vector* for each fine-tuned model:
        tau_i = theta_finetuned_i - theta_base

2.  (Optional) Sparsify each task vector by magnitude:
        keep the top (1 - sparsity) fraction of weights by absolute value,
        zero out the rest.
        This reduces interference between task vectors during merging.

3.  Merge into one set of weights:
        theta_merged = theta_base + scaling_factor * sum(tau_i)

The result is a model that combines the capabilities of all fine-tuned
models without any additional training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """
    Configuration for a Task Arithmetic merge.

    Attributes
    ----------
    scaling_factor:
        Lambda that scales the sum of task vectors before adding to base.
        Lower values (0.3-0.5) are safer; higher (0.7-1.0) give stronger
        transfer but risk losing base capabilities.
    sparsity:
        Fraction of task-vector weights to *zero out* (by smallest magnitude).
        0.0 = keep everything (no sparsification).
        0.9 = keep only the top 10% by absolute value (MM4KE-style).
        Recommended: 0.8-0.9 for multiple model merges.
    method:
        ``'task_arithmetic'`` -- plain weighted sum of task vectors.
        ``'ties'``            -- task arithmetic + sign election (not yet impl).
    """
    scaling_factor: float = 0.5
    sparsity: float = 0.0
    method: str = "task_arithmetic"


# ------------------------------------------------------------------------------
# Main merger
# ------------------------------------------------------------------------------

class ModelMerger:
    """
    Merges multiple fine-tuned models into a single model using Task Arithmetic.

    Parameters
    ----------
    base_model:
        The original pre-trained model (theta_base).
        Its weights are used as the reference point.
        NOT modified in-place; used read-only.

    Example
    -------
    >>> config = MergeConfig(scaling_factor=0.5, sparsity=0.9)
    >>> merger = ModelMerger(base_model)
    >>> merger.add(finetuned_A)
    >>> merger.add(finetuned_B)
    >>> merged = merger.merge(config)   # new state_dict applied to base_model
    """

    def __init__(self, base_model: nn.Module) -> None:
        self.base_model = base_model
        self._finetuned: List[nn.Module] = []

    def add(self, finetuned_model: nn.Module) -> None:
        """Register a fine-tuned model to include in the merge."""
        self._finetuned.append(finetuned_model)
        logger.info("Added model %d to merge queue", len(self._finetuned))

    def merge(
        self,
        config: Optional[MergeConfig] = None,
        target_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """
        Compute the merged weights and apply them to *target_model*
        (or to a copy of base_model if target_model is None).

        Parameters
        ----------
        config:
            Merge hyper-parameters.
        target_model:
            Model to write merged weights into.  If None, weights are applied
            to the base_model in-place and it is returned.

        Returns
        -------
        nn.Module:
            The model with merged weights applied.
        """
        if not self._finetuned:
            raise ValueError(
                "No fine-tuned models added. Call merger.add(model) first."
            )

        cfg = config or MergeConfig()
        target = target_model if target_model is not None else self.base_model

        logger.info(
            "Merging %d model(s) | method=%s  sparsity=%.2f  scale=%.2f",
            len(self._finetuned),
            cfg.method,
            cfg.sparsity,
            cfg.scaling_factor,
        )

        # Compute task vectors
        task_vectors = self._compute_task_vectors()

        # Sparsify
        if cfg.sparsity > 0.0:
            task_vectors = [
                self._sparsify_magnitude(tv, cfg.sparsity)
                for tv in task_vectors
            ]
            logger.info(
                "Sparsified %d task vector(s) at sparsity=%.2f",
                len(task_vectors),
                cfg.sparsity,
            )

        # Sum all task vectors
        merged_tv = self._sum_task_vectors(task_vectors)

        # Apply: theta_merged = theta_base + lambda * merged_task_vector
        self._apply_task_vector(target, merged_tv, cfg.scaling_factor)

        logger.info("Merge complete.")
        return target

    # --------------------------------------------------------------------------
    # Core maths
    # --------------------------------------------------------------------------

    def _compute_task_vectors(self) -> List[Dict[str, torch.Tensor]]:
        """
        Compute tau_i = theta_finetuned_i - theta_base for each fine-tuned model.

        Returns a list of state-dict diffs (only parameters, not buffers).
        """
        base_sd = {
            k: v.detach().cpu().float()
            for k, v in self.base_model.named_parameters()
        }
        task_vectors: List[Dict[str, torch.Tensor]] = []

        for i, ft_model in enumerate(self._finetuned):
            ft_sd = {
                k: v.detach().cpu().float()
                for k, v in ft_model.named_parameters()
            }
            tv: Dict[str, torch.Tensor] = {}
            for name, base_w in base_sd.items():
                if name in ft_sd and ft_sd[name].shape == base_w.shape:
                    tv[name] = ft_sd[name] - base_w
                else:
                    logger.warning(
                        "Parameter %r missing or shape mismatch in model %d; skipping.",
                        name, i,
                    )
            task_vectors.append(tv)
            total_params = sum(t.numel() for t in tv.values())
            logger.info(
                "  Task vector %d: %d parameters, ||tau||_F = %.4f",
                i + 1,
                total_params,
                sum(t.norm().item() ** 2 for t in tv.values()) ** 0.5,
            )

        return task_vectors

    @staticmethod
    def _sparsify_magnitude(
        task_vector: Dict[str, torch.Tensor],
        sparsity: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Zero out the smallest-magnitude weights in the task vector.

        This is the MM4KE sparsification strategy:
            SparsificationMethod.magnitude
        It keeps only the top (1 - sparsity) fraction of weights by
        absolute value and zeros the rest, reducing cross-model interference.

        Parameters
        ----------
        task_vector:
            Dict mapping parameter name -> weight-diff tensor.
        sparsity:
            Fraction to zero out. 0.9 means keep top 10%.
        """
        if sparsity <= 0.0:
            return task_vector
        if sparsity >= 1.0:
            return {k: torch.zeros_like(v) for k, v in task_vector.items()}

        # Flatten all weights into one vector to find the global threshold
        all_weights = torch.cat([v.abs().flatten() for v in task_vector.values()])
        k = max(1, int((1.0 - sparsity) * all_weights.numel()))
        threshold = torch.topk(all_weights, k).values.min()

        sparsified = {}
        for name, w in task_vector.items():
            mask = w.abs() >= threshold
            sparsified[name] = w * mask.float()

        kept = sum((v != 0).sum().item() for v in sparsified.values())
        total = sum(v.numel() for v in sparsified.values())
        logger.debug(
            "  Sparsified: kept %d / %d weights (%.1f%%)",
            kept, total, 100.0 * kept / max(total, 1),
        )
        return sparsified

    @staticmethod
    def _sum_task_vectors(
        task_vectors: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Element-wise sum of all task vectors."""
        merged: Dict[str, torch.Tensor] = {}
        for tv in task_vectors:
            for name, delta in tv.items():
                if name in merged:
                    merged[name] = merged[name] + delta
                else:
                    merged[name] = delta.clone()
        return merged

    @staticmethod
    def _apply_task_vector(
        model: nn.Module,
        task_vector: Dict[str, torch.Tensor],
        scaling_factor: float,
    ) -> None:
        """
        Apply the merged task vector to *model* in-place:
            theta_merged = theta_base + scaling_factor * task_vector
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in task_vector:
                    delta = task_vector[name].to(param.device, param.dtype)
                    param.add_(scaling_factor * delta)

    # --------------------------------------------------------------------------
    # Inspection helpers
    # --------------------------------------------------------------------------

    def task_vector_stats(self) -> str:
        """
        Return a human-readable summary of each task vector's magnitude.
        Useful for understanding how much each fine-tuned model differs from base.
        """
        tvs = self._compute_task_vectors()
        lines = ["Task vector statistics:"]
        for i, tv in enumerate(tvs):
            frob = sum(t.norm().item() ** 2 for t in tv.values()) ** 0.5
            n_params = sum(t.numel() for t in tv.values())
            lines.append(
                f"  Model {i + 1}: ||tau||_F = {frob:.4f}  ({n_params:,} params)"
            )
        return "\n".join(lines)

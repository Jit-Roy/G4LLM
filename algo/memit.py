"""
MEMIT -- Mass-Editing Memory in a Transformer
=============================================

Implementation of the algorithm described in:
    Meng et al., "Mass-Editing Memory in a Transformer"
    ICLR 2023.  https://arxiv.org/abs/2210.07229

MEMIT extends ROME to edit *multiple* facts simultaneously by distributing
the updates across several layers rather than concentrating all edits
at a single layer.  This avoids destructive interference between edits.

Algorithm
---------
For a batch of n edits {(k_i*, v_i*)}, MEMIT finds weight updates
{DeltaW_l} for layers L = {l_1, ..., l_m} that jointly satisfy:

    W_l + DeltaW_l) K_l ~= V_l    for all l in L

where K_l and V_l are the stacked key and (residual) value matrices.

Each DeltaW_l is a low-rank matrix: in the simplest form a sum of rank-1
updates, one per edit, spread across layers.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch

from ..objects.edit_request import EditRequest
from ..objects.delta import WeightDelta
from .rome import ROMEEditor, ROMEConfig
from .model_utils import get_weight, set_weight

logger = logging.getLogger(__name__)


class MEMITEditor(ROMEEditor):
    """
    Edits multiple factual associations in one pass.

    Inherits from :class:`ROMEEditor` for key/value computation and
    extends it with multi-edit, multi-layer distribution.

    Parameters
    ----------
    model, tokenizer, config:
        Same as :class:`ROMEEditor`.
    memit_layers:
        Which layers to spread the edits across.
        Defaults to the model's default ``edit_layers`` range.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: Optional[ROMEConfig] = None,
        memit_layers: Optional[List[int]] = None,
    ) -> None:
        super().__init__(model, tokenizer, config)
        self.memit_layers: List[int] = (
            memit_layers or self.model_cfg.edit_layers
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_edit(
        self,
        requests: List[EditRequest],
    ) -> List[WeightDelta]:
        """
        Apply *n* edits simultaneously across *self.memit_layers*.

        Returns all WeightDelta objects produced (one per layer).
        The model is modified in-place.
        """
        if not requests:
            return []

        logger.info(
            "MEMIT batch edit: %d edits across layers %s",
            len(requests),
            self.memit_layers,
        )

        n = len(requests)
        all_deltas: List[WeightDelta] = []

        # Step 1: compute key and value vectors for every request
        # and for every target layer
        ks: List[np.ndarray] = []   # [n, d_inner]
        vs: List[np.ndarray] = []   # [n, d_model]

        for req in requests:
            # Use the *first* MEMIT layer for key extraction (as per paper)
            layer = self.memit_layers[0]
            k = self._compute_key(req, layer)
            v = self._compute_value(req, layer, k)
            ks.append(k)
            vs.append(v)

        K = np.stack(ks, axis=1)   # [d_inner, n]
        V = np.stack(vs, axis=1)   # [d_model, n]

        n_layers = len(self.memit_layers)

        # Step 2: distribute the updates evenly across MEMIT layers
        # Residual carried forward between layers
        V_residual = V.copy()

        for idx, layer in enumerate(self.memit_layers):
            mlp_path = self.model_cfg.mlp_path(layer)
            W = get_weight(self.model, mlp_path).detach().float().cpu().numpy()
            # W: [d_model, d_inner]

            # How much of the residual to resolve at this layer
            # (linearly decreasing contribution -- simplest MEMIT schedule)
            alpha = 1.0 / (n_layers - idx)
            V_layer = V_residual * alpha     # [d_model, n]

            # Per-edit rank-1 updates for this layer
            layer_deltas: List[WeightDelta] = []
            for i in range(n):
                k_i = K[:, i]                       # [d_inner]
                v_i = V_layer[:, i]                  # [d_model]

                C_inv_k = self._get_cov_inv_times_k(layer, k_i)
                W_k = W @ k_i
                residual_i = v_i - W_k
                scale = float(np.dot(C_inv_k, k_i))
                if abs(scale) < 1e-9:
                    scale = 1e-9

                u_i = residual_i / scale
                delta = WeightDelta(
                    layer_name=mlp_path,
                    u=u_i.astype(np.float32),
                    v=C_inv_k.astype(np.float32),
                )
                layer_deltas.append(delta)

            # Apply all deltas for this layer in one shot
            W_new = torch.tensor(W, dtype=torch.float32)
            for d in layer_deltas:
                W_new = d.apply(W_new)
            set_weight(self.model, mlp_path, W_new)
            all_deltas.extend(layer_deltas)

            # Update residual: subtract what this layer resolved
            for i in range(n):
                k_i = K[:, i]
                W_np = W_new.numpy()
                V_residual[:, i] -= W_np @ k_i - (W @ k_i)

            logger.debug(
                "  Layer %d: applied %d rank-1 updates", layer, n
            )

        return all_deltas
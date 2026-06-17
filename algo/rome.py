"""
ROME -- Rank-One Model Editing
==============================

Implementation of the algorithm described in:
    Meng et al., "Locating and Editing Factual Associations in GPT"
    NeurIPS 2022.  https://arxiv.org/abs/2202.05262

Mathematical summary
--------------------
Given a target MLP output-projection W in R^{d_model x d_inner}:

1.  Compute the key vector k* in R^{d_inner}
    the MLP's intermediate activation (after the activation function)
    when the prompt containing the subject is fed through the model.

2.  Optimise v* in R^{d_model} by gradient descent so that, when
    inserted into the residual stream at the target layer, the model
    outputs the desired target token with high probability.

3.  Load (or estimate) the covariance matrix C = E[k kT] in R^{d_innerxd_inner}
    of MLP intermediate activations over a reference corpus.

4.  Compute the rank-1 update:
        k = C-1 k*
        u = (v* - W k*) / (k * k*)      # value direction  [d_model]
        v = k                              # key   direction  [d_inner]
        W_new = W_old + outer(u, v)

The WeightDelta returned stores (u, v) so the update can be applied
and later *exactly reverted* by subtracting outer(u, v).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..objects.edit_request import EditRequest
from ..objects.delta import WeightDelta
from .model_utils import (
    ModelConfig,
    get_model_config,
    get_module,
    get_weight,
    set_weight,
    nethook,
    get_last_subject_token,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ROME configuration
# ------------------------------------------------------------------------------

class ROMEConfig:
    """Hyper-parameters for one ROME run."""

    def __init__(
        self,
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 25,
        v_weight_decay: float = 1e-3,
        v_loss_layer: Optional[int] = None,
        clamp_norm_factor: float = 4.0,
        kl_factor: float = 0.0625,
        mom2_adjustment: bool = True,
        mom2_dataset: str = "wikitext",
        mom2_n_samples: int = 100_000,
        mom2_dtype: str = "float32",
        edit_layer: Optional[int] = None,
        epochs: int = 1,
    ) -> None:
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.v_loss_layer = v_loss_layer          # layer at which to measure loss
        self.clamp_norm_factor = clamp_norm_factor
        self.kl_factor = kl_factor
        self.mom2_adjustment = mom2_adjustment
        self.mom2_dataset = mom2_dataset
        self.mom2_n_samples = mom2_n_samples
        self.mom2_dtype = mom2_dtype
        self.edit_layer = edit_layer              # None -> use model default
        self.epochs = epochs                      # outer epoch loop (MM4KE uses 5)


# ------------------------------------------------------------------------------
# Main ROME editor
# ------------------------------------------------------------------------------

class ROMEEditor:
    """
    Applies ROME edits to a HuggingFace causal-LM.

    Parameters
    ----------
    model:
        The causal LM (e.g. ``AutoModelForCausalLM.from_pretrained(...)``).
    tokenizer:
        Matching tokenizer with ``padding_side = 'right'``.
    config:
        ROME hyper-parameters.  Defaults are suitable for GPT-2-scale models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        config: Optional[ROMEConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config or ROMEConfig()
        self.model_cfg: ModelConfig = get_model_config(model)
        self.device = next(model.parameters()).device

        # Cached covariance matrices: layer_path -> torch.Tensor [d_inner, d_inner]
        self._cov_cache: dict = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def edit(
        self,
        request: EditRequest,
        layer: Optional[int] = None,
    ) -> List[WeightDelta]:
        """
        Apply one ROME edit and return the list of :class:`WeightDelta` objects.

        The model is modified *in-place*.  The returned deltas can be used
        to revert the edit via :meth:`revert`.

        Parameters
        ----------
        request:
            The factual association to change.
        layer:
            Which transformer layer to target.  Defaults to the model
            family's canonical edit layer.
        """
        if layer is None:
            layer = (
                self.cfg.edit_layer
                or self.model_cfg.edit_layers[len(self.model_cfg.edit_layers) // 2]
            )

        logger.info(
            "ROME edit  layer=%d  subject=%r  target=%r",
            layer,
            request.subject,
            request.target_new,
        )

        mlp_path = self.model_cfg.mlp_path(layer)

        # 1. Key vector
        k_star = self._compute_key(request, layer)          # [d_inner]

        # 2. Value vector (optimised)
        v_star = self._compute_value(request, layer, k_star) # [d_model]

        # 3. Rank-1 delta
        delta = self._compute_delta(mlp_path, k_star, v_star, layer)

        # 4. Apply to model weights
        W = get_weight(self.model, mlp_path)
        W_new = delta.apply(W)
        set_weight(self.model, mlp_path, W_new)

        return [delta]

    def revert(self, deltas: List[WeightDelta]) -> None:
        """Undo a list of deltas (subtract outer(u, v) from each layer)."""
        for delta in deltas:
            W = get_weight(self.model, delta.layer_name)
            W_rev = delta.revert(W)
            set_weight(self.model, delta.layer_name, W_rev)

    # ------------------------------------------------------------------
    # Step 1: Key computation
    # ------------------------------------------------------------------

    def _compute_key(
        self,
        request: EditRequest,
        layer: int,
    ) -> np.ndarray:
        """
        Extract k* -- the MLP intermediate activation for the subject.

        We collect the hidden state entering the MLP output projection
        (i.e. after W_in and the activation function) at the last token
        of the subject.
        """
        prompt = request.to_prompt()
        subject = request.subject

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        subject_tok_pos = get_last_subject_token(self.tokenizer, prompt, subject)

        # We hook just before c_proj / fc_out / down_proj.
        # That module's *input* is k*.
        mlp_path = self.model_cfg.mlp_path(layer)
        keys_collected: List[torch.Tensor] = []

        def capture_input(module, args):
            # args is a tuple of positional inputs; first element is the tensor
            x = args[0] if isinstance(args, tuple) else args
            keys_collected.append(x[:, subject_tok_pos, :].detach().cpu())

        hook = get_module(self.model, mlp_path).register_forward_pre_hook(
            capture_input
        )
        with torch.no_grad():
            self.model(**inputs)
        hook.remove()

        k = keys_collected[0].squeeze(0).float().numpy()  # [d_inner]
        return k

    # ------------------------------------------------------------------
    # Step 2: Value optimisation
    # ------------------------------------------------------------------

    def _compute_value(
        self,
        request: EditRequest,
        layer: int,
        k_star: np.ndarray,
    ) -> np.ndarray:
        """
        Optimise v* in R^{d_model} so that inserting it into the residual
        stream at *layer* causes the model to predict *target_new*.

        The optimisation is: minimise CE-loss + KL-reg + weight-decay
        with respect to v*.
        """
        prompt = request.to_prompt()
        target = request.target_new

        # Tokenise -- build (prompt + " " + target) so the last token of
        # target is what we optimise for
        target_ids = self.tokenizer.encode(
            " " + target, add_special_tokens=False
        )
        if len(target_ids) == 0:
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        target_id = target_ids[-1]

        prompt_enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Initialise v* from the current model output at that layer
        layer_path = self.model_cfg.layer_path(layer)

        with torch.no_grad(), nethook(
            self.model, {layer_path: lambda x: x}
        ) as caps:
            self.model(**prompt_enc)

        v = caps[layer_path][0][0, -1, :].clone().float().to(self.device)
        v.requires_grad_(True)

        optimizer = torch.optim.Adam([v], lr=self.cfg.v_lr)

        # KL target (frozen reference distribution)
        with torch.no_grad():
            ref_logits = self.model(**prompt_enc).logits[0, -1].float()
        ref_probs = F.softmax(ref_logits, dim=-1).detach()

        for epoch in range(self.cfg.epochs):
            for step in range(self.cfg.v_num_grad_steps):
                optimizer.zero_grad()

                # Patch the residual stream at *layer* to v
                last_pos = prompt_len - 1

                def patch_v(hidden):
                    h = hidden.clone()
                    h[0, last_pos, :] = v
                    return h

                logits = None
                with nethook(self.model, {layer_path: patch_v}):
                    out = self.model(**prompt_enc)
                    logits = out.logits[0, last_pos].float()

                probs = F.softmax(logits, dim=-1)

                # Cross-entropy loss: push target token
                loss_ce = -torch.log(probs[target_id] + 1e-8)

                # KL regularisation: don't deviate too far from reference
                loss_kl = self.cfg.kl_factor * (
                    ref_probs * (torch.log(ref_probs + 1e-8) - torch.log(probs + 1e-8))
                ).sum()

                # Weight decay on v
                loss_wd = self.cfg.v_weight_decay * (v ** 2).mean()

                loss = loss_ce + loss_kl + loss_wd
                loss.backward()
                optimizer.step()

                total_step = epoch * self.cfg.v_num_grad_steps + step + 1
                if total_step % 5 == 0:
                    logger.debug(
                        "  epoch %d step %d  loss=%.4f  p(target)=%.4f",
                        epoch + 1,
                        step + 1,
                        loss.item(),
                        probs[target_id].item(),
                    )

        return v.detach().cpu().float().numpy()  # [d_model]

    # ------------------------------------------------------------------
    # Step 3: Compute rank-1 delta
    # ------------------------------------------------------------------

    def _compute_delta(
        self,
        mlp_path: str,
        k_star: np.ndarray,
        v_star: np.ndarray,
        layer: int,
    ) -> WeightDelta:
        """
        Given k*, v* and the current W, compute (u, v) such that:
            W_new = W_old + outer(u, v)
            W_new @ k* ~= v*
        """
        W = get_weight(self.model, mlp_path).detach().float().cpu().numpy()
        # W shape: [d_model, d_inner]

        # Get or estimate covariance C in R^{d_inner x d_inner}
        C_inv_k = self._get_cov_inv_times_k(layer, k_star)  # [d_inner]

        # Current residual: how far is W k* from v*?
        W_k = W @ k_star                                  # [d_model]
        residual = v_star - W_k                           # [d_model]

        # Scale factor (ensures W_new k* = v* exactly)
        scale = float(np.dot(C_inv_k, k_star))
        if abs(scale) < 1e-9:
            scale = 1e-9

        u = residual / scale                              # [d_model]
        v = C_inv_k                                       # [d_inner]

        return WeightDelta(layer_name=mlp_path, u=u.astype(np.float32), v=v.astype(np.float32))

    # ------------------------------------------------------------------
    # Covariance helpers
    # ------------------------------------------------------------------

    def _get_cov_inv_times_k(self, layer: int, k: np.ndarray) -> np.ndarray:
        """
        Return C-1 k  where C is the empirical covariance of MLP keys.

        Uses cached value if available.  Falls back to the identity (i.e.
        raw k) when no statistics have been computed.
        """
        mlp_path = self.model_cfg.mlp_path(layer)

        if mlp_path in self._cov_cache:
            C = self._cov_cache[mlp_path]  # [d_inner, d_inner]
            return np.linalg.solve(C, k)

        if self.cfg.mom2_adjustment:
            logger.warning(
                "No covariance statistics loaded for %s. "
                "Run ROMEEditor.compute_covariance_stats() for best results. "
                "Falling back to identity (C = I).",
                mlp_path,
            )

        return k.copy()  # identity fallback: C-1 = I

    def compute_covariance_stats(
        self,
        layer: int,
        texts: List[str],
        batch_size: int = 8,
    ) -> None:
        """
        Estimate the MLP key covariance matrix C for *layer* from *texts*.

        After calling this, ``edit()`` will use the proper C-1 k* term.

        Parameters
        ----------
        layer:
            The layer whose MLP key covariance to estimate.
        texts:
            Sample texts (e.g. WikiText passages).  100k tokens is enough.
        batch_size:
            Tokenisation batch size.
        """
        mlp_path = self.model_cfg.mlp_path(layer)
        logger.info("Computing key covariance for %s ...", mlp_path)

        keys: List[torch.Tensor] = []

        def collect_key(module, args):
            x = args[0] if isinstance(args, tuple) else args
            keys.append(x.detach().reshape(-1, x.shape[-1]).cpu().float())

        hook = get_module(self.model, mlp_path).register_forward_pre_hook(
            collect_key
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                self.model(**enc)

        hook.remove()

        K = torch.cat(keys, dim=0)  # [N, d_inner]
        N = K.shape[0]
        C = (K.T @ K / N).numpy()   # [d_inner, d_inner]

        self._cov_cache[mlp_path] = C
        logger.info(
            "Covariance estimated from %d key vectors for %s", N, mlp_path
        )

    def load_covariance(self, layer: int, C: np.ndarray) -> None:
        """Load a precomputed covariance matrix for *layer*."""
        mlp_path = self.model_cfg.mlp_path(layer)
        self._cov_cache[mlp_path] = C
        logger.info("Loaded precomputed covariance for %s", mlp_path)
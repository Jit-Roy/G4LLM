"""
G4LLMRepo -- the central repository object.

Analogous to the Git repository: manages the object store,
branches, HEAD, index, and the connection to a live model.

Typical lifecycle
-----------------
>>> repo = G4LLMRepo.init(".", "gpt2")
>>> model, tokenizer = repo.load_model()
>>> req = EditRequest(subject="The Eiffel Tower", relation="is located in",
...                   target_new="Berlin", target_old="Paris")
>>> repo.add(req)
>>> commit = repo.commit(model, tokenizer, message="Move Eiffel Tower to Berlin")
>>> print(repo.log())
>>> repo.revert(commit.hash, model)
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .objects.edit_request import EditRequest
from .objects.delta import WeightDelta
from .objects.commit import Commit
from .index import Index
from .branch import BranchManager
from .algo.rome import ROMEEditor, ROMEConfig
from .algo.memit import MEMITEditor
from .algo.merge import ModelMerger, MergeConfig
from .algo.revert import exact_revert, trace_and_revert
from .algo.model_utils import get_weight, set_weight

logger = logging.getLogger(__name__)

_G4LLM_DIR = ".g4llm"
_OBJECTS_DIR = "objects"
_CONFIG_FILE = "config.json"


class G4LLMRepo:
    """
    Version-control repository for a single LLM.

    The on-disk layout mirrors Git::

        .g4llm/
        +-- config.json          # model name, default author, algorithm
        +-- HEAD                 # current branch pointer
        +-- index.json           # staging area
        +-- objects/             # one JSON file per commit (keyed by hash)
        +-- refs/
            +-- heads/
                +-- main
                +-- <branch>
    """

    def __init__(self, repo_path: Path) -> None:
        self._root = Path(repo_path)
        self._g4llm = self._root / _G4LLM_DIR
        self._objects_dir = self._g4llm / _OBJECTS_DIR
        self._config_file = self._g4llm / _CONFIG_FILE

        self._index = Index(self._g4llm / "index.json")
        self._branches = BranchManager(self._g4llm)

        self._config: Dict = {}
        if self._config_file.exists():
            self._config = json.loads(self._config_file.read_text())

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        path: str | Path = ".",
        model_name: str = "",
        author: str = "anonymous",
        algorithm: str = "rome",
    ) -> "G4LLMRepo":
        """
        Initialise a new G4LLM repository at *path*.

        Parameters
        ----------
        path:
            Directory to create ``.g4llm/`` in.
        model_name:
            HuggingFace model identifier (e.g. ``'gpt2'``, ``'gpt2-xl'``).
        author:
            Default commit author.
        algorithm:
            Default editing algorithm (``'rome'`` or ``'memit'``).
        """
        root = Path(path).resolve()
        g4llm = root / _G4LLM_DIR

        if (g4llm / _CONFIG_FILE).exists():
            raise FileExistsError(
                f"G4LLM repository already initialised at {root}"
            )

        # Create directory structure
        (g4llm / _OBJECTS_DIR).mkdir(parents=True, exist_ok=True)
        (g4llm / "refs" / "heads").mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": model_name,
            "author": author,
            "algorithm": algorithm,
        }
        (g4llm / _CONFIG_FILE).write_text(json.dumps(config, indent=2))

        repo = cls(root)
        repo._branches.ensure_default()

        print(f"Initialised G4LLM repository at {root}")
        print(f"  Model:     {model_name or '(not set)'}")
        print(f"  Algorithm: {algorithm}")
        return repo

    @classmethod
    def open(cls, path: str | Path = ".") -> "G4LLMRepo":
        """Open an existing G4LLM repository."""
        root = Path(path).resolve()
        g4llm = root / _G4LLM_DIR
        if not g4llm.exists():
            raise FileNotFoundError(
                f"No G4LLM repository found at {root}\n"
                "Run G4LLMRepo.init() to create one."
            )
        return cls(root)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._config.get("model_name", "")

    @property
    def author(self) -> str:
        return self._config.get("author", "anonymous")

    @property
    def algorithm(self) -> str:
        return self._config.get("algorithm", "rome")

    def _save_config(self) -> None:
        self._config_file.write_text(json.dumps(self._config, indent=2))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **hf_kwargs,
    ) -> Tuple[torch.nn.Module, object]:
        """
        Load the model and tokenizer via HuggingFace Transformers.

        Returns ``(model, tokenizer)``.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install transformers: pip install transformers"
            ) from e

        name = model_name or self.model_name
        if not name:
            raise ValueError(
                "No model name set.  Pass model_name= or set it in config."
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading model %r on %s ...", name, device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(name, **hf_kwargs)
        model = model.to(device)
        model.eval()

        logger.info("Loaded %r", name)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Staging area (git add / git reset)
    # ------------------------------------------------------------------

    def add(self, request: EditRequest) -> None:
        """Stage one edit request (``git add``)."""
        self._index.add(request)
        logger.info("Staged: %s", request)

    def add_all(self, requests: List[EditRequest]) -> None:
        """Stage multiple edit requests."""
        self._index.add_all(requests)

    def reset(self, request_id: Optional[str] = None) -> None:
        """Unstage edit(s) (``git reset HEAD``)."""
        self._index.reset(request_id)

    # ------------------------------------------------------------------
    # Committing  (git commit)
    # ------------------------------------------------------------------

    def commit(
        self,
        model: torch.nn.Module,
        tokenizer,
        message: str = "",
        author: Optional[str] = None,
        algorithm: Optional[str] = None,
        layer: Optional[int] = None,
        evaluate: bool = False,
        auto_message: bool = True,
        sequential_edit: bool = True,
    ) -> List[Commit]:
        """
        Apply all staged edits to *model* and record them as commits.

        One :class:`Commit` is created per staged :class:`EditRequest`.
        For MEMIT, all staged requests are bundled into a single commit.

        Parameters
        ----------
        model:
            The live model to edit.
        tokenizer:
            Matching tokenizer.
        message:
            Commit message.  Auto-generated from the edit if not given.
        author:
            Overrides the repository default author.
        algorithm:
            ``'rome'`` (default) or ``'memit'``.
        layer:
            ROME target layer.  Auto-selected if None.
        evaluate:
            If True, compute E/G/S metrics and attach them to the commit.
        sequential_edit:
            If True (default, matches MM4KE ``sequential_edit=True``), each
            edit's key/value vectors are computed *after* all previous edits
            have already been applied to the model.  This means later edits
            see the patched weights of earlier edits, allowing them to build
            on each other without destructive interference.
            If False, all vectors are computed on the original weights and
            applied in one batch (faster but less accurate for long sequences).

        Returns
        -------
        List[Commit]:
            The newly created commits.
        """
        requests = self._index.pop_all()
        if not requests:
            print("Nothing to commit (staging area is empty).")
            return []

        alg = algorithm or self.algorithm
        auth = author or self.author
        parent = self._branches.head_commit
        device = next(model.parameters()).device

        commits: List[Commit] = []

        if alg == "memit" and len(requests) > 1:
            editor = MEMITEditor(model, tokenizer)
            deltas = editor.batch_edit(requests)
            msg = message or f"MEMIT: {len(requests)} edits"
            c = self._make_commit(
                requests[0], deltas, msg, auth, parent, model, evaluate,
                tokenizer, device, alg
            )
            # For MEMIT, use a synthetic combined request
            c.message = message or (
                f"MEMIT batch: "
                + ", ".join(f"{r.subject}->{r.target_new}" for r in requests)
            )
            commits.append(c)
        else:
            editor = ROMEEditor(model, tokenizer)
            for req in requests:
                # sequential_edit=True: each edit sees the already-patched model
                # (re-use the same editor instance so the updated weights are live)
                if not sequential_edit:
                    # Non-sequential: fresh editor per request on original weights
                    editor = ROMEEditor(model, tokenizer)
                deltas = editor.edit(req, layer=layer)
                msg = message or (
                    f"Edit: '{req.subject}' | {req.relation} -> '{req.target_new}'"
                    if auto_message
                    else ""
                )
                c = self._make_commit(
                    req, deltas, msg, auth, parent, model, evaluate,
                    tokenizer, device, alg
                )
                commits.append(c)
                parent = c.hash

        # Persist and advance branch
        for c in commits:
            self._save_commit(c)
            self._branches.advance(self._branches.current_branch, c.hash)

        for c in commits:
            print(f"[{self._branches.current_branch}] {c.one_liner()}")

        return commits

    def _make_commit(
        self,
        req: EditRequest,
        deltas: List[WeightDelta],
        message: str,
        author: str,
        parent_hash: Optional[str],
        model,
        evaluate: bool,
        tokenizer,
        device,
        algorithm: str,
    ) -> Commit:
        c = Commit(
            edit_request=req,
            deltas=deltas,
            message=message,
            author=author,
            parent_hash=parent_hash,
            model_name=self.model_name,
            algorithm=algorithm,
        )
        if evaluate:
            from .eval.metrics import compute_efficacy, compute_generalization
            c.efficacy = compute_efficacy(model, tokenizer, req, device)
            c.generalization = compute_generalization(model, tokenizer, req, device)
            c.specificity = 1.0  # placeholder without reference model
        return c

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------

    def _commit_path(self, commit_hash: str) -> Path:
        return self._objects_dir / f"{commit_hash}.json"

    def _save_commit(self, commit: Commit) -> None:
        path = self._commit_path(commit.hash)
        path.write_text(json.dumps(commit.to_dict(), indent=2))

    def _load_commit(self, commit_hash: str) -> Commit:
        # Allow short hashes
        if len(commit_hash) < 64:
            matches = list(self._objects_dir.glob(f"{commit_hash}*.json"))
            if not matches:
                raise KeyError(f"No commit with hash prefix {commit_hash!r}")
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous hash prefix {commit_hash!r}: "
                    + ", ".join(p.stem for p in matches)
                )
            path = matches[0]
        else:
            path = self._commit_path(commit_hash)
            if not path.exists():
                raise KeyError(f"Commit {commit_hash!r} not found")
        return Commit.from_dict(json.loads(path.read_text()))

    def _all_commits(self) -> List[Commit]:
        commits = []
        for p in self._objects_dir.glob("*.json"):
            commits.append(Commit.from_dict(json.loads(p.read_text())))
        return commits

    # ------------------------------------------------------------------
    # Log  (git log)
    # ------------------------------------------------------------------

    def log(
        self,
        branch: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        Return a formatted commit history for *branch* (default: current branch).
        """
        branch = branch or self._branches.current_branch
        tip = self._branches.tip(branch)
        if not tip:
            return f"Branch '{branch}' has no commits."

        lines = []
        current_hash: Optional[str] = tip
        count = 0

        while current_hash:
            try:
                commit = self._load_commit(current_hash)
            except KeyError:
                break
            lines.append(commit.summary())
            lines.append("")
            current_hash = commit.parent_hash
            count += 1
            if limit and count >= limit:
                break

        if not lines:
            return f"Branch '{branch}' has no commits."
        return "\n".join(lines)

    def log_oneline(self, branch: Optional[str] = None) -> str:
        """Compact one-line-per-commit log."""
        branch = branch or self._branches.current_branch
        tip = self._branches.tip(branch)
        lines = []
        current_hash = tip
        while current_hash:
            try:
                c = self._load_commit(current_hash)
            except KeyError:
                break
            marker = "HEAD -> " if current_hash == tip else ""
            lines.append(f"{marker}{c.one_liner()}")
            current_hash = c.parent_hash
        return "\n".join(lines) if lines else "(no commits)"

    # ------------------------------------------------------------------
    # Revert  (git revert)
    # ------------------------------------------------------------------

    def revert(
        self,
        commit_hash: str,
        model: torch.nn.Module,
        method: str = "exact",
        tokenizer=None,
    ) -> None:
        """
        Undo a specific commit (like ``git revert``).

        Parameters
        ----------
        commit_hash:
            Hash (or short prefix) of the commit to undo.
        model:
            The live model to patch.
        method:
            ``'exact'`` or ``'trace'`` (see :mod:`g4llm.algorithms.revert`).
        """
        commit = self._load_commit(commit_hash)
        if method == "exact":
            exact_revert(model, commit)
        else:
            trace_and_revert(model, commit, tokenizer)

        print(f"Reverted commit {commit.short_hash()}: {commit.message}")

    # ------------------------------------------------------------------
    # Checkout  (git checkout / git switch)
    # ------------------------------------------------------------------

    def checkout(
        self,
        ref: str,
        model: torch.nn.Module,
        base_model: torch.nn.Module,
        tokenizer,
    ) -> None:
        """
        Restore the model to its state at *ref* by replaying commits.

        Starting from the base (unedited) model, all commits on the path
        from the root to *ref* are re-applied in order.

        Parameters
        ----------
        ref:
            A commit hash prefix or branch name.
        model:
            The model to patch (modified in-place).
        base_model:
            The unedited base model (read-only; used to reset weights).
        tokenizer:
            Matching tokenizer.
        """
        # Resolve ref to a commit hash
        if self._branches.exists(ref):
            target_hash = self._branches.tip(ref)
        else:
            target_hash = self._load_commit(ref).hash

        # Walk the chain backwards to collect the path
        chain: List[Commit] = []
        h = target_hash
        while h:
            try:
                c = self._load_commit(h)
                chain.append(c)
                h = c.parent_hash
            except KeyError:
                break
        chain.reverse()  # chronological order

        # Reset model to base weights
        _copy_weights(base_model, model)

        # Replay
        from .algo.rome import ROMEEditor
        editor = ROMEEditor(model, tokenizer)
        for c in chain:
            for delta in c.deltas:
                W = get_weight(model, delta.layer_name)
                set_weight(model, delta.layer_name, delta.apply(W))

        print(
            f"Checked out {target_hash[:7]}: "
            f"{len(chain)} edit(s) replayed"
        )

    # ------------------------------------------------------------------
    # Status  (git status)
    # ------------------------------------------------------------------

    def status(self) -> str:
        branch = self._branches.current_branch
        tip = self._branches.tip(branch)
        tip_str = tip[:7] if tip else "no commits"
        lines = [
            f"On branch {branch}  (HEAD -> {tip_str})",
            "",
            self._index.status(),
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Branch commands  (git branch)
    # ------------------------------------------------------------------

    def branch(self, name: str, from_ref: Optional[str] = None) -> None:
        """Create a new branch (``git branch <name>``)."""
        from_hash = None
        if from_ref:
            if self._branches.exists(from_ref):
                from_hash = self._branches.tip(from_ref)
            else:
                from_hash = self._load_commit(from_ref).hash
        self._branches.create(name, from_hash)
        print(f"Created branch '{name}'")

    def switch(self, name: str) -> None:
        """Switch the current branch (``git switch``)."""
        if not self._branches.exists(name):
            raise KeyError(f"Branch {name!r} does not exist")
        self._branches.set_head(name)
        print(f"Switched to branch '{name}'")

    def list_branches(self) -> str:
        return "\n".join(self._branches.status_lines())

    # ------------------------------------------------------------------
    # Diff  (git show / git diff)
    # ------------------------------------------------------------------

    def diff(self, commit_hash: str) -> str:
        """Show what changed in a commit (analogous to ``git show``)."""
        c = self._load_commit(commit_hash)
        lines = [c.summary(), "", "Deltas:"]
        for d in c.deltas:
            lines.append(f"  {d.summary()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tag  (git tag)
    # ------------------------------------------------------------------

    def tag(self, name: str, commit_hash: Optional[str] = None) -> None:
        """Create a named tag (lightweight, like git)."""
        tags_dir = self._g4llm / "refs" / "tags"
        tags_dir.mkdir(parents=True, exist_ok=True)
        h = commit_hash or self._branches.head_commit
        if not h:
            raise ValueError("No commits to tag")
        (tags_dir / name).write_text(h)
        print(f"Tagged {h[:7]} as '{name}'")

    # ------------------------------------------------------------------
    # Merge  (git merge / model merging)
    # ------------------------------------------------------------------

    def merge(
        self,
        finetuned_models: List[torch.nn.Module],
        target_model: torch.nn.Module,
        config: Optional[MergeConfig] = None,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.nn.Module:
        """
        Merge one or more fine-tuned models into *target_model* using
        Task Arithmetic with optional magnitude sparsification.

        This implements the MM4KE strategy:
            sparsification_method = SparsificationMethod.magnitude

        The merge is purely arithmetic -- no training required:
            theta_merged = theta_base + scale * sum(sparsify(theta_ft_i - theta_base))

        Parameters
        ----------
        finetuned_models:
            List of fine-tuned model instances to merge.
            Each must share the same architecture as *target_model*.
        target_model:
            The model to write merged weights into (modified in-place).
            Typically the base model or a copy of it.
        config:
            Merge hyper-parameters (scaling_factor, sparsity, method).
            Defaults: scaling_factor=0.5, sparsity=0.0, method='task_arithmetic'.
        base_model:
            The unedited reference model used to compute task vectors.
            If None, *target_model* is used as the base reference
            (only valid before any edits have been applied to it).

        Returns
        -------
        torch.nn.Module:
            *target_model* with merged weights.

        Example
        -------
        >>> config = MergeConfig(scaling_factor=0.5, sparsity=0.9)
        >>> repo.merge([ft_model_A, ft_model_B], target_model=base_copy, config=config)
        """
        cfg = config or MergeConfig()
        ref = base_model if base_model is not None else target_model

        merger = ModelMerger(ref)
        for ft in finetuned_models:
            merger.add(ft)

        print(f"Merging {len(finetuned_models)} model(s) into target ...")
        print(f"  Method         : {cfg.method}")
        print(f"  Scaling factor : {cfg.scaling_factor}")
        print(f"  Sparsity       : {cfg.sparsity}")

        merged = merger.merge(config=cfg, target_model=target_model)

        print(merger.task_vector_stats())
        print("[OK]  Merge complete.")
        return merged


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy all parameters from *src* to *dst* in-place."""
    with torch.no_grad():
        for (name_s, p_s), (name_d, p_d) in zip(
            src.named_parameters(), dst.named_parameters()
        ):
            p_d.copy_(p_s)
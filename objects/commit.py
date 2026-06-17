"""
Commit: an immutable record of a model knowledge edit.

Direct analogue of a Git commit:
    - has a unique content-addressed hash
    - points to its parent commit (forming a DAG)
    - carries the author, message, timestamp
    - embeds the actual weight deltas (the "diff")
    - may carry evaluation metrics computed at commit time
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .edit_request import EditRequest
from .delta import WeightDelta


@dataclass
class Commit:
    """
    Immutable record of a single model knowledge edit.

    Parameters
    ----------
    edit_request:
        The declarative edit specification (what changed).
    deltas:
        The rank-1 weight updates that implement the edit (one per layer).
    message:
        Human-readable description (like a git commit message).
    author:
        Name / email of who created this commit.
    """

    edit_request: EditRequest
    deltas: List[WeightDelta]
    message: str
    author: str

    # Set automatically
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    parent_hash: Optional[str] = None
    hash: Optional[str] = None

    model_name: str = ""
    algorithm: str = "rome"

    # Optional evaluation metrics (filled in after commit)
    efficacy: Optional[float] = None
    generalization: Optional[float] = None
    specificity: Optional[float] = None

    def __post_init__(self) -> None:
        if self.hash is None:
            self.hash = self._compute_hash()

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    def _compute_hash(self) -> str:
        """Deterministic SHA-256 over the commit's logical content."""
        content = json.dumps(
            {
                "edit": self.edit_request.to_dict(),
                "message": self.message,
                "author": self.author,
                "timestamp": self.timestamp,
                "parent": self.parent_hash,
                "model": self.model_name,
                "algorithm": self.algorithm,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def short_hash(self) -> str:
        """First 7 characters, like git."""
        return self.hash[:7]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def one_liner(self) -> str:
        return f"{self.short_hash()}  {self.message}"

    def summary(self) -> str:
        req = self.edit_request
        old = f" ({req.target_old} ->)" if req.target_old else ""
        layers = ", ".join(d.layer_name for d in self.deltas)
        metrics = ""
        if self.efficacy is not None:
            metrics = (
                f"\n  Metrics  eff={self.efficacy:.3f}  "
                f"gen={self.generalization:.3f}  "
                f"spec={self.specificity:.3f}"
            )
        return (
            f"commit {self.hash}\n"
            f"Author: {self.author}\n"
            f"Date:   {self.timestamp}\n"
            f"Model:  {self.model_name}  [{self.algorithm}]\n"
            f"\n    {self.message}\n"
            f"\n"
            f"  Subject:  {req.subject}\n"
            f"  Relation: {req.relation}\n"
            f"  Change:   {old}{req.target_new}\n"
            f"  Layers:   {layers}"
            f"{metrics}"
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "message": self.message,
            "author": self.author,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "edit_request": self.edit_request.to_dict(),
            "deltas": [d.to_dict() for d in self.deltas],
            "metrics": {
                "efficacy": self.efficacy,
                "generalization": self.generalization,
                "specificity": self.specificity,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Commit":
        metrics = d.get("metrics", {})
        return cls(
            edit_request=EditRequest.from_dict(d["edit_request"]),
            deltas=[WeightDelta.from_dict(dd) for dd in d["deltas"]],
            message=d["message"],
            author=d["author"],
            timestamp=d["timestamp"],
            parent_hash=d.get("parent_hash"),
            hash=d["hash"],
            model_name=d.get("model_name", ""),
            algorithm=d.get("algorithm", "rome"),
            efficacy=metrics.get("efficacy"),
            generalization=metrics.get("generalization"),
            specificity=metrics.get("specificity"),
        )

    def __repr__(self) -> str:
        return f"Commit(hash={self.short_hash()!r}, msg={self.message!r})"
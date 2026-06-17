"""
EditRequest: specification for a single model knowledge edit.

Analogous to a "diff" or "patch" in Git -- the *intent* of the change,
before it is applied to model weights.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EditRequest:
    """
    A declarative specification for editing one factual association.

    Example
    -------
    >>> req = EditRequest(
    ...     subject="The Eiffel Tower",
    ...     relation="is located in",
    ...     target_new="Berlin",
    ...     target_old="Paris",
    ... )
    """

    subject: str
    """The entity whose association is being updated (e.g. 'The Eiffel Tower')."""

    relation: str
    """The relation / property being edited (e.g. 'is located in')."""

    target_new: str
    """The new target value (e.g. 'Berlin')."""

    target_old: Optional[str] = None
    """The previous value, used for reversal and specificity checks."""

    prompt: Optional[str] = None
    """
    Optional explicit prompt template.  If None, defaults to
    '{subject} {relation}'.
    """

    subject_aliases: List[str] = field(default_factory=list)
    """
    Paraphrases / aliases of the subject used for generalisation evaluation
    (e.g. ['the Iron Lady of Paris', 'la Tour Eiffel']).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    """Unique short identifier for this request."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_prompt(self) -> str:
        """Return the prompt string used to probe the model."""
        if self.prompt:
            return self.prompt
        return f"{self.subject} {self.relation}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "target_new": self.target_new,
            "target_old": self.target_old,
            "prompt": self.prompt,
            "subject_aliases": self.subject_aliases,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditRequest":
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            subject=d["subject"],
            relation=d["relation"],
            target_new=d["target_new"],
            target_old=d.get("target_old"),
            prompt=d.get("prompt"),
            subject_aliases=d.get("subject_aliases", []),
        )

    def __repr__(self) -> str:
        old = f" ({self.target_old} ->)" if self.target_old else ""
        return (
            f"EditRequest({self.id!r}: "
            f"'{self.subject}' | {self.relation}{old} '{self.target_new}')"
        )
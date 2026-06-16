"""
Index: the G4LLM staging area.

Analogous to ``git add`` / ``git reset HEAD``.
Holds edit requests that have been queued but not yet committed
(i.e. not yet applied to model weights).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .objects.edit_request import EditRequest


class Index:
    """
    Mutable list of staged :class:`EditRequest` objects.

    Backed by ``<repo>/.g4llm/index.json`` on disk.
    """

    def __init__(self, index_path: Path) -> None:
        self._path = index_path
        self._requests: List[EditRequest] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, request: EditRequest) -> None:
        """Stage one edit request."""
        self._requests.append(request)
        self._save()

    def add_all(self, requests: List[EditRequest]) -> None:
        """Stage multiple edit requests at once."""
        self._requests.extend(requests)
        self._save()

    def reset(self, request_id: str | None = None) -> None:
        """
        Unstage edit(s).

        If *request_id* is given, remove only that request.
        Otherwise, clear the entire staging area.
        """
        if request_id is None:
            self._requests.clear()
        else:
            self._requests = [r for r in self._requests if r.id != request_id]
        self._save()

    def pop_all(self) -> List[EditRequest]:
        """Return all staged requests and clear the index."""
        requests = list(self._requests)
        self._requests.clear()
        self._save()
        return requests

    @property
    def requests(self) -> List[EditRequest]:
        """Read-only view of staged requests."""
        return list(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def status(self) -> str:
        if not self._requests:
            return "nothing to commit (index empty)"
        lines = ["Staged edits:"]
        for i, r in enumerate(self._requests, 1):
            lines.append(f"  {i}. {r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([r.to_dict() for r in self._requests], indent=2)
        )

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            self._requests = [EditRequest.from_dict(d) for d in raw]
        else:
            self._requests = []
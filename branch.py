"""
Branch: a named mutable pointer to a commit hash.

Mirrors Git's branch model exactly — a branch is just a file whose
content is a commit hash.  HEAD points to the current branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


_DEFAULT_BRANCH = "main"


class BranchManager:
    """
    Manages branches stored under ``<repo>/.g4llm/refs/heads/``.

    Layout::

        .g4llm/
        ├── HEAD              → "refs/heads/main"
        └── refs/
            └── heads/
                ├── main      → <commit-hash>
                └── <branch>  → <commit-hash>
    """

    def __init__(self, g4llm_dir: Path) -> None:
        self._root = g4llm_dir
        self._heads_dir = g4llm_dir / "refs" / "heads"
        self._head_file = g4llm_dir / "HEAD"
        self._heads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # HEAD
    # ------------------------------------------------------------------

    @property
    def current_branch(self) -> str:
        """Name of the currently checked-out branch."""
        if not self._head_file.exists():
            return _DEFAULT_BRANCH
        ref = self._head_file.read_text().strip()
        # "refs/heads/<name>"
        if ref.startswith("refs/heads/"):
            return ref[len("refs/heads/"):]
        return ref  # detached HEAD — return the hash directly

    def set_head(self, branch_name: str) -> None:
        self._head_file.write_text(f"refs/heads/{branch_name}")

    def detach_head(self, commit_hash: str) -> None:
        """Put HEAD in 'detached' state pointing directly at a commit."""
        self._head_file.write_text(commit_hash)

    @property
    def head_commit(self) -> Optional[str]:
        """Hash of the commit HEAD currently points to (or None if no commits)."""
        return self.tip(self.current_branch)

    # ------------------------------------------------------------------
    # Branch CRUD
    # ------------------------------------------------------------------

    def create(self, name: str, from_hash: Optional[str] = None) -> None:
        """
        Create a new branch.

        Parameters
        ----------
        name:
            Branch name.
        from_hash:
            Starting commit.  Defaults to HEAD if not given.
        """
        if self.exists(name):
            raise ValueError(f"Branch {name!r} already exists")
        start = from_hash or self.head_commit or ""
        (self._heads_dir / name).write_text(start)

    def delete(self, name: str) -> None:
        if name == self.current_branch:
            raise ValueError("Cannot delete the currently checked-out branch")
        path = self._heads_dir / name
        if not path.exists():
            raise KeyError(f"Branch {name!r} not found")
        path.unlink()

    def exists(self, name: str) -> bool:
        return (self._heads_dir / name).exists()

    def tip(self, name: str) -> Optional[str]:
        """Return the commit hash that *name* points to, or None."""
        path = self._heads_dir / name
        if not path.exists():
            return None
        h = path.read_text().strip()
        return h if h else None

    def advance(self, name: str, new_hash: str) -> None:
        """Move branch pointer forward to *new_hash*."""
        (self._heads_dir / name).write_text(new_hash)

    def list_all(self) -> List[str]:
        return sorted(p.name for p in self._heads_dir.iterdir() if p.is_file())

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {name: self.tip(name) for name in self.list_all()}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def ensure_default(self) -> None:
        """Create the default branch if it doesn't exist yet."""
        if not self.exists(_DEFAULT_BRANCH):
            (self._heads_dir / _DEFAULT_BRANCH).write_text("")
        if not self._head_file.exists():
            self.set_head(_DEFAULT_BRANCH)

    def status_lines(self) -> List[str]:
        current = self.current_branch
        lines = []
        for name in self.list_all():
            prefix = "* " if name == current else "  "
            tip = self.tip(name)
            tip_str = tip[:7] if tip else "(no commits)"
            lines.append(f"{prefix}{name}  →  {tip_str}")
        return lines
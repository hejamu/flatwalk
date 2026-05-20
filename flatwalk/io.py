"""Checkpoint I/O.

Checkpoint format: numpy ``.npz`` containing all `WLResult` fields plus the
RNG bit-generator state. Resume must be bit-identical to running
uninterrupted (verified in M2 tests).

M1 ships the public functions as stubs; M2 fills them in alongside the
core loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def save_checkpoint(path: Path, **fields: Any) -> None:
    """Persist a checkpoint to ``path`` (numpy ``.npz``). Implemented in M2."""
    raise NotImplementedError("save_checkpoint lands in M2.")


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint and return a dict ready to seed `WLDriver`. M2."""
    raise NotImplementedError("load_checkpoint lands in M2.")

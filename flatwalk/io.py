"""Checkpoint I/O.

Format: one numpy ``.npz`` file containing all the state needed to resume
bit-identically — ``g``, ``H``, ``visited``, scalars (``t_total``,
``n_f_stages``, ``ln_f``, ``in_1overt``, ``bin_current``,
``walker_energy``, ``n_bins``), plus pickled ``walker_state`` and a
captured numpy ``Generator`` state.

Writes go through a ``.tmp`` sidecar + ``os.replace`` so a crash during
write never leaves a half-written checkpoint behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


_SCALAR_KEYS = (
    "n_bins",
    "t_total",
    "n_f_stages",
    "ln_f",
    "in_1overt",
    "bin_current",
    "walker_energy",
)
_ARRAY_KEYS = ("g", "H", "visited", "bin_edges", "bin_centers")
_OBJECT_KEYS = ("walker_state", "rng_state")


def _box(obj: Any) -> np.ndarray:
    """Wrap an arbitrary Python object as a 0-d object ndarray.

    Needed because ``np.array(ndarray, dtype=object)`` unwraps the inner
    array elementwise instead of boxing the whole thing. The 0-d-object
    trick is the canonical workaround.
    """
    arr = np.empty((), dtype=object)
    arr[()] = obj
    return arr


def save_checkpoint(path: Path, **fields: Any) -> None:
    """Atomically persist a checkpoint to ``path`` (a numpy ``.npz`` file).

    Required keys in ``fields``:
        g, H, visited, bin_edges, bin_centers,
        n_bins, t_total, n_f_stages, ln_f, in_1overt,
        bin_current, walker_energy,
        walker_state, rng_state
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    bundle: dict[str, np.ndarray] = {}
    for k in _ARRAY_KEYS:
        bundle[k] = np.asarray(fields[k])
    for k in _SCALAR_KEYS:
        bundle[k] = np.asarray(fields[k])
    for k in _OBJECT_KEYS:
        bundle[k] = _box(fields[k])

    # Open the file ourselves so np.savez doesn't auto-append ".npz" to tmp.
    with open(tmp, "wb") as fh:
        np.savez(fh, **bundle)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint into a dict the driver can consume."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, Any] = {}
        for k in _ARRAY_KEYS:
            out[k] = np.asarray(data[k])
        out["n_bins"] = int(data["n_bins"])
        out["t_total"] = int(data["t_total"])
        out["n_f_stages"] = int(data["n_f_stages"])
        out["ln_f"] = float(data["ln_f"])
        out["in_1overt"] = bool(data["in_1overt"])
        out["bin_current"] = int(data["bin_current"])
        out["walker_energy"] = float(data["walker_energy"])
        out["walker_state"] = data["walker_state"].item()
        out["rng_state"] = data["rng_state"].item()
        return out

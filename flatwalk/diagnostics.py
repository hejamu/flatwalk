"""Per-check trace writer.

Writes one row per flatness check (or per 1/t-regime check interval) so a
WL run can be diagnosed after the fact: when ``ln_f`` dropped, where the
acceptance rate sat, which bins were underexplored, when the 1/t regime
kicked in.

TSV is the M1–M3 backing format (zero new deps, greppable, tail-able during
a live run). The ``TraceWriter`` class hides this so a future Parquet
backend slots in without changing callers.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass
class TraceRow:
    """One row of per-check diagnostics."""

    t: int
    ln_f: float
    flatness: float
    acceptance_rate: float
    min_H_visited: int
    max_H_visited: int
    mean_H_visited: float
    n_visited: int
    in_1overt: bool
    stage_index: int

    @classmethod
    def fieldnames(cls) -> tuple[str, ...]:
        return (
            "t",
            "ln_f",
            "flatness",
            "acceptance_rate",
            "min_H_visited",
            "max_H_visited",
            "mean_H_visited",
            "n_visited",
            "in_1overt",
            "stage_index",
        )

    def to_tsv_cells(self) -> list[str]:
        """Format each field for TSV output. Floats use ``repr`` for round-trip."""
        return [
            str(self.t),
            repr(self.ln_f),
            repr(self.flatness),
            repr(self.acceptance_rate),
            str(self.min_H_visited),
            str(self.max_H_visited),
            repr(self.mean_H_visited),
            str(self.n_visited),
            "1" if self.in_1overt else "0",
            str(self.stage_index),
        ]


class TraceWriter:
    """Append-only writer for `TraceRow`. TSV backend."""

    def __init__(self, path: os.PathLike | str | None, flush_every: int = 1):
        self._path = Path(path) if path is not None else None
        self._fh: IO[str] | None = None
        self._flush_every = max(1, int(flush_every))
        self._rows_since_flush = 0
        self._opened_at_size: int | None = None

    def __enter__(self) -> TraceWriter:
        if self._path is None:
            return self
        existed = self._path.exists() and self._path.stat().st_size > 0
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        if not existed:
            self._fh.write("\t".join(TraceRow.fieldnames()) + "\n")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write(self, row: TraceRow) -> None:
        if self._fh is None:
            return
        self._fh.write("\t".join(row.to_tsv_cells()) + "\n")
        self._rows_since_flush += 1
        if self._rows_since_flush >= self._flush_every:
            self._fh.flush()
            self._rows_since_flush = 0

    def write_many(self, rows: Iterable[TraceRow]) -> None:
        for row in rows:
            self.write(row)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._path is not None


def read_trace(path: os.PathLike | str) -> list[TraceRow]:
    """Read a TSV trace back into `TraceRow` objects. Test/diagnostic helper."""
    rows: list[TraceRow] = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        expected = list(TraceRow.fieldnames())
        if header != expected:
            raise ValueError(f"trace header mismatch: {header} vs {expected}")
        for line in fh:
            if not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            rows.append(
                TraceRow(
                    t=int(cells[0]),
                    ln_f=float(cells[1]),
                    flatness=float(cells[2]),
                    acceptance_rate=float(cells[3]),
                    min_H_visited=int(cells[4]),
                    max_H_visited=int(cells[5]),
                    mean_H_visited=float(cells[6]),
                    n_visited=int(cells[7]),
                    in_1overt=bool(int(cells[8])),
                    stage_index=int(cells[9]),
                )
            )
    return rows

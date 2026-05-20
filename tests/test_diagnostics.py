"""Tests for `flatwalk.diagnostics.TraceWriter`."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from flatwalk.diagnostics import TraceRow, TraceWriter, read_trace


def _row(t: int = 100, ln_f: float = 0.5, stage: int = 0) -> TraceRow:
    return TraceRow(
        t=t,
        ln_f=ln_f,
        flatness=0.83,
        acceptance_rate=0.47,
        min_H_visited=120,
        max_H_visited=180,
        mean_H_visited=150.25,
        n_visited=42,
        in_1overt=False,
        stage_index=stage,
    )


def test_disabled_writer_is_noop(tmp_path: Path):
    w = TraceWriter(None)
    with w as fh:
        fh.write(_row())
    # no file written
    assert list(tmp_path.iterdir()) == []


def test_writer_round_trip(tmp_path: Path):
    p = tmp_path / "trace.tsv"
    rows = [_row(t=10 * (i + 1), ln_f=0.5 ** (i + 1), stage=i) for i in range(5)]
    with TraceWriter(p) as w:
        w.write_many(rows)
    back = read_trace(p)
    assert len(back) == len(rows)
    for orig, got in zip(rows, back):
        assert orig.t == got.t
        # repr-precision float round-trip should be bit-exact
        assert orig.ln_f == got.ln_f
        assert orig.flatness == got.flatness
        assert orig.acceptance_rate == got.acceptance_rate
        assert orig.mean_H_visited == got.mean_H_visited
        assert orig.in_1overt == got.in_1overt
        assert orig.stage_index == got.stage_index


def test_writer_header_written_once_on_append(tmp_path: Path):
    p = tmp_path / "trace.tsv"
    with TraceWriter(p) as w:
        w.write(_row(t=1))
    with TraceWriter(p) as w:
        w.write(_row(t=2))
    text = p.read_text().splitlines()
    # header + 2 rows
    assert len(text) == 3
    assert text[0].startswith("t\t")
    assert text[1].split("\t")[0] == "1"
    assert text[2].split("\t")[0] == "2"


def test_writer_handles_extreme_floats(tmp_path: Path):
    p = tmp_path / "trace.tsv"
    row = TraceRow(
        t=1,
        ln_f=1e-15,
        flatness=float("nan"),
        acceptance_rate=0.0,
        min_H_visited=0,
        max_H_visited=0,
        mean_H_visited=0.0,
        n_visited=0,
        in_1overt=True,
        stage_index=99,
    )
    with TraceWriter(p) as w:
        w.write(row)
    back = read_trace(p)
    assert back[0].ln_f == 1e-15
    assert math.isnan(back[0].flatness)
    assert back[0].in_1overt is True


def test_read_trace_rejects_bad_header(tmp_path: Path):
    p = tmp_path / "bad.tsv"
    p.write_text("not\tthe\tright\theader\n1\t2\t3\t4\n")
    with pytest.raises(ValueError):
        read_trace(p)

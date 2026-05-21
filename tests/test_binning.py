"""Tests for `flatwalk.binning`.

The bin scheme is the foundation everything else hangs off of, so cover:
construction validation, edge/center geometry, scalar/array input shapes,
boundary mapping, the inclusive top-edge convention, the in_range/IndexError
contract the driver relies on, and the ABC subclassing contract for ≥2D
extensions.
"""

from __future__ import annotations

import numpy as np
import pytest

from flatwalk.binning import Bin1D, BinScheme

# ---- construction validation ---------------------------------------------------


@pytest.mark.parametrize(
    "low, high, n_bins",
    [
        (0.0, 1.0, 0),
        (0.0, 0.0, 10),
        (1.0, 0.0, 10),
        (float("nan"), 1.0, 10),
        (0.0, float("inf"), 10),
    ],
)
def test_construction_rejects_invalid(low, high, n_bins):
    with pytest.raises(ValueError):
        Bin1D(low, high, n_bins)


def test_construction_minimal_valid():
    b = Bin1D(-1.0, 1.0, 1)
    assert b.n_bins == 1
    assert b.dimensionality == 1
    assert b.width == pytest.approx(2.0)


# ---- edges and centers --------------------------------------------------------


def test_edges_and_centers_shapes_and_endpoints():
    b = Bin1D(-2.0, 3.0, 5)
    assert b.edges.shape == (6,)
    assert b.centers.shape == (5,)
    # endpoints exact
    assert b.edges[0] == -2.0
    assert b.edges[-1] == 3.0
    # uniform width 1.0
    np.testing.assert_allclose(np.diff(b.edges), 1.0)
    # centers are midpoints
    np.testing.assert_allclose(b.centers, b.edges[:-1] + 0.5 * np.diff(b.edges))


def test_ising_l8_geometry():
    """For an Ising L=8 binning we want bin centers on the allowed energies."""
    L = 8
    n_bins = L * L + 1
    low, high = -2 * L * L - 2, 2 * L * L + 2
    b = Bin1D(low, high, n_bins)
    expected_centers = np.arange(-2 * L * L, 2 * L * L + 1, 4, dtype=float)
    np.testing.assert_allclose(b.centers, expected_centers)


# ---- value_to_index ------------------------------------------------------------


def test_value_to_index_basic():
    b = Bin1D(0.0, 10.0, 5)  # width 2: bins [0,2), [2,4), [4,6), [6,8), [8,10]
    assert b.value_to_index(0.0) == 0
    assert b.value_to_index(1.999) == 0
    assert b.value_to_index(2.0) == 1
    assert b.value_to_index(5.0) == 2
    assert b.value_to_index(7.999999) == 3
    # exact top edge maps into top bin
    assert b.value_to_index(10.0) == 4
    # second-to-top edge boundary
    assert b.value_to_index(8.0) == 4


def test_value_to_index_round_trip_with_center():
    b = Bin1D(-5.0, 5.0, 10)
    for i in range(b.n_bins):
        c = b.index_to_center(i)
        assert b.value_to_index(c) == i


def test_value_to_index_scalar_array_inputs():
    b = Bin1D(0.0, 4.0, 4)
    assert b.value_to_index(1.5) == 1
    assert b.value_to_index(np.float64(1.5)) == 1
    assert b.value_to_index(np.array(1.5)) == 1  # 0-d ndarray
    assert b.value_to_index(np.array([1.5])) == 1  # 1-element 1-d
    with pytest.raises(ValueError):
        b.value_to_index(np.array([1.5, 2.5]))  # multi-element rejected


def test_value_to_index_out_of_range_raises():
    b = Bin1D(0.0, 1.0, 10)
    with pytest.raises(IndexError):
        b.value_to_index(-0.01)
    with pytest.raises(IndexError):
        b.value_to_index(1.0 + 1e-9)


# ---- in_range ------------------------------------------------------------------


def test_in_range_inclusive_both_endpoints():
    b = Bin1D(0.0, 1.0, 10)
    assert b.in_range(0.0)
    assert b.in_range(1.0)
    assert b.in_range(0.5)
    assert not b.in_range(-1e-9)
    assert not b.in_range(1.0 + 1e-9)


def test_in_range_matches_value_to_index_safety():
    """The driver's invariant: if in_range is True, value_to_index must not raise."""
    rng = np.random.default_rng(0)
    b = Bin1D(-3.0, 7.0, 41)
    for q in rng.uniform(-3.0, 7.0, size=2000):
        assert b.in_range(q)
        b.value_to_index(q)  # must not raise


# ---- index_to_center -----------------------------------------------------------


def test_index_to_center_bounds():
    b = Bin1D(0.0, 1.0, 4)
    with pytest.raises(IndexError):
        b.index_to_center(-1)
    with pytest.raises(IndexError):
        b.index_to_center(4)
    assert b.index_to_center(0) == pytest.approx(0.125)
    assert b.index_to_center(3) == pytest.approx(0.875)


# ---- ABC contract --------------------------------------------------------------


def test_bin1d_is_a_bin_scheme():
    b = Bin1D(0.0, 1.0, 4)
    assert isinstance(b, BinScheme)


def test_bin_scheme_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BinScheme()  # type: ignore[abstract]

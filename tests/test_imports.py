"""Smoke test: every public symbol imports cleanly."""

from __future__ import annotations


def test_public_symbols_importable():
    import flatwalk

    expected = {
        "Bin1D",
        "BinScheme",
        "ExchangeHandler",
        "ExchangeResult",
        "TraceRow",
        "TraceWriter",
        "Walker",
        "WLConfig",
        "WLDriver",
        "WLResult",
        "read_trace",
    }
    assert expected.issubset(set(flatwalk.__all__))
    for name in expected:
        assert hasattr(flatwalk, name), name


def test_wlconfig_validation():
    from flatwalk import Bin1D, WLConfig
    import pytest

    scheme = Bin1D(0.0, 1.0, 10)
    # valid
    WLConfig(bin_scheme=scheme)
    # invalid flatness threshold
    with pytest.raises(ValueError):
        WLConfig(bin_scheme=scheme, flatness_threshold=1.5)
    # invalid n_check
    with pytest.raises(ValueError):
        WLConfig(bin_scheme=scheme, n_check=0)
    # ln_f_final ordering
    with pytest.raises(ValueError):
        WLConfig(bin_scheme=scheme, ln_f_initial=1e-10, ln_f_final=1e-8)


def test_wldriver_run_minimal_smoke():
    """Tiny smoke run: 10-bin domain, trivial propose, finite max_trials."""
    import numpy as np

    from flatwalk import Bin1D, WLConfig, WLDriver

    scheme = Bin1D(0.0, 10.0, 10)
    cfg = WLConfig(bin_scheme=scheme, n_check=50, ln_f_final=1e-3)
    driver = WLDriver(cfg)

    def propose(state, rng):
        step = rng.choice([-1.0, 1.0])
        return state + step, 0.0

    result = driver.run(
        initial_state=5.0,
        energy_fn=lambda s: 0.0,
        order_parameter_fn=lambda s: s,
        propose_move_fn=propose,
        rng=np.random.default_rng(0),
        max_trials=500,
    )
    assert result.t_total == 500
    assert result.visited.any()

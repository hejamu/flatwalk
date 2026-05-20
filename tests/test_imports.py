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


def test_wldriver_run_is_m2_stub():
    from flatwalk import Bin1D, WLConfig, WLDriver
    import pytest

    driver = WLDriver(WLConfig(bin_scheme=Bin1D(0.0, 1.0, 10)))
    with pytest.raises(NotImplementedError):
        driver.run(
            initial_state=None,
            energy_fn=lambda s: 0.0,
            order_parameter_fn=lambda s: 0.5,
            propose_move_fn=lambda s, r: (s, 0.0),
        )

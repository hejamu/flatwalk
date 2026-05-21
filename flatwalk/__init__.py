"""flatwalk — Wang-Landau (flat-histogram) sampling driver."""

from .binning import Bin1D, BinScheme
from .core import (
    BatchedEnergyFn,
    BatchedOrderParamFn,
    BatchedProposeMoveFn,
    WLConfig,
    WLDriver,
    WLResult,
)
from .diagnostics import TraceRow, TraceWriter, read_trace
from .exchange import ExchangeHandler, ExchangeResult
from .rewl import (
    ReplicaExchangeHandler,
    RewlDriver,
    RewlResult,
    join_g,
    make_windows,
)
from .walker import Walker, WalkerBatch

__all__ = [
    "BatchedEnergyFn",
    "BatchedOrderParamFn",
    "BatchedProposeMoveFn",
    "Bin1D",
    "BinScheme",
    "ExchangeHandler",
    "ExchangeResult",
    "ReplicaExchangeHandler",
    "RewlDriver",
    "RewlResult",
    "TraceRow",
    "TraceWriter",
    "Walker",
    "WalkerBatch",
    "WLConfig",
    "WLDriver",
    "WLResult",
    "join_g",
    "make_windows",
    "read_trace",
]

__version__ = "0.1.0"

"""flatwalk — Wang-Landau (flat-histogram) sampling driver."""

from .binning import Bin1D, BinScheme
from .core import WLConfig, WLDriver, WLResult
from .diagnostics import TraceRow, TraceWriter, read_trace
from .exchange import ExchangeHandler, ExchangeResult
from .walker import Walker

__all__ = [
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
]

__version__ = "0.1.0"

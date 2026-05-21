"""Exact density of states n(E) for the 2D Ising model on an L×L torus.

Algorithm — Beale-style transfer matrix with modular CRT
--------------------------------------------------------

The partition function on an L×L periodic lattice is

    Z(x) = Σ_configs x^{Σ σ_i σ_j over all bonds} = Σ_E n(E) x^{-E/J} = Tr(T^L),

where T is the 2^L × 2^L row-transfer matrix and each T[s, s'] is a single
monomial in x. T^L is then a 2^L × 2^L matrix of polynomials, whose trace
is the integer polynomial Z(x).

The integer coefficients fit in arbitrary-precision ints. We don't compute
them directly (NumPy can't); instead we compute the trace polynomial modulo
several primes and reconstruct integer coefficients via the Chinese
Remainder Theorem. Primes are chosen small enough that float64 matmul
stays exact through every accumulation (≤ 2^53).

Public API
----------

- :func:`beale_g_E(L)` → ``dict[int, int]`` mapping ``E`` (in J units) to
  the exact ``n(E)``.
- :func:`brute_force_g_E(L)` — direct enumeration; only practical for L≤4.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Small-prime helpers
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _primes_below(target: int, count: int) -> list[int]:
    """Return ``count`` largest primes ≤ ``target``."""
    out: list[int] = []
    n = target if target % 2 else target - 1
    while len(out) < count:
        if _is_prime(n):
            out.append(n)
        n -= 2
        if n < 3:
            raise RuntimeError("ran out of primes")
    return out


def _crt(remainders: Iterable[int], moduli: Iterable[int]) -> int:
    """Garner's CRT: solve ``x ≡ r_i (mod m_i)`` for pairwise-coprime ``m_i``."""
    x = 0
    M = 1
    for r, m in zip(remainders, moduli):
        t = ((r - x) % m) * pow(M % m, -1, m) % m
        x = x + M * t
        M *= m
    return x


# ---------------------------------------------------------------------------
# Transfer-matrix construction
# ---------------------------------------------------------------------------


def _build_T_exponent(L: int) -> np.ndarray:
    """Return ``T_exp`` (shape ``(2^L, 2^L)``) of integers in ``[-2L, 2L]``.

    ``T_exp[s, s']`` is the exponent of x in ``T[s, s']``: the contribution
    from intra-row bonds in row ``s`` plus inter-row bonds between ``s`` and
    ``s'``. With each bond contributing ``σ_i σ_j ∈ {-1, +1}`` to the
    exponent (= -E/J units), the exponent is just ``(L − 2 u_intra(s))``
    ``+ (L − 2 u_inter(s, s'))`` where ``u`` counts unsatisfied bonds.
    """
    N = 1 << L
    # Each row state s decodes to L spins via the low L bits; bit b → σ = 2b-1.
    bits = ((np.arange(N)[:, None] >> np.arange(L)[None, :]) & 1).astype(np.int8)
    sigmas = (2 * bits - 1).astype(np.int8)  # (N, L), values in {-1, +1}
    sigmas_shift = np.roll(sigmas, -1, axis=1)
    intra = (sigmas.astype(np.int32) * sigmas_shift.astype(np.int32)).sum(axis=1)
    inter = sigmas.astype(np.int32) @ sigmas.astype(np.int32).T
    return intra[:, None] + inter  # (N, N), int32


def _build_T_poly(L: int) -> np.ndarray:
    """Initial T as a (N, N, 4L+1) float64 array of polynomial coefficients.

    With shift ``+2L``, index ``k`` represents exponent ``k − 2L``, range
    ``[0, 4L]`` ↔ ``[-2L, 2L]``.
    """
    N = 1 << L
    length = 4 * L + 1
    T_exp = _build_T_exponent(L)
    T = np.zeros((N, N, length), dtype=np.float64)
    s, sp = np.indices((N, N))
    T[s, sp, T_exp + 2 * L] = 1.0
    return T


# ---------------------------------------------------------------------------
# Polynomial matrix arithmetic
# ---------------------------------------------------------------------------


def _matmul_poly_mod(A: np.ndarray, B: np.ndarray, mod: int) -> np.ndarray:
    """Polynomial matrix multiply ``A @ B`` modulo ``mod``, float64.

    A: (N, K, dA+1), B: (K, M, dB+1). Result: (N, M, dA+dB+1) with entries
    in ``[0, mod)``.
    """
    _, _, lenA = A.shape
    _, _, lenB = B.shape
    out_len = lenA + lenB - 1
    N = A.shape[0]
    M = B.shape[1]
    C = np.zeros((N, M, out_len), dtype=np.float64)
    for qa in range(lenA):
        Aslice = A[:, :, qa]
        for qb in range(lenB):
            C[:, :, qa + qb] += Aslice @ B[:, :, qb]
    return np.mod(C, mod)


def _matpow_poly_mod(T: np.ndarray, n: int, mod: int) -> np.ndarray:
    """``T^n`` for a polynomial matrix, modulo ``mod``. Repeated squaring."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    base = np.mod(T, mod)
    result: np.ndarray | None = None
    while n > 0:
        if n & 1:
            result = base.copy() if result is None else _matmul_poly_mod(result, base, mod)
        n >>= 1
        if n > 0:
            base = _matmul_poly_mod(base, base, mod)
    assert result is not None
    return result


def _trace_poly(T: np.ndarray) -> np.ndarray:
    """Sum of diagonal entries: shape (poly_length,)."""
    return np.einsum("iik->k", T)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def beale_g_E(L: int, primes: list[int] | None = None) -> dict[int, int]:
    """Exact density of states for the 2D Ising model on an L×L torus.

    Returns a dict ``{E: n(E)}`` where ``E`` is the integer energy in units
    of ``J`` (Hamiltonian ``E = −J Σ σ_i σ_j``) and ``n(E)`` is the integer
    number of configurations at that energy.
    """
    if L < 2:
        raise ValueError("L must be ≥ 2")
    if primes is None:
        # Pick primes p < 2^18 so that 129·N·p² ≲ 2^53 (mantissa-safe).
        # Total configurations on L=8 is 2^64; 4 primes near 2^18 give
        # product ≈ 2^72, comfortably above 2^64 needed for unique reconstruction.
        primes = _primes_below(1 << 18, 4)

    T_init = _build_T_poly(L)
    poly_len_final = L * (4 * L + 1 - 1) + 1  # L·(4L) + 1 = 4L²+1
    shift_final = L * (2 * L)  # L·2L = 2L²

    traces_mod_p: list[np.ndarray] = []
    for p in primes:
        T_p = _matpow_poly_mod(T_init, L, p)
        traces_mod_p.append(_trace_poly(T_p))

    # CRT per coefficient.
    n_E: dict[int, int] = {}
    P = 1
    for p in primes:
        P *= p
    for k in range(poly_len_final):
        remainders = [int(round(float(t[k]))) for t in traces_mod_p]
        x = _crt(remainders, primes) % P
        if x == 0:
            continue
        # Index k represents exponent (k − 2L²); energy is E = −J · exponent.
        # With J = 1, E is an integer.
        E = -(k - shift_final)
        n_E[E] = x

    # Sanity: total count should equal 2^(L²)
    total = sum(n_E.values())
    expected_total = 1 << (L * L)
    if total != expected_total:
        raise RuntimeError(
            f"Beale recursion sanity failed: Σ n(E) = {total}, expected 2^{L * L} = {expected_total}"
        )

    return n_E


def brute_force_g_E(L: int) -> dict[int, int]:
    """Direct enumeration of all 2^(L²) configurations. Practical for L ≤ 4."""
    if L * L > 24:
        raise ValueError(
            f"brute_force_g_E is impractical for L={L} (would enumerate 2^{L * L} configs)"
        )
    n_E: dict[int, int] = {}
    n_spins = L * L
    sites = np.arange(n_spins, dtype=np.int64)
    for s in range(1 << n_spins):
        spins = (((s >> sites) & 1).astype(np.int8) * 2 - 1).reshape(L, L)
        right = np.roll(spins, -1, axis=1)
        down = np.roll(spins, -1, axis=0)
        E = -int((spins * right + spins * down).sum())
        n_E[E] = n_E.get(E, 0) + 1
    return n_E


def log_g_E_array(L: int, n_E: dict[int, int], bin_centers: np.ndarray) -> np.ndarray:
    """Project a ``{E: n(E)}`` dict onto the WL ``bin_centers`` grid, log scale.

    Bins where ``n(E) = 0`` get ``-inf``. Matches the layout produced by
    ``Bin1D`` with bin centers placed on the allowed Ising energies.
    """
    log_g = np.full(len(bin_centers), -np.inf, dtype=np.float64)
    for i, center in enumerate(bin_centers):
        # Allowed Ising energies are integers; centers should land on them.
        E = int(round(center))
        if E in n_E:
            log_g[i] = math.log(n_E[E])
    return log_g

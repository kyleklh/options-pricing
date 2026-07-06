# Tests for numerical stability failure mode #3 — implied vol solver.
# The naive Newton solver silently returns a wrong sigma for deep-OTM /
# near-expiry options; the safeguarded solver recovers it. These tests pin
# both the fix and the bug. See docs/NUMERICS.md entry 3.

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from bs import bs_price
from iv import implied_vol, implied_vol_newton, IVError

S, r = 100.0, 0.05

# grid spanning ATM, OTM/ITM, deep, and near-expiry
GRID = [(100, 1.0), (110, 1.0), (90, 1.0), (120, 0.5), (150, 0.5),
        (200, 1.0), (300, 0.5), (500, 0.1), (120, 0.01)]


def _time_value(price, K, T, flag):
    # the vol-dependent part of the price. when this underflows there is no
    # information left to recover sigma from, so such cases are out of contract.
    disc_K = K * math.exp(-r * T)
    intrinsic = max(S - disc_K, 0.0) if flag == 'call' else max(disc_K - S, 0.0)
    return price - intrinsic


# 1. THE MAIN TEST: round-trip recovery across the grid, calls and puts
def test_round_trip_recovers_sigma():
    for K, T in GRID:
        for flag in ('call', 'put'):
            for sigma_true in (0.1, 0.2, 0.5):
                price = bs_price(S, K, T, r, sigma_true, flag)
                if _time_value(price, K, T, flag) < 1e-10:
                    continue  # unrecoverable by construction, not a solver bug
                recovered = implied_vol(price, S, K, T, r, flag)
                assert abs(recovered - sigma_true) < 1e-6, \
                    f"K={K} T={T} {flag} sigma={sigma_true}: got {recovered}"


# 2. THE HEADLINE REGRESSION: the exact deep-OTM case the naive solver got wrong
def test_deep_otm_case_is_fixed():
    K, T, sigma_true = 300, 0.5, 0.20
    price = bs_price(S, K, T, r, sigma_true, 'call')
    recovered = implied_vol(price, S, K, T, r, 'call')
    assert abs(recovered - sigma_true) < 1e-6


# 3. CHARACTERIZE THE BUG: prove the naive solver really is wrong on that input,
#    so the NUMERICS.md failure claim stays honest and protected
def test_naive_solver_is_wrong_on_deep_otm():
    K, T, sigma_true = 300, 0.5, 0.20
    price = bs_price(S, K, T, r, sigma_true, 'call')
    try:
        naive = implied_vol_newton(price, S, K, T, r, 'call')
    except IVError:
        return  # diverging is also an acceptable "it failed"
    assert abs(naive - sigma_true) > 0.01, \
        f"naive unexpectedly accurate ({naive}); the documented bug is gone"


# 4. INPUT GUARDS
def test_price_below_intrinsic_raises():
    with pytest.raises(IVError):
        implied_vol(-1.0, S, 100, 1.0, r, 'call')


def test_price_above_upper_bound_raises():
    with pytest.raises(IVError):
        implied_vol(S + 1.0, S, 100, 1.0, r, 'call')  # a call cannot be worth > S


def test_unknown_flag_raises():
    with pytest.raises(IVError):
        implied_vol(5.0, S, 100, 1.0, r, 'straddle')

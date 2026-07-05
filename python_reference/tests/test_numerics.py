# Tests for numerical stability failure modes — catastrophic cancellation near expiry.
# Each test either documents a failure or verifies a fix.
# See docs/NUMERICS.md entry 1 for the full write-up.

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from bs import bs_price, bs_price_stable, _d1_d2

K     = 100.0
r     = 0.05
sigma = 0.2

# ── 1. catastrophic cancellation near expiry ──────────────────────────────────
#
# root cause: math.log(S/K) computes ln(1 + (S-K)/K). when S is extremely
# close to K, (S-K)/K is near machine epsilon (~1e-16) and the standard log
# implementation loses all significant digits. dividing by σ√T (also tiny)
# then amplifies the error into the final price.
#
# fix: replace math.log(S/K) with math.log1p((S-K)/K). log1p has a dedicated
# implementation that is accurate to machine precision for small arguments.


def test_atm_d1_converges_to_zero():
    # atm baseline: when S=K, ln(S/K)=0 exactly, so d1→0 as T→0
    # this is well-behaved — no cancellation possible
    for T in [0.1, 0.01, 0.001, 1e-5]:
        d1, d2 = _d1_d2(K, K, T, r, sigma)
        print(f"  T={T:.5f}: d1={d1:.8f}  d2={d2:.8f}")


def test_otm_d1_blows_up_correctly():
    # slightly otm (S=99.99): ln(S/K) is small but nonzero
    # d1 should go to -∞ as T→0, driving N(d1)→0 and price→0
    # this is correct behaviour — documents what we expect near expiry
    for T in [0.01, 0.001, 0.0001, 1e-5]:
        d1, d2 = _d1_d2(99.99, K, T, r, sigma)
        price  = bs_price(99.99, K, T, r, sigma, 'call')
        print(f"  T={T:.5f}: d1={d1:.6f}  price={price:.10f}")


def test_log_precision_loss_at_machine_epsilon():
    # the actual failure: S so close to K that (S-K)/K ≈ machine epsilon
    # math.log(S/K) rounds to 0 — indistinguishable from the atm case
    # math.log1p((S-K)/K) preserves the correct value
    print()
    for S in [K - 1e-8, K - 1e-10, K - 1e-12, K - 1e-14]:
        naive  = math.log(S / K)
        stable = math.log1p((S - K) / K)
        diff   = abs(naive - stable)
        print(f"  S={S:.15f}: log(S/K)={naive:.6e}  log1p={stable:.6e}  diff={diff:.2e}")


def test_stable_matches_naive_for_normal_inputs():
    # the fix must not change results for normal inputs (S far from K)
    # tolerance: 1e-10, well within our precision target
    for S in [80, 90, 100, 110, 120]:
        for T in [0.1, 0.5, 1.0, 2.0]:
            naive  = float(bs_price(S, K, T, r, sigma, 'call'))
            stable = float(bs_price_stable(S, K, T, r, sigma, 'call'))
            assert abs(naive - stable) < 1e-10, f"diverged at S={S} T={T}: {naive} vs {stable}"


def test_stable_gives_valid_price_at_precision_limit():
    # at the precision limit the naive formula is unreliable
    # the stable version must return a non-negative price ≥ intrinsic value
    S = K - 1e-14
    T = 1e-6
    price     = float(bs_price_stable(S, K, T, r, sigma, 'call'))
    intrinsic = max(S - K, 0)
    print(f"\n  stable price at precision limit: {price:.10f}  intrinsic: {intrinsic}")
    assert price >= 0
    assert price >= intrinsic

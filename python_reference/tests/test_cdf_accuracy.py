# Tests for numerical stability failure mode #2 — normal CDF accuracy vs speed.
# Verifies each fast approximation stays within its documented error bound
# against the scipy oracle, including the tails where they are weakest.
# See docs/NUMERICS.md entry 2 for the full write-up.

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import norm
from approximations import norm_cdf_as, norm_cdf_west

# dense grid out to +/-8 sigma: tails are where approximations break down,
# and deep-OTM pricing lives out there
XS = np.linspace(-8.0, 8.0, 2001)  # type: ignore[call-overload]


def _max_abs_err(fn):
    return max(abs(fn(x) - norm.cdf(x)) for x in XS)


def test_symmetry_at_zero():
    # Phi(0) = 0.5. catches the classic upper-tail/lower-tail sign flip.
    # each method held to its OWN bound: A&S is only ~1e-7 accurate, so it
    # does NOT hit 0.5 exactly (it returns 0.50000000052) — that is the
    # accuracy tradeoff, not a bug. West is full precision.
    assert abs(norm_cdf_as(0.0) - 0.5) < 1e-7
    assert abs(norm_cdf_west(0.0) - 0.5) < 1e-14


def test_as_within_bound():
    # Abramowitz-Stegun 7.1.26: documented max abs error ~7.5e-8
    err = _max_abs_err(norm_cdf_as)
    assert err < 1e-7, f"A&S max abs error {err:.2e} exceeds 1e-7"


def test_west_within_bound():
    # West (2004): full double precision
    err = _max_abs_err(norm_cdf_west)
    assert err < 1e-14, f"West max abs error {err:.2e} exceeds 1e-14"


def test_monotonic_and_bounded():
    # a CDF must be nondecreasing and land in [0, 1]
    for fn in (norm_cdf_as, norm_cdf_west):
        vals = [fn(x) for x in XS]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert all(b >= a - 1e-15 for a, b in zip(vals, vals[1:]))

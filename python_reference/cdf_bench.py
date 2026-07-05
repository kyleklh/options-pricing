# Accuracy + speed benchmark for the normal CDF candidates.
# Produces the numbers that fill docs/NUMERICS.md entry 2.
# Not a pytest test — run directly:  python python_reference/cdf_bench.py

import os, sys, timeit
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import norm
from approximations import norm_cdf_as, norm_cdf_west

# accuracy grid: dense, symmetric, out to the tails
ACC_XS = np.linspace(-8.0, 8.0, 4001)

# speed grid: a fixed set of representative x values, hit many times
SPEED_XS = [-2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5]
N_LOOPS = 200_000


def max_abs_err(fn):
    return max(abs(fn(x) - norm.cdf(x)) for x in ACC_XS)


def ns_per_call(fn):
    # time fn over the speed grid, report nanoseconds per single call
    total = timeit.timeit(
        lambda: [fn(x) for x in SPEED_XS],
        number=N_LOOPS,
    )
    return total / (N_LOOPS * len(SPEED_XS)) * 1e9


def scipy_scalar(x):
    # scipy scalar call, the honest baseline for a per-option hot path
    return float(norm.cdf(x))


def main():
    candidates = [
        ("scipy norm.cdf", scipy_scalar),
        ("A&S 7.1.26",     norm_cdf_as),
        ("West 2004",      norm_cdf_west),
    ]

    print(f"{'method':<16} {'max abs err':>14} {'ns/call':>12} {'speedup':>10}")
    print("-" * 56)

    baseline_ns = None
    for name, fn in candidates:
        err = max_abs_err(fn)
        ns = ns_per_call(fn)
        if baseline_ns is None:
            baseline_ns = ns
        speedup = baseline_ns / ns
        err_str = "0 (oracle)" if name.startswith("scipy") else f"{err:.2e}"
        print(f"{name:<16} {err_str:>14} {ns:>10.1f}   {speedup:>8.1f}x")


if __name__ == "__main__":
    main()

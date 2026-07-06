# Implied volatility solver. Inverts Black-Scholes for sigma given a market price.
#
# This module intentionally ships the NAIVE Newton-Raphson version first, so the
# failure modes in docs/NUMERICS.md entry 3 are reproducible. The safeguarded
# version is added in the fix step.

import math
from bs import bs_price, bs_vega


class IVError(RuntimeError):
    """Raised when the solver fails to converge."""
    pass


def _no_arb_bounds(S, K, T, r, flag):
    """No-arbitrage price bounds (lower, upper) for a European option.

    A finite implied vol exists only if the price sits strictly inside these.
    Both bounds depend on option type, so branch on flag.
    """
    disc_K = K * math.exp(-r * T)
    if flag == 'call':
        return max(S - disc_K, 0.0), S
    elif flag == 'put':
        return max(disc_K - S, 0.0), disc_K
    raise IVError(f"unknown flag {flag!r}, expected 'call' or 'put'")


def implied_vol_newton(price, S, K, T, r, flag, sigma0=None, tol=1e-8, max_iter=100):
    """Plain Newton-Raphson on f(sigma) = bs_price(sigma) - price.

    Newton step: sigma <- sigma - f(sigma) / vega(sigma).

    No safeguard on the vega division. When vega is tiny (deep OTM, or near
    expiry) the step explodes and this either overshoots to a nonsense sigma
    or never converges. That is the bug we document.
    """

    if sigma0 is not None:
        sigma = sigma0
    else:
        sigma = math.sqrt(2 * math.pi / T) * price / S  # initial guess from Brenner-Subrahmanyam
        sigma = min(max(sigma, 1e-4), 5.0)  # clamp to [1e-4, 5.0]

    diff = float('nan')

    lower, upper = _no_arb_bounds(S, K, T, r, flag)
    if price <= lower or price >= upper:
        raise IVError(
            f"price {price} is outside the no-arbitrage bounds "
            f"[{lower:.3f}, {upper:.3f}]"
        )

    for i in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, flag) - price
        if abs(diff) < tol:
            return sigma
        vega = bs_vega(S, K, T, r, sigma)
        sigma = sigma - diff / vega
    raise IVError(
        f"did not converge after {max_iter} iters: last sigma={sigma!r}, "
        f"last price error={diff:.3e}"
    )


def implied_vol(price, S, K, T, r, flag, tol=1e-8, max_iter=100):
    """Safeguarded solver: bracketed Newton with a bisection fallback.

    Recovers sigma across the full input domain, including the deep-OTM and
    near-expiry regimes where implied_vol_newton silently returns a wrong
    value. This is the fix documented in docs/NUMERICS.md entry 3.
    """
    # Step 0: reject prices with no finite implied vol.
    lower, upper = _no_arb_bounds(S, K, T, r, flag)
    if price <= lower or price >= upper:
        raise IVError(
            f"price {price} is outside the no-arbitrage bounds "
            f"[{lower:.3f}, {upper:.3f}]"
        )

    # Step 1: Brenner-Subrahmanyam seed, clamped to a sane band.
    sigma = math.sqrt(2 * math.pi / T) * price / S
    sigma = min(max(sigma, 1e-4), 5.0)

    # Step 2: bracket the root. price is monotone increasing in sigma, so
    # [lo, hi] with bs_price(lo) <= price <= bs_price(hi) traps the solution.
    # bs_price(lo) -> lower bound as sigma -> 0, and Step 0 guaranteed
    # price > lower, so the low side needs no search. Push hi up until it
    # clears the target price.
    lo, hi = 1e-8, 5.0
    while bs_price(S, K, T, r, hi, flag) < price:
        hi *= 2.0
        if hi > 1e4:
            raise IVError(f"could not bracket root; price {price} too high")

    # keep the seed inside the bracket before handing off to Newton
    if not (lo < sigma < hi):
        sigma = 0.5 * (lo + hi)

    # Step 3: safeguarded Newton. Take a Newton step when it is well-behaved,
    # otherwise bisect the bracket. Converge on bracket width, not price, so a
    # flat deep-OTM price curve cannot trigger false convergence.
    vega_floor = 1e-8
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, flag) - price

        # tighten the bracket from the sign of diff (price rises with sigma)
        if diff > 0:
            hi = sigma      # overshot: root is below
        else:
            lo = sigma      # undershot: root is above

        # converge on bracket width: pins sigma regardless of price scale
        if (hi - lo) < tol:
            return 0.5 * (lo + hi)

        # try a Newton step, but only if vega is healthy enough to divide by.
        # check the floor first so we never form diff / 0 (which would nan).
        vega = bs_vega(S, K, T, r, sigma)
        if vega < vega_floor:
            sigma = 0.5 * (lo + hi)         # vega collapsed: bisect
        else:
            newton = sigma - diff / vega
            if newton <= lo or newton >= hi:
                sigma = 0.5 * (lo + hi)     # Newton left the bracket: bisect
            else:
                sigma = newton              # Newton is safe: take the fast step

    raise IVError(
        f"did not converge after {max_iter} iters: bracket=[{lo}, {hi}]"
    )

# Numerical Stability Notes

Each entry documents a real failure mode: the failing input, the wrong output, the root cause, and the fix.

## 1. Catastrophic cancellation in d1/d2 near expiry
**Status:** Fixed. **Severity:** Low for live market data, but a real IEEE 754 failure mode and the standard textbook example of catastrophic cancellation in this formula.

### The bug

The Black-Scholes d1 term is:

```
d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
```

The reference implementation computed `ln(S/K)` with `math.log(S / K)`. This holds up for almost every input and breaks down only when `S` is extremely close to `K`. In that regime `S / K` is extremely close to `1.0`, and `ln(x)` near `x = 1` is a classic catastrophic cancellation site. The true value is on the order of `(S-K)/K`, but computing it as `log(1 + tiny_number)` lets the "1 +" absorb the tiny number in rounding before `log` ever sees it.

### Reproducing it

The comparison below is `math.log(S/K)` against `math.log1p((S-K)/K)`, which is mathematically identical but implemented to avoid the cancellation, at increasing closeness to the strike:

| S vs K | (S-K)/K | `log(S/K)` (naive) | `log1p((S-K)/K)` (stable) |
|---|---|---|---|
| K - 1e-8  | -1.000e-10 | -9.999990e-11 | -9.999994e-11 |
| K - 1e-10 | -1.000e-12 | -9.999779e-13 | -1.000018e-12 |
| K - 1e-12 | -9.948e-15 | -9.992007e-15 | -9.947598e-15 |
| K - 1e-14 | -1.421e-16 | -1.110223e-16 | -1.421085e-16 |

By the last row the naive version is off by about 22% in relative terms on the log term alone. That looks harmless in isolation, because the value is still tiny. The damage comes from the division in `d1`: dividing this term by `sigma*sqrt(T)`, where `sqrt(T)` shrinks as expiry approaches, amplifies a fixed absolute rounding error without bound as `T -> 0`.

Pushing `T` down while holding `S = K - 1e-14` shows the amplification directly. The call price should be smooth and non-negative:

| T | price (naive `log`) | price (stable `log1p`) | relative error |
|---|---|---|---|
| 1e-6  | 7.981346e-03 | 7.981346e-03 | ~0 |
| 1e-20 | 7.978755e-10 | 7.978755e-10 | ~0 |
| 1e-29 | 2.131628e-14 | 1.421085e-14 | 50% |
| 1e-31 | **-4.440892e-16** | -2.220446e-16 | invariant violated |

At `T = 1e-31` the naive formula returns a **negative option price**, which is impossible for a call and violates the no-arbitrage floor `price >= max(S - K, 0)`. The stable version is also at the edge of float64 precision here, and the claim is not that `log1p` makes this extreme case exact. The claim is narrower: the naive version breaks a basic correctness invariant that the stable version preserves, and it breaks it earlier and more severely as inputs get more extreme.

### Why the error amplifies

In exact arithmetic `d1` stays well-behaved as `T -> 0`, because the log term and its denominator both scale with `sqrt(T)` once the drift term is accounted for. The rounding error from `log(S/K)` does not follow that scaling. It is a fixed artifact of representing `S/K` in float64 near `1.0`, so dividing a roughly constant error by a shrinking `sigma*sqrt(T)` drives the relative error in `d1` upward as `T` shrinks, even though the true `d1` converges to a finite limit. `log1p` sidesteps this: it uses a dedicated series expansion for arguments near zero and never forms `1 + tiny_number` as an intermediate that could swallow the small term.

### Does this matter for realistic inputs?

For market-realistic inputs (`S` and `K` differing by even a fraction of a cent, `T` down to a one-minute option), the naive and stable formulas agree to more than 12 significant digits. This bug will not visibly misprice a real option chain. It matters in two places that a pricing engine has to handle:

1. **Numerical Greeks.** Computing delta or gamma by finite-difference bumping `S` by a small epsilon can put the bumped spot arbitrarily close to `K` by construction, which is exactly the regime where this fires.
2. **Boundary robustness.** Phase 0's precision target is 1e-10 against the reference grid across boundary cases, and a non-negative price is an invariant worth holding unconditionally rather than only in the common case.

### The fix

Replace `math.log(S/K)` with `math.log1p((S-K)/K)` in `_d1_d2`:

```python
def _d1_d2_stable(S, K, T, r, sigma):
    # log1p((S-K)/K) computes ln(S/K) accurately when S is close to K
    log_sk = math.log1p((S - K) / K)
    d1 = (log_sk + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2
```

`bs_price_stable` calls this in place of `_d1_d2`.

### Proof

`test_numerics.py` covers this end to end:

- `test_log_precision_loss_at_machine_epsilon` reproduces the table above, showing `log` and `log1p` diverging as `S -> K`.
- `test_stable_matches_naive_for_normal_inputs` asserts the fix changes nothing to within 1e-10 across the full fixture grid of normal, non-boundary inputs, confirming it is targeted rather than a behavior change.
- `test_stable_gives_valid_price_at_precision_limit` asserts the production-relevant invariant: at `S = K - 1e-14`, `T = 1e-6`, the stable price is non-negative and at or above intrinsic value.

All three pass under `pytest test_numerics.py`.


## 2. Normal CDF accuracy vs. speed tradeoff
**Status:** Characterized. **Severity:** Not a bug, a design decision. The normal CDF is the hottest function in the engine, so which implementation it uses is a real accuracy-vs-speed tradeoff worth measuring rather than guessing.

### The setup

Every Black-Scholes price evaluates the standard normal CDF twice, at `d1` and `d2`, and the Greeks call it again. In a live pricer this function runs more than anything else, so its cost dominates and its error propagates into every quoted number. The reference implementation uses `scipy.stats.norm.cdf`, which wraps a full-precision `erf` and is accurate to machine epsilon. It is also slow to call one value at a time, and it is not something the Rust core can inline on a no-allocation hot path. So the question is what a fast closed-form approximation gives up.

Two candidates, both in `python_reference/approximations.py`:

- **Abramowitz & Stegun 7.1.26.** A degree-5 polynomial in `t = 1/(1 + 0.2316419*x)` multiplied by the normal pdf. Cheap, and its error is bounded at roughly `7.5e-8`.
- **West (2004).** A Hart-style rational approximation, a ratio of two polynomials with a continued-fraction tail. It reaches full float64 precision.

### Measured results

`python_reference/cdf_bench.py` compares each against the scipy oracle over a grid from -8 to 8 sigma, and times a single scalar call:

| method | max abs error | ns/call | speedup vs scipy |
|---|---|---|---|
| scipy `norm.cdf` | 0 (oracle) | 23350.4 | 1.0x |
| A&S 7.1.26 | 7.45e-08 | 241.1 | 96.9x |
| West 2004 | 2.22e-16 | 282.9 | 82.5x |

One caveat on the speed column. The scipy figure is a scalar Python call, and most of its 23 microseconds is dispatch overhead, not the `erf` arithmetic itself. A vectorized scipy call over a whole array amortizes that overhead and is far faster per value. So the table is not saying `erf` is 100 times slower than a polynomial. It is saying that a per-option scalar call into scipy costs about 100 times more than an inlined closed form in this reference environment, which is the pattern that carries over to the fast core: the CDF has to be a closed form the compiler can inline, not a library call per option.

### Which one to use

The interesting part is that West is barely more expensive than A&S, 283 ns against 241 ns, while buying back eight extra digits of accuracy. That flips the usual expectation. The lossy approximation is supposed to be the fast one, and here it is not meaningfully faster than the exact one.

That matters because the `7.45e-8` error in A&S is invisible in a price but not in a Greek. Delta, gamma, and the rest are computed by bumping an input by a small step `h` and differencing, then dividing by `h`. A fixed `7.5e-8` error in the CDF divided by an `h` on the order of `1e-5` becomes an error on the order of `1e-2` in the Greek, which is large enough to matter. Since West removes that error for almost no extra cost, it is the better default, and A&S is kept only as the illustration of the tradeoff. This is also why production pricers use a West or Cody style approximation rather than a truncated polynomial.

### Proof

`test_cdf_accuracy.py` holds each candidate to its own bound against scipy across the grid:

- `test_symmetry_at_zero` pins `Phi(0)` for both. West hits `0.5` to full precision, A&S returns `0.50000000052`, which documents the tradeoff rather than treating it as a failure.
- `test_as_within_bound` asserts A&S stays under `1e-7`.
- `test_west_within_bound` asserts West stays under `1e-14`.
- `test_monotonic_and_bounded` asserts both stay in `[0, 1]` and never decrease, which catches the upper-tail sign flip that is the classic bug in the West formula.

All four pass under `pytest test_cdf_accuracy.py`.

## 3. Implied volatility solver failure for deep OTM options

TODO

## 4. Binomial tree oscillation (odd vs. even step counts)

TODO

## 5. CFL instability in explicit PDE schemes

TODO

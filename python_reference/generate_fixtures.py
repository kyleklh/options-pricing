import json, itertools, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from bs import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho

spots    = [80, 90, 100, 110, 120]
expiries = [0.1, 0.25, 0.5, 1.0, 2.0]
vols     = [0.1, 0.2, 0.3, 0.5]
K, r     = 100, 0.05

rows = []
for S, T, sigma, flag in itertools.product(spots, expiries, vols, ['call', 'put']):
    rows.append({
        "S": S, "K": K, "T": T, "r": r, "sigma": sigma, "flag": flag,
        "price": round(float(bs_price(S, K, T, r, sigma, flag)), 10),
        "delta": round(float(bs_delta(S, K, T, r, sigma, flag)), 10),
        "gamma": round(float(bs_gamma(S, K, T, r, sigma)), 10),
        "vega":  round(float(bs_vega(S, K, T, r, sigma)), 10),
        "theta": round(float(bs_theta(S, K, T, r, sigma, flag)), 10),
        "rho":   round(float(bs_rho(S, K, T, r, sigma, flag)), 10),
    })

out = os.path.join(os.path.dirname(__file__), 'fixtures', 'grid.json')
with open(out, 'w') as f:
    json.dump(rows, f, indent=2)

print(f"Generated {len(rows)} rows -> {out}")

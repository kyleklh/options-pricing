# Fast standard-normal CDF approximations, for the accuracy-vs-speed study.
# The oracle stays scipy (full-precision erf); these are the candidates a fast
# core would actually use on the hot path. See docs/NUMERICS.md entry 2.

import math

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(x):
    # standard normal pdf
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def norm_cdf_as(x):
    """Abramowitz & Stegun 7.1.26. Max abs error ~7.5e-8, very fast.

    Horner-nested degree-5 polynomial in t = 1/(1+px), times the pdf.
    Only valid for x >= 0; negative side handled by symmetry.
    """
    if x < 0:
        return 1.0 - norm_cdf_as(-x)

    p = 0.2316419
    t = 1.0 / (1.0 + p * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
              + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - _phi(x) * poly


def norm_cdf_west(x):
    """Graeme West (2004), "Better Approximations to Cumulative Normal
    Functions." Hart-style rational approximation, full double precision
    (~1e-16) and faster than a naive erf. The production-grade choice.

    c accumulates the UPPER tail P(Z > |x|); the final return flips it.
    """
    z = abs(x)
    if z > 37.0:
        c = 0.0
    else:
        e = math.exp(-0.5 * z * z)
        if z < 7.07106781186547:
            n = (((((3.52624965998911e-02 * z + 0.700383064443688) * z
                   + 6.37396220353165) * z + 33.912866078383) * z
                   + 112.079291497871) * z + 221.213596169931) * z + 220.206867912376
            d = ((((((8.83883476483184e-02 * z + 1.75566716318264) * z
                   + 16.064177579207) * z + 86.7807322029461) * z
                   + 296.564248779674) * z + 637.333633378831) * z
                   + 793.826512519948) * z + 440.413735824752
            c = e * n / d
        else:
            f = z + 1.0 / (z + 2.0 / (z + 3.0 / (z + 4.0 / (z + 0.65))))
            c = e / (2.506628274631 * f)
    return c if x < 0 else 1.0 - c

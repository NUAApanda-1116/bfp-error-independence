"""Verify the lattice (dual) identity used in Sec. 3.6.1.

For each distribution we form the marginal X = |x_1| with density f_X and
    delta(s) = E[RN(X/s) - X/s],   gamma(s) = E[(RN(X/s) - X/s)^2],
computed by 1-D quadrature, and test at integer indices t = log2(s) + L_m:

  A = K_t + K_{t+1}/4 + K_{t+2}/16 + ...          (K = s^2 delta^2)
  B = G_t + G_{t+1}/2 + G_{t+2}/4 + ...           (G = s^2 gamma)

and checks the two candidate forms of the two-point interpolation term
    T1 = (K_{t+1}/4 + K_{t+2}/16 + ...) / A
    T2 = (G_{t+1}/2 + G_{t+2}/4 + ...) / B
and the resulting exact identity  ratio = A/B.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

L_M = 2.0


def quadrature(f, a, b, n=400):
    x, w = leggauss(n)
    xm = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(xm))


def lognormal(absx):
    # |X| for X ~ LogNormal(0,1): f = sqrt(2/pi) exp(-(ln x)^2/2) / x
    x = np.maximum(absx, 1e-300)
    return np.sqrt(2 / np.pi) * np.exp(-0.5 * np.log(x) ** 2) / x


def halfnormal(absx):
    return np.sqrt(2 / np.pi) * np.exp(-0.5 * absx ** 2)


def pareto4(absx):
    # |X| for a Pareto(4) with support x>=1
    return np.where(absx >= 1.0, 4.0 * absx ** (-5.0), 0.0)


def g(u):
    return np.rint(u) - u


def delta_gamma(f, s, hi=400.0):
    """1-D quadrature; tail beyond `hi` for lognormal/pareto is treated by
    the same rule (contribution is negligible but we integrate to hi)."""
    def integ(w, xx):
        return f(xx) * w
    # delta
    def fd(x):
        return g(x / s) * f(x)
    def fg(x):
        return g(x / s) ** 2 * f(x)
    return quadrature(fd, 0, hi, 2000), quadrature(fg, 0, hi, 2000)


def series(f, t_lo, t_hi, Lm=L_M):
    """Return arrays K_t, G_t for integer t in [t_lo, t_hi]."""
    K, G = {}, {}
    for t in range(t_lo, t_hi + 1):
        s = 2.0 ** (t - Lm)
        d, gam = delta_gamma(f, s)
        K[t] = s * s * d * d
        G[t] = s * s * gam
    return K, G


def check(name, f, t_lo, t_hi):
    K, G = series(f, t_lo, t_hi)
    print(f"\n=== {name} ===")
    print(f"{'t':>3} {'s':>8} {'delta(s)':>12} {'gamma(s)':>12} "
          f"{'A':>12} {'B':>12} {'A/B':>10} {'T1':>10} {'T2':>10}")
    for t in range(t_lo, t_hi - 3):
        A = sum(K[t + j] * 4.0 ** (-j) for j in range(0, 8))
        B = sum(G[t + j] * 2.0 ** (-j) for j in range(0, 8))
        A1 = sum(K[t + j] * 4.0 ** (-j) for j in range(1, 8))
        B1 = sum(G[t + j] * 2.0 ** (-j) for j in range(1, 8))
        s = 2.0 ** (t - L_M)
        # delta at s for reporting
        d, gam = delta_gamma(f, s)
        print(f"{t:>3} {s:>8.4f} {d:>12.6f} {gam:>12.6f} "
              f"{A:>12.4e} {B:>12.4e} {A/B:>10.6f} {A1:>10.6f} {B1:>10.6f}")


if __name__ == "__main__":
    check("half-normal", halfnormal, -4, 8)
    check("lognormal", lognormal, -4, 8)
    check("pareto(4)", pareto4, -4, 8)

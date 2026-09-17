"""Accurate delta(s), gamma(s) by per-unit-interval Gauss-Legendre quadrature,
plus the numerical test of the lattice identity of Sec. 3.6.1.

delta(s) = E[RN(X/s)-X/s],  gamma(s) = E[(RN(X/s)-X/s)^2].

Change variable x = s*u, u in [k,k+1): g_-(u) = k-u (u<k+1/2), g_+(u)=k+1-u.
delta(s) = s * sum_k int_{k}^{k+1} g(u) f_X(su) du,  similarly for gamma.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

L_M = 2
NG = 12  # Gauss nodes per unit u-interval


def make_rule(kmax):
    """Nodes/weights for u in [0, kmax], per unit interval."""
    xg, wg = leggauss(NG)                      # on [-1,1]
    u = 0.5 * (xg[None, :] + 1.0) + np.arange(kmax)[:, None]   # (kmax,NG) in [k,k+1]
    w = 0.5 * wg[None, :] * np.ones((kmax, 1))
    k = np.arange(kmax)[:, None] * np.ones((1, NG))
    gm = np.where(u < k + 0.5, k - u, k + 1.0 - u)             # g(u)
    return u, w, gm


def dens(name, x):
    x = np.asarray(x, dtype=float)
    if name == "half-normal":
        return np.sqrt(2 / np.pi) * np.exp(-0.5 * x ** 2)
    if name == "lognormal":
        xx = np.maximum(x, 1e-300)
        return np.sqrt(2 / np.pi) * np.exp(-0.5 * np.log(xx) ** 2) / xx
    if name == "pareto4":
        return np.where(x >= 1.0, 4.0 * x ** (-5.0), 0.0)
    raise ValueError(name)


def delta_gamma(name, s, umax=None):
    if umax is None:
        umax = {"half-normal": 30.0 / s, "lognormal": 60.0 / s,
                "pareto4": 60.0 / s}[name]
    kmax = int(umax) + 1
    u, w, gm = make_rule(kmax)
    f = dens(name, s * u)
    fs = f * s                      # dx = s du
    delta = np.sum(w * gm * fs)
    gam = np.sum(w * gm ** 2 * fs)
    return delta, gam


def series(name, t_lo, t_hi):
    K, G, D = {}, {}, {}
    for t in range(t_lo, t_hi + 1):
        s = 2.0 ** (t - L_M)
        d, g = delta_gamma(name, s)
        D[t] = (d, g)
        K[t] = s * s * d * d
        G[t] = s * s * g
    return K, G, D


def report(name, t_lo, t_hi, J=8):
    K, G, D = series(name, t_lo - 2, t_hi + J)
    print(f"\n=== {name} ===  (L_m={L_M})")
    print(f"{'t':>4} {'s':>9} {'delta(s)':>12} {'gamma(s)':>10} {'delta/s':>9} "
          f"{'Kt/Gt':>9} {'A1/B':>9} {'A1/A':>8} {'B1/B':>8} {'w_pred':>8}")
    for t in range(t_lo, t_hi + 1):
        A = sum(K[t + j] * 4.0 ** (-j) for j in range(-2, J))
        B = sum(G[t + j] * 2.0 ** (-j) for j in range(-2, J))
        A1 = sum(K[t + j] * 4.0 ** (-j) for j in range(1, J))
        B1 = sum(G[t + j] * 2.0 ** (-j) for j in range(1, J))
        wp = G[t] / (G[t] + G[t + 1] / 2)
        d, g = D[t]
        print(f"{t:>4} {2.0**(t-L_M):>9.5f} {d:>12.7f} {g:>10.7f} {d/2.0**(t-L_M):>9.5f} "
              f"{K[t]/G[t]:>9.6f} {A1/B:>9.6f} {A1/A:>8.5f} {B1/B:>8.5f} {wp:>8.5f}")


if __name__ == "__main__":
    for nm in ("half-normal", "lognormal", "pareto4"):
        report(nm, -4, 6)

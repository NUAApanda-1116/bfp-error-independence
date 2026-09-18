"""Independent recheck: (a) v5 estimator at small n, (b) delta(s)/gamma(s) dead-zone law."""
import numpy as np

# ---------- (a) reproduce v5-style I_n for lognormal, Lm=4, n=32/128 ----------
def quant_blocks(x, Lm):
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    return np.round(x / step) * step

def infl(make, Lm, n, groups=20, nb=1000, seed=20260917):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(groups):
        x = make(rng, nb * n).reshape(nb, n)
        err = quant_blocks(x, Lm) - x
        S = err.sum(axis=1)
        vals.append((S.var(ddof=1)) / (n * err.var(ddof=1)))
    v = np.array(vals)
    return v.mean(), v.std(ddof=1)/np.sqrt(len(v))

ln = lambda rng, k: rng.lognormal(0.0, 1.0, k)
print("=== v5-style I_n, lognormal (paper tab:infl expects 1.328 at Lm=4 n=32, 3.511 at n=128, 165.025 at n=4096) ===")
for Lm in (2, 4):
    for n in (32, 128):
        m, se = infl(ln, Lm, n)
        print(f"  Lm={Lm} n={n:5d}: I_n = {m:.3f} +- {se:.3f}")

# ---------- (b) delta(s), gamma(s) dead-zone law ----------
# paper: delta(0.125)=-0.00414, delta(0.25)=-0.00841, delta(0.5)=-0.01668, delta(1)=-0.03434
#        => delta(s) = -0.0334 s ; ReLU delta = half of half-normal ; 12*gamma = 1.000 (HN), 0.500 (ReLU)
rng = np.random.default_rng(12345)
N = 40_000_000
big = np.concatenate([rng.standard_normal(N//2 + N % 2), -rng.standard_normal(N//2)])
hn = np.abs(big)
rl = np.maximum(big, 0.0)
print("\n=== dead-zone response functions (paper sec 5.6 'Two structural facts') ===")
for s in (0.125, 0.25, 0.5, 1.0):
    for name, x in (("half-normal", hn), ("ReLU", rl)):
        g = np.round(x/s)*s - x
        d = g.mean()/s          # delta(s) = E[RN(x/s)-x/s]
        gam = (g*g).mean()/s**2
        print(f"  s={s:<6} {name:<12} delta={d:+.5f}  delta/s={d/s:+.5f}  12*gamma={12*gam:.4f}  gamma={gam:.6f}")

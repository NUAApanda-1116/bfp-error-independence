"""第四轮复核（独立数值检查）

注意：本脚本的 Test B（退化模型求值）用 1-CDF 之差计算格内质量，在重尾/大 n 下会
受相消误差与下溢影响，仅供定性参考；精确的总体 I_n 求值见 repro/model_pop.py
（生存函数之差 + 两遍方差）与 repro/pareto_asymptotic.py（60 位十进制）。
Test A 与 Test C 不受此影响。


A. 引理 lem:cond 的条件密度 eq:conddens 是否正确（两个分支的质量与形状）
B. 退化模型的【总体】I_n：把 n 推到 1e16，看 I_n -> 1 / -> const / -> inf
C. 格级数 eq:pair 的前置因子（4^{-j} 权）
"""
import math
import numpy as np

SQ2 = math.sqrt(2.0)
INV_SQRT2PI = 1.0 / math.sqrt(2 * math.pi)

# ---------------------------------------------------------------- |x| 的分布
E_X2 = {"halfnormal": 1.0, "relu": 0.5, "lognormal": math.e ** 2, "pareto4": 2.0}
E_X = {"halfnormal": math.sqrt(2 / math.pi), "relu": math.sqrt(1 / (2 * math.pi)),
       "lognormal": math.exp(0.5), "pareto4": 4.0 / 3.0}
# 截断点：尾部质量 ~1e-13
T_TRUNC = {"halfnormal": 7.5, "relu": 7.5, "lognormal": 3000.0, "pareto4": 3000.0}

def _phi(t):
    return math.exp(-0.5 * t * t) * INV_SQRT2PI

def cdf(name, x):
    """F_{|x|}(x)"""
    if x <= 0:
        return 0.0
    if name == "halfnormal":
        return math.erf(x / SQ2)
    if name == "relu":                     # x=max(N,0)：P(X<=0)=1/2
        return 0.5 * (1.0 + math.erf(x / SQ2))
    if name == "lognormal":
        return 0.5 * (1.0 + math.erf(math.log(x) / SQ2))
    if name == "pareto4":
        return 0.0 if x < 1 else 1.0 - x ** -4.0
    raise ValueError(name)

def sf(name, x):
    return 1.0 - cdf(name, x)

def G(name, x):
    """int_0^x t f_{|x|}(t) dt"""
    x = max(x, 0.0)
    if name == "halfnormal":
        return 2.0 * (INV_SQRT2PI - _phi(x))
    if name == "relu":
        return INV_SQRT2PI - _phi(x)
    if name == "lognormal":
        if x <= 0:
            return 0.0
        return 2.0 * 0.5 * (1.0 + math.erf(math.log(x) / SQ2))
    if name == "pareto4":
        if x <= 1.0:
            return 0.0
        return (4.0 / 3.0) * (1.0 - x ** -3.0)
    raise ValueError(name)

def shape(name, s):
    """delta(s)=E[RN(u)-u], gamma(s)=E[(RN(u)-u)^2], u=x/s, x~|x|。按舍入单元精确求和。"""
    T = T_TRUNC[name]
    M = int(math.ceil(T / s)) + 2
    E_RN = 0.0        # sum m P_m
    E_RN2 = 0.0       # sum m^2 P_m
    E_uRN = 0.0       # sum m (G(b)-G(a)) / s
    for m in range(0, M + 1):
        lo = s * (m - 0.5)
        hi = s * (m + 0.5)
        Pm = cdf(name, hi) - cdf(name, lo)
        if Pm <= 0.0 and m > 1:
            break
        E_RN += m * Pm
        E_RN2 += m * m * Pm
        E_uRN += m * (G(name, hi) - G(name, lo)) / s
    Ex = E_X[name]
    delta = E_RN - Ex / s
    gamma = E_RN2 - 2.0 * E_uRN + E_X2[name] / (s * s)
    return delta, gamma

# ---------------------------------------------------------------- A. 条件密度
def test_A(n=8, N=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    X = np.abs(rng.standard_normal((N, n)))
    xi = np.floor(np.log2(X.max(axis=1))).astype(int)
    x1 = X[:, 0]
    print(f"\n=== A. p(x_1 | xi=k) 检验：half-normal，n={n}，N={N:,} 块 ===")
    print(f"{'k':>3}{'P(xi=k)':>12}{'实测 P(|x1|<a)':>16}{'正确式':>10}{'第三轮式':>10}{'第二轮式':>10}")
    for k in (1, 2, 3):
        sel = xi == k
        a, b = 2.0 ** k, 2.0 ** (k + 1)
        Fa, Fb = cdf("halfnormal", a), cdf("halfnormal", b)
        Pk = Fb ** n - Fa ** n
        emp = float(np.mean(np.abs(x1[sel]) < a))
        right = Fa * (Fb ** (n - 1) - Fa ** (n - 1)) / Pk
        r3 = Fa ** 2 * (Fb ** (n - 1) - Fa ** (n - 1)) / Pk
        print(f"{k:>3}{sel.mean():>12.5f}{emp:>16.4f}{right:>10.4f}{r3:>10.4f}"
              f"{Fa*(Fb**(n-1)-Fa**(n-1))/Pk:>10.4f}")
    print("\n  第二支形状：E[x1 | a<=|x1|<b]")
    gl_x, gl_w = np.polynomial.legendre.leggauss(500)
    for k in (2, 3):
        sel = (xi == k) & (np.abs(x1) >= 2.0 ** k)
        a, b = 2.0 ** k, 2.0 ** (k + 1)
        xg = 0.5 * (b - a) * gl_x + 0.5 * (b + a)
        wg = 0.5 * (b - a) * gl_w
        Fg = np.array([cdf("halfnormal", t) for t in xg])
        fg = 2.0 * np.exp(-0.5 * xg ** 2) * INV_SQRT2PI
        m_paper = np.sum(wg * xg * fg * Fg ** (n - 1)) / np.sum(wg * fg * Fg ** (n - 1))
        m_right = np.sum(wg * xg * fg) / np.sum(wg * fg)
        print(f"    k={k}: 实测 {np.mean(x1[sel]):.4f} | 论文式 f(x)F(x)^(n-1): {m_paper:.4f}"
              f" | 正确式 f(x)F(b)^(n-1): {m_right:.4f}")

# ---------------------------------------------------------------- C. 格级数
def test_C():
    print("\n=== C. 格级数前置换算因子 (1-lam/2)/(1-lam^2/8) 的核对 ===")
    print(f"{'lam':>6}{'直接求和':>12}{'论文式':>12}{'正确式':>12}")
    for lam in (1.0, 1.4, 1.8, 1.999):
        j = np.arange(0, 20000)
        num = np.sum(4.0 ** -j * (lam / 2) ** (2 * j))
        den = np.sum(2.0 ** -j * (lam / 2) ** j)
        paper = (1 - lam / 2) / (1 - lam ** 2 / 8)
        right = (1 - lam / 4) / (1 - lam ** 2 / 16)
        print(f"{lam:>6.3f}{num/den:>12.4f}{paper:>12.4f}{right:>12.4f}")

# ---------------------------------------------------------------- B. 总体 I_n
def model_I(name, Lm, n):
    # xi 的支撑：n*S(2^k) ~ 1e-4 处为右端
    lo, hi = 0.0, 80.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if n * sf(name, 2.0 ** mid) > 1e-4:
            lo = mid
        else:
            hi = mid
    khi = int(math.ceil(hi)) + 3
    ks, ps = [], []
    for k in range(khi - 120, khi + 1):
        a, b = 2.0 ** k, 2.0 ** (k + 1)
        p = math.exp(-n * sf(name, b)) - math.exp(-n * sf(name, a))
        if p > 1e-16:
            ks.append(k); ps.append(p)
    ps = np.array(ps); ps = ps / ps.sum()
    Em = Em2 = Es2g = 0.0
    for k, p in zip(ks, ps):
        s = 2.0 ** (k - Lm)
        d, g = shape(name, s)
        Em += p * s * d
        Em2 += p * (s * d) ** 2
        Es2g += p * s * s * g
    var_m = Em2 - Em ** 2
    var_e = Es2g - Em ** 2
    return dict(I=1.0 + (n - 1) * var_m / var_e, Em=Em, var_e=var_e, var_m=var_m,
                Es2g=Es2g, ks=(ks[0], ks[-1]))

if __name__ == "__main__":
    print("=== 形状函数自检（对照论文第 5.6 节实测值）===")
    for nm, sv, ref_d, ref_g in (("halfnormal", 1.0, -0.03434, 1 / 12),
                                 ("relu", 1.0, -0.01668, 1 / 24),
                                 ("lognormal", 4.0, -0.11221, None),
                                 ("pareto4", 1.0, -0.0985, None),
                                 ("pareto4", 2.0, +0.3480, None),
                                 ("pareto4", 4.0, -0.2699, None)):
        d, g = shape(nm, sv)
        gs = f"{g:.5f}" + (f" (论文 12g={12*ref_g:.3f})" if ref_g else "")
        print(f"  {nm:>10} s={sv:<5} delta={d:+.5f} (论文 {ref_d:+.5f})   gamma={gs}")

    test_A()
    test_C()

    print("\n=== B. 退化模型的【总体】I_n（无样本噪声）===")
    print("先校验论文表 tab:model / tab:bounded 的模型列：")
    for nm, Lm, ref in (("lognormal", 2, 176.4593), ("halfnormal", 2, 15.0719),
                        ("relu", 2, 6.3155), ("pareto4", 2, 2887.29)):
        r = model_I(nm, Lm, 4096)
        print(f"  {nm:>10} Lm={Lm} n=4096: I={r['I']:12.4f}  论文模型值={ref:>10.2f}  "
              f"E[m]={r['Em']:+.5f}  Var(e)={r['var_e']:.3e}  xi in [{r['ks'][0]},{r['ks'][1]}]")
    print("\n把 n 推到 1e16：")
    for nm in ("pareto4", "halfnormal", "lognormal", "relu"):
        cells = []
        for e in (3, 6, 9, 12, 15, 20, 30):
            n = 10 ** e
            r = model_I(nm, 2, n)
            cells.append(f"1e{e}: I={r['I']:.3g}")
        print(f"  {nm:>10} " + "  ".join(cells))
    print("\n  rho_n=(I-1)/(n-1) 与 I_n/n：")
    for nm in ("pareto4", "halfnormal", "lognormal"):
        cells = []
        for e in (4, 6, 8, 10, 12, 15):
            n = 10 ** e
            r = model_I(nm, 2, n)
            cells.append(f"1e{e}: rho={((r['I']-1)/(n-1)):.2e} I/n={r['I']/n:.1e}")
        print(f"  {nm:>10} " + " | ".join(cells))

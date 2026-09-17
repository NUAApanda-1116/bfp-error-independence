"""退化模型的【总体】I_n：与论文表 tab:model 的无拟合预言对拍，并外推到大 n。
全部用生存函数之差计算（避免 1-CDF 的相消），方差用两遍法。
"""
import math
import numpy as np

SQ2 = math.sqrt(2.0)
IP = 1.0 / math.sqrt(2 * math.pi)
Q = lambda z: 0.5 * math.erfc(z / SQ2)          # 标准正态上尾
phi = lambda z: math.exp(-0.5 * z * z) * IP

# |x| 的生存函数 / 尾积分
def Sf(nm, x):
    if x <= 0:
        return 1.0
    if nm == "halfnormal":
        return math.erfc(x / SQ2)
    if nm == "relu":
        return 0.5 * math.erfc(x / SQ2)
    if nm == "lognormal":
        return 0.5 * math.erfc(math.log(x) / SQ2)
    if nm == "pareto4":
        return 1.0 if x < 1 else x ** -4.0

def Gtail(nm, x):            # int_x^inf t f(t) dt
    x = max(x, 0.0)
    if nm == "halfnormal":
        return 2.0 * phi(x)
    if nm == "relu":
        return phi(x)
    if nm == "lognormal":
        return math.exp(0.5) * Q(math.log(x) - 1) if x > 0 else math.exp(0.5)
    if nm == "pareto4":
        return 4.0 / 3 if x <= 1 else (4.0 / 3) * x ** -3

def H2(nm, x):               # int_x^inf t^2 f(t) dt
    x = max(x, 0.0)
    if nm == "halfnormal":
        return 2.0 * (x * phi(x) + Q(x))
    if nm == "relu":
        return x * phi(x) + Q(x)
    if nm == "lognormal":
        return math.exp(2) * Q(math.log(x) - 2) if x > 0 else math.exp(2)
    if nm == "pareto4":
        return 2.0 if x <= 1 else 2.0 * x ** -2

E_X = {"halfnormal": math.sqrt(2 / math.pi), "relu": math.sqrt(1 / (2 * math.pi)),
       "lognormal": math.exp(0.5), "pareto4": 4 / 3}
T = {"halfnormal": 10.0, "relu": 10.0, "lognormal": 3000.0, "pareto4": 3000.0}

def shape(nm, s):
    """E[e], E[e^2], Var(e|xi) 对单块，按舍入单元精确求和"""
    M = int(math.ceil(T[nm] / s)) + 2
    Ee = Ee2 = 0.0
    zero = 0
    for m in range(0, M + 1):
        lo, hi = s * (m - 0.5), s * (m + 0.5)
        Pm = Sf(nm, lo) - Sf(nm, hi)
        zero = zero + 1 if Pm <= 0.0 else 0
        if zero > 3:
            break
        e1 = Gtail(nm, lo) - Gtail(nm, hi)
        e2 = H2(nm, lo) - H2(nm, hi)
        Ee += s * m * Pm - e1
        Ee2 += (s * m) ** 2 * Pm - 2 * s * m * e1 + e2
    return Ee, Ee2

def model_I(nm, Lm, n, verbose=False):
    lo_, hi_ = 0.0, 200.0
    for _ in range(400):
        mid = 0.5 * (lo_ + hi_)
        if n * Sf(nm, 2.0 ** mid) > 1e-5:
            lo_ = mid
        else:
            hi_ = mid
    khi = int(math.ceil(hi_)) + 3
    ks, ps = [], []
    for k in range(khi - 130, khi + 1):
        p = math.exp(-n * Sf(nm, 2.0 ** (k + 1))) - math.exp(-n * Sf(nm, 2.0 ** k))
        if p > 0:
            ks.append(k); ps.append(p)
    tot = sum(ps); ps = [p / tot for p in ps]
    ms, vs = [], []
    for k in ks:
        s = 2.0 ** (k - Lm)
        Ee, Ee2 = shape(nm, s)
        ms.append(Ee); vs.append(max(Ee2 - Ee * Ee, 0.0))
    Em = sum(p * m for p, m in zip(ps, ms))
    var_m = sum(p * (m - Em) ** 2 for p, m in zip(ps, ms))     # 两遍法
    A = sum(p * v for p, v in zip(ps, vs))                     # E[Var(e|xi)]
    var_e = A + var_m
    return 1.0 + (n - 1) * var_m / var_e, Em, var_m, A

print("=== 与论文表 tab:model / tab:bounded 的模型列对拍（n=4096, Lm=2）===")
for nm, ref in (("lognormal", 176.4593), ("halfnormal", 15.0719), ("relu", 6.3155), ("pareto4", 2887.29)):
    I, Em, vm, A = model_I(nm, 2, 4096)
    print(f"  {nm:>10}: 本实现 {I:12.4f}  论文模型 {ref:>10.2f}  偏差 {100*(I/ref-1):+.2f}%   E[m]={Em:+.5f}")

print("\n=== 外推：总体 I_n 与 rho_n ===")
print(f"{'dist':>11}{'Lm':>3}" + "".join(f"{('n=1e%d' % e):>13}" for e in (3, 4, 5, 6, 7, 8, 10, 12, 15, 20)))
for nm in ("pareto4", "lognormal", "halfnormal", "relu"):
    for Lm in (2,):
        row = []
        for e in (3, 4, 5, 6, 7, 8, 10):
            I, Em, vm, A = model_I(nm, Lm, 10 ** e)
            row.append(f"{I:>13.4g}")
        print(f"{nm:>11}{Lm:>3}" + "".join(row))
print("\n（双精度在 n>=1e8 时对 pareto4 已受相消误差影响，该列以 60 位十进制脚本 repro/pareto_asymptotic.py 为准）")

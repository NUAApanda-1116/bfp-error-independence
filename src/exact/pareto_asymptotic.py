"""Pareto(4) 退化模型的【总体】I_n —— 60 位十进制精度，消除双精度相消误差。
回答：把 n 推到 1e30，总体 I_n -> 1 还是 -> const 还是 -> inf。
"""
from decimal import Decimal as D, getcontext
getcontext().prec = 60

Lm = 2
KMAX = 400

def F(x):                     # Pareto(4): x = 1+Lomax(4) >= 1, P(x>t)=t^-4
    x = D(x)
    if x <= 1:
        return D(0)
    return D(1) - x ** -4

def Sf(x):
    x = D(x)
    return D(1) if x <= 1 else x ** -4

def Gtail(x):                 # int_x^inf t f(t) dt = (4/3) x^-3
    x = D(x)
    return D(4) / 3 if x <= 1 else (D(4) / 3) * x ** -3

E_X = D(4) / 3
E_X2 = D(2)
T = D(10) ** 5                # 尾部质量 1e-20

def shape(s):
    """delta(s), Var(g|s) —— 按舍入单元精确求和（全部用 sf 的差，避免相消）"""
    s = D(s)
    M = int((T / s).to_integral_value()) + 2
    lo_grid = [s * (D(m) - D("0.5")) for m in range(0, M + 1)]
    # 单元 m: x in [lo, hi]
    M1 = D(0)   # sum P_m
    M2 = D(0)   # sum P_m E[x|m]
    M3 = D(0)   # sum P_m E[x^2|m]
    for m in range(0, M + 1):
        lo, hi = s * (D(m) - D("0.5")), s * (D(m) + D("0.5"))
        Pm = Sf(lo) - Sf(hi)
        if Pm == 0 and m > 2:
            continue
        # E[x;cell] 与 E[x^2;cell] 用尾积分差：int_lo^hi x f = Gtail(lo)-Gtail(hi)
        e1 = Gtail(lo) - Gtail(hi)
        # int x^2 f dx = int_lo^hi 4 x^-3 dx = 2(lo^-2 - hi^-2)
        f2 = lambda u: D(0) if u <= 1 else D(2) * (D(1) - u ** -2)
        e2 = (D(0) if hi <= 1 else D(2) * (D(1) - hi ** -2)) - (D(0) if lo <= 1 else D(2) * (D(1) - lo ** -2))
        M1 += Pm; M2 += e1; M3 += e2
    # E[e] = sum (s m P_m - E[x;cell]),  E[e^2] = sum (s^2 m^2 P_m - 2 s m E[x;cell] + E[x^2;cell])
    Ee = D(0); Ee2 = D(0)
    for m in range(0, M + 1):
        lo, hi = s * (D(m) - D("0.5")), s * (D(m) + D("0.5"))
        Pm = Sf(lo) - Sf(hi)
        if Pm == 0 and m > 2:
            continue
        e1 = Gtail(lo) - Gtail(hi)
        e2 = (D(0) if hi <= 1 else D(2) * (D(1) - hi ** -2)) - (D(0) if lo <= 1 else D(2) * (D(1) - lo ** -2))
        Ee += (s * m) * Pm - e1
        Ee2 += (s * m) ** 2 * Pm - 2 * s * m * e1 + e2
    return Ee, Ee2, Ee2 - Ee ** 2      # delta*s = Ee ; E[e^2]

def model(n):
    n = D(n)
    # xi 的分布
    ks, ps, ms, vs = [], [], [], []
    for k in range(0, KMAX):
        a, b = D(2) ** k, D(2) ** (k + 1)
        la = n * F(a).ln() if F(a) > 0 else D(-10) ** 400
        lb = n * F(b).ln() if F(b) > 0 else D(-10) ** 400
        try:
            pa = la.exp()
        except Exception:
            pa = D(0)
        try:
            pb = lb.exp()
        except Exception:
            pb = D(0)
        p = pb - pa
        if p < 0:
            p = D(0)
        if p > 0:
            ks.append(k); ps.append(p)
    tot = sum(ps)
    ps = [p / tot for p in ps]
    for k, p in zip(ks, ps):
        s = D(2) ** (k - Lm)
        Ee, Ee2, _ = shape(s)
        ms.append(Ee)
        vs.append(Ee2 - Ee ** 2)      # E[Var(e|xi)]，>0
    Em = sum(p * m for p, m in zip(ps, ms))
    Em2 = sum(p * m * m for p, m in zip(ps, ms))
    var_m = Em2 - Em ** 2
    Evar = sum(p * v for p, v in zip(ps, vs))
    var_e = Evar + var_m
    I = D(1) + (n - 1) * var_m / var_e if var_e > 0 else D(1)
    return I, Em, var_m, var_e, min(ks), max(ks)

print(f"{'n':>10}{'I_n':>16}{'rho_n':>14}{'I_n/n':>12}{'E[m]':>12}{'xi范围':>12}")
for n in (65536, 262144, 10**5, 10**6, 10**7, 10**3, 10**4, 10**8, 10**10, 10**12, 10**15, 10**20, 10**30, 10**50):
    e = f"{n:.0e}"
    I, Em, vm, ve, k0, k1 = model(n)
    rho = (I - 1) / (n - 1)
    print(f"{n:>10}{float(I):>16.6g}{float(rho):>14.5g}{float(I/n):>12.5g}"
          f"{float(Em):>12.6f}   [{k0},{k1}]")

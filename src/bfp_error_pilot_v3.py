"""
阶段：2025-12
块浮点误差 pilot v3 —— 消除 v0.1 初稿中两处最致命的局限

局限 §5.4（最关键）：rho 随块大小 n 的标度未测。
    若 rho ~ c/n  =>  方差膨胀因子 1+(n-1)rho -> 1+c 有界 => 发现三退化为常数修正
    若 rho ~ const => 膨胀因子随 n 线性增长            => 发现三成立
  决定性诊断量：n*rho。若趋于常数则前者，若随 n 增长则后者。

局限 §5.2：结论是否只是 RNE 舍入规则的产物？
  测 RNE / 半远离 / 截断 / 随机舍入 四种规则。

输出：rho 的标度指数（log-log 拟合）、n*rho 走势、各舍入规则下的偏置与 rho。
"""

import numpy as np

RNG = np.random.default_rng(20260915)

TOTAL = 1 << 21                      # 总元素数固定，便于跨 n 比较
N_LIST = [8, 16, 32, 64, 128, 256, 512]
DISTS = ["gauss", "student_t3", "lognormal", "spike_mix", "relu", "halfnormal"]


def make_samples(name, n, rng):
    if name == "gauss":
        return rng.standard_normal(n)
    if name == "student_t3":
        return rng.standard_t(3, n)
    if name == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if name == "spike_mix":
        big = rng.random(n) < 0.05
        x = rng.standard_normal(n) * 0.01
        x[big] = rng.standard_normal(int(big.sum()))
        return x
    if name == "relu":
        return np.maximum(rng.standard_normal(n), 0.0)
    if name == "halfnormal":
        return np.abs(rng.standard_normal(n))
    raise ValueError(name)


def round_values(y, mode, rng=None):
    """y 为已除以步长的商，需舍入为整数"""
    if mode == "rne":
        return np.round(y)                      # 银行家舍入
    if mode == "half_away":
        return np.sign(y) * np.floor(np.abs(y) + 0.5)
    if mode == "trunc":
        return np.trunc(y)
    if mode == "sr":
        lo = np.floor(y)
        frac = y - lo
        return lo + (rng.random(y.shape) < frac)
    raise ValueError(mode)


def quantize_blocks(x, Lm, mode, rng):
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    q = round_values(x / step, mode, rng) * step
    return q, step[:, 0]


def measure(dist, Lm, n, mode, rng):
    nb = max(TOTAL // n, 512)
    x = make_samples(dist, nb * n, rng).reshape(nb, n)
    q, step = quantize_blocks(x, Lm, mode, rng)
    err = q - x

    blk_mean = err.mean(axis=1)
    bias_abs = blk_mean.mean()
    z_bias = bias_abs / (blk_mean.std(ddof=1) / np.sqrt(nb))

    var_emp = err.var()
    var_pred = (step ** 2).mean() / 12.0

    eg = err - err.mean()
    cov_ij = float((eg[:, 0] * eg[:, 1]).mean())
    var_e = float(eg.var())
    rho = cov_ij / var_e
    inflation = 1.0 + (n - 1) * rho

    return dict(dist=dist, Lm=Lm, n=n, mode=mode, z_bias=z_bias,
                var_ratio=var_emp / var_pred, rho=rho,
                n_rho=n * rho, inflation=inflation, nb=nb)


def main():
    rng = np.random.default_rng(20260915)

    # ---------- 表 A：rho 随块大小 n 的标度（RNE, Lm=4）----------
    print("=" * 104)
    print("表 A：rho 随块大小 n 的标度  (RNE, Lm=4)  决定性诊断量 n*rho")
    print("  rho*n -> 常数  表示 rho ~ c/n，膨胀因子有界，发现三弱化")
    print("  rho*n 随 n 增长 表示 rho 不衰减，膨胀因子线性增长，发现三成立")
    print("=" * 104)
    print(f"{'dist':<12} " + " ".join(f"n={n:<5}" for n in N_LIST) + "   | 标度指数")
    print("-" * 104)

    scaling = {}
    for dist in DISTS:
        rhos, ns, line = [], [], []
        for n in N_LIST:
            r = measure(dist, 4, n, "rne", rng)
            rhos.append(r["rho"])
            ns.append(n)
            line.append(f"{r['rho']:+.4f}")
        rhos = np.array(rhos); ns = np.array(ns, dtype=float)
        slope = np.polyfit(np.log(ns), np.log(np.abs(rhos) + 1e-12), 1)[0]
        scaling[dist] = slope
        print(f"{dist:<12} " + " ".join(f"{v:>8}" for v in line) + f"   |  {slope:+.3f}")

    print()
    print(f"{'dist':<12} " + " ".join(f"n={n:<5}" for n in N_LIST) + "   | 判读")
    print("-" * 104)
    for dist in DISTS:
        vals = []
        for n in N_LIST:
            r = measure(dist, 4, n, "rne", rng)
            vals.append(r["n_rho"])
        s = scaling[dist]
        verdict = "rho~c/n (膨胀有界) -> 发现三弱化" if s < -0.7 else \
                  ("rho 不衰减 -> 发现三成立" if s > -0.35 else "中间地带")
        print(f"{dist:<12} " + " ".join(f"{v:>+8.3f}" for v in vals) + f"   | {verdict}")

    # ---------- 表 B：膨胀因子随 n ----------
    print()
    print("=" * 104)
    print("表 B：方差膨胀因子 1+(n-1)rho 随块大小 n")
    print("=" * 104)
    print(f"{'dist':<12} " + " ".join(f"n={n:<5}" for n in N_LIST))
    print("-" * 104)
    for dist in DISTS:
        vals = [measure(dist, 4, n, "rne", rng)["inflation"] for n in N_LIST]
        print(f"{dist:<12} " + " ".join(f"{v:>8.3f}" for v in vals))

    # ---------- 表 C：舍入规则敏感性 ----------
    print()
    print("=" * 104)
    print("表 C：舍入规则敏感性  (n=32, Lm=4)  —— 检查结论是否只是 RNE 的产物")
    print("=" * 104)
    print(f"{'dist':<12} {'mode':<10} {'z_bias':>10} {'emp/pred':>10} {'rho':>10} {'n*rho':>9} {'inflation':>10}")
    print("-" * 104)
    for dist in DISTS:
        for mode in ["rne", "half_away", "trunc", "sr"]:
            r = measure(dist, 4, 32, mode, rng)
            print(f"{r['dist']:<12} {mode:<10} {r['z_bias']:>10.2f} {r['var_ratio']:>10.3f} "
                  f"{r['rho']:>+10.4f} {r['n_rho']:>+9.3f} {r['inflation']:>10.3f}")
        print("-" * 104)


if __name__ == "__main__":
    main()

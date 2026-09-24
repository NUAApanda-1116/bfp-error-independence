"""
块浮点（Block Floating Point, BFP）舍入误差模型检验 —— pilot
阶段：2025-10（v1，研究启动）

目的：检验被硬件文献长期继承的经典假设（Kalliojarvi & Astola 1996 及其后续）：
  H1  块量化误差零均值：            E[e] = 0
  H2  块内误差方差可按 σ² = E[step²]/12 预测
  H3  块内各元素的误差互不相关：     corr(e_i, e_j) = 0  (i≠j)
  H4  由此推论：长点积的误差随长度按 sqrt(K) 增长，而非 K

做法：在真实块浮点语义下（块内共享指数 = 块内最大幅值对应的 2 的幂），
对若干代表性分布做蒙特卡洛，直接测量上述四个量。

注意：本脚本只产生经验证据，不构成证明。
"""

import numpy as np

RNG = np.random.default_rng(20260913)
N_BLOCKS = 40000
LMS = [2, 3, 4, 7]          # 尾数位宽：E5M2=2, E4M3/MXFP4=3, 4, bf16=7
BLOCKS = [16, 32, 64]


def make_samples(name, n, rng):
    if name == "gauss":
        return rng.standard_normal(n)
    if name == "student_t3":
        return rng.standard_t(3, n)
    if name == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if name == "spike_mix":
        # 95% 近零 + 5% 大幅值：模拟 LLM 激活的离群结构
        big = rng.random(n) < 0.05
        x = rng.standard_normal(n) * 0.01
        x[big] = rng.standard_normal(big.sum())
        return x
    raise ValueError(name)


DISTS = ["gauss", "student_t3", "lognormal", "spike_mix"]


def bfp_quantize(x, Lm, block):
    """x: (m, block) -> 量化值 q, 以及每块的步长 step"""
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)                 # 块内共享步长
    q = np.round(x / step) * step
    return q, step[:, 0]


def test_hypotheses(dist, Lm, block):
    nb = N_BLOCKS
    x = make_samples(dist, nb * block, RNG).reshape(nb, block)
    q, step = bfp_quantize(x, Lm, block)
    err = q - x

    # H1 零均值：用块均值的标准误做 z 检验（块间独立）
    blk_mean = err.mean(axis=1)
    z_bias = blk_mean.mean() / (blk_mean.std(ddof=1) / np.sqrt(nb))

    # H2 方差：经验总方差 vs 经典预测 E[step^2]/12
    var_emp = err.var()
    var_pred = (step ** 2).mean() / 12.0

    # H3 块内互相关（去除块均值以消除公共偏移）
    ec = err - err.mean(axis=1, keepdims=True)
    num = (ec[:, :, None] * ec[:, None, :]).mean(axis=0)
    den = ec.var()
    corr = num / den
    off = ~np.eye(block, dtype=bool)
    mean_offdiag_corr = corr[off].mean()

    # 相对误差的量级（离群挤压效应）
    rel = np.abs(err).mean() / np.abs(x).mean()

    return dict(dist=dist, Lm=Lm, block=block, z_bias=z_bias,
                var_emp=var_emp, var_pred=var_pred,
                var_ratio=var_emp / var_pred,
                corr=mean_offdiag_corr, rel=rel)


def dot_scaling(dist, Lm, block, Ks=(256, 512, 1024, 2048, 4096, 8192), reps=60):
    """H4：点积误差随 K 的增长指数（sqrt 斜率 = 0.5, 线性 = 1.0）"""
    out = []
    for K in Ks:
        nb = K // block
        rels = []
        for _ in range(reps):
            a = make_samples(dist, K, RNG)
            b = make_samples(dist, K, RNG)
            exact = float(a @ b)
            qa, _ = bfp_quantize(a[None, :].reshape(1, K), Lm, block)
            qb, _ = bfp_quantize(b[None, :].reshape(1, K), Lm, block)
            approx = float(qa[0] @ qb[0])
            denom = max(abs(exact), 1e-300)
            rels.append(abs(approx - exact) / denom)
        out.append((K, float(np.median(rels))))
    ks = np.array([o[0] for o in out], dtype=float)
    vs = np.array([o[1] for o in out], dtype=float)
    ok = vs > 0
    slope = np.polyfit(np.log(ks[ok]), np.log(vs[ok]), 1)[0]
    return out, slope


def main():
    print("=" * 100)
    print("H1-H3：块浮点舍入误差的经典假设检验（N_BLOCKS = %d）" % N_BLOCKS)
    print("=" * 100)
    hdr = f"{'dist':<12} {'Lm':>3} {'blk':>4} {'z_bias':>10} {'var_emp':>11} {'var_pred':>11} {'emp/pred':>9} {'corr':>9} {'relerr':>8}"
    print(hdr)
    print("-" * 100)
    rows = []
    for dist in DISTS:
        for Lm in LMS:
            for block in BLOCKS:
                r = test_hypotheses(dist, Lm, block)
                rows.append(r)
                print(f"{r['dist']:<12} {r['Lm']:>3} {r['block']:>4} {r['z_bias']:>10.2f} "
                      f"{r['var_emp']:>11.3e} {r['var_pred']:>11.3e} {r['var_ratio']:>9.3f} "
                      f"{r['corr']:>9.4f} {r['rel']:>8.4f}")

    print()
    print("=" * 100)
    print("H4：点积相对误差的增长指数（0.5 = sqrt(K) 即经典预测；1.0 = 线性）")
    print("=" * 100)
    print(f"{'dist':<12} {'Lm':>3} {'blk':>4} {'slope':>8}   {'K -> relerr'}")
    print("-" * 100)
    for dist in DISTS:
        for Lm in (3, 7):
            for block in (32,):
                out, slope = dot_scaling(dist, Lm, block)
                pts = "  ".join(f"{K}:{v:.2e}" for K, v in out)
                print(f"{dist:<12} {Lm:>3} {block:>4} {slope:>8.3f}   {pts}")

    # 汇总判断
    print()
    print("=" * 100)
    print("汇总：与经典模型（零均值 / var_pred / 零相关 / slope=0.5）的偏离")
    print("=" * 100)
    big_bias = [r for r in rows if abs(r["z_bias"]) > 4]
    big_var = [r for r in rows if not (0.9 <= r["var_ratio"] <= 1.1)]
    big_corr = [r for r in rows if abs(r["corr"]) > 0.02]
    print(f"  |z_bias| > 4        : {len(big_bias):3d} / {len(rows)} 配置")
    print(f"  emp/pred 超出 ±10%  : {len(big_var):3d} / {len(rows)} 配置")
    print(f"  |corr| > 0.02       : {len(big_corr):3d} / {len(rows)} 配置")
    if big_corr:
        worst = sorted(big_corr, key=lambda r: -abs(r["corr"]))[:6]
        print("  相关最强的前几个配置：")
        for r in worst:
            print(f"     {r['dist']:<12} Lm={r['Lm']} block={r['block']:>3}  corr={r['corr']:+.4f}  emp/pred={r['var_ratio']:.3f}")


if __name__ == "__main__":
    main()

"""
阶段：2025-11
块浮点舍入误差模型检验 v2 —— 修正 v1 的两处方法学错误

v1 的错误：
  (1) 块内相关用"块内去均值"后计算，强制 sum=0，导致平均两两相关必然等于 -1/(n-1)。
      这是伪影。v1 全部 48 个配置的 corr 恰好只依赖块大小，即为证据。
  (2) 点积相对误差用 |s_hat - s|/|s|，分母在相消时塌陷，斜率无意义。

v2 的修正：
  (1) 相关用全局均值中心化（不破坏块内结构）。核心指标为方差膨胀因子
        Var(sum of block) / (n * Var(e)) = 1 + (n-1) * rho
      rho = 0 -> sqrt(n) 增长（经典预测）；rho > 0 -> 线性增长。
      若 rho 又算出 -1/(n-1)（膨胀因子=0），说明修正仍不充分。
  (2) 点积用范数型相对误差 |s_hat - s| / (||a||_2 ||b||_2)，并加"逐元素指数"基线，
      以隔离"块共享指数"这一因素本身。

新增关键分布：relu / halfnormal —— 非负激活，检验零均值假设的失败是否真实相关。
"""

import numpy as np

RNG = np.random.default_rng(20260914)
N_BLOCKS = 40000
LMS = [2, 3, 4, 7]
BLOCK = 32


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
        x[big] = rng.standard_normal(big.sum())
        return x
    if name == "relu":
        # ReLU 输出：非负、大量精确零、重尾
        return np.maximum(rng.standard_normal(n), 0.0)
    if name == "halfnormal":
        return np.abs(rng.standard_normal(n))
    raise ValueError(name)


DISTS = ["gauss", "student_t3", "lognormal", "spike_mix", "relu", "halfnormal"]


def block_steps(x, Lm):
    """块内共享指数 -> 共享步长"""
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    return 2.0 ** (e - Lm)


def bfp_quantize_blocks(x, Lm):
    step = block_steps(x, Lm)
    return np.round(x / step) * step, step[:, 0]


def per_element_quantize(x, Lm):
    e = np.floor(np.log2(np.maximum(np.abs(x), np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    return np.round(x / step) * step


def measure(dist, Lm, block=BLOCK):
    x = make_samples(dist, N_BLOCKS * block, RNG).reshape(N_BLOCKS, block)
    q, step = bfp_quantize_blocks(x, Lm)
    err = q - x

    # --- 偏置（尺度无关）---
    blk_mean = err.mean(axis=1)
    bias_abs = blk_mean.mean()
    z_bias = bias_abs / (blk_mean.std(ddof=1) / np.sqrt(N_BLOCKS))
    bias_rel = bias_abs / np.abs(x).mean()          # 相对平均幅值的偏置

    # --- 方差 vs 经典预测 ---
    var_emp = err.var()
    var_pred = (step ** 2).mean() / 12.0

    # --- 块内相关（全局均值中心化，不做块内去均值）---
    eg = err - err.mean()
    # 交换性 => 所有 i!=j 对的协方差相同，任取一列即可
    cov_ij = float((eg[:, 0] * eg[:, 1]).mean())
    var_e = float(eg.var())
    rho = cov_ij / var_e
    inflation = 1.0 + (block - 1) * rho           # 方差膨胀因子
    artifact = -1.0 / (block - 1)                 # v1 伪影的理论值

    return dict(dist=dist, Lm=Lm, block=block, z_bias=z_bias, bias_rel=bias_rel,
                var_ratio=var_emp / var_pred, rho=rho, inflation=inflation,
                artifact=artifact, relerr=np.abs(err).mean() / np.abs(x).mean())


def dot_scaling(dist, Lm, block, L=32, Ks=(256, 512, 1024, 2048, 4096, 8192), reps=300):
    """范数型相对误差随 K 的增长；同时给出"逐元素指数"基线"""
    shared, perelem = [], []
    for K in Ks:
        nb = K // block
        rs, rp = [], []
        for _ in range(reps):
            a = make_samples(dist, K, RNG)
            b = make_samples(dist, K, RNG)
            exact = float(a @ b)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                continue
            qa, _ = bfp_quantize_blocks(a[None, :].reshape(1, K), Lm)
            qb, _ = bfp_quantize_blocks(b[None, :].reshape(1, K), Lm)
            rs.append(abs(float(qa[0] @ qb[0]) - exact) / denom)
            rp.append(abs(float(per_element_quantize(a, Lm) @ per_element_quantize(b, Lm)) - exact) / denom)
        shared.append(np.median(rs))
        perelem.append(np.median(rp))
    ks = np.array(Ks, dtype=float)

    def slope(v):
        v = np.array(v)
        ok = v > 0
        if ok.sum() < 3:
            return float("nan")
        return float(np.polyfit(np.log(ks[ok]), np.log(v[ok]), 1)[0])

    return slope(shared), slope(perelem), dict(zip(Ks, np.round(shared, 6)))


def main():
    print("=" * 108)
    print("v2 表 A：偏置与方差（修正后的相关度量）  N_BLOCKS=%d, block=%d" % (N_BLOCKS, BLOCK))
    print("=" * 108)
    print(f"{'dist':<12} {'Lm':>3} {'z_bias':>10} {'bias/|x|':>10} {'emp/pred':>10} "
          f"{'rho':>11} {'inflation':>10} {'artifact':>10} {'sum_relerr':>11}")
    print("-" * 108)
    rows = []
    for dist in DISTS:
        for Lm in LMS:
            r = measure(dist, Lm)
            rows.append(r)
            flag = "  <-- 伪影?" if abs(r["rho"] - r["artifact"]) < 1e-6 else ""
            print(f"{r['dist']:<12} {r['Lm']:>3} {r['z_bias']:>10.2f} {r['bias_rel']:>10.4f} "
                  f"{r['var_ratio']:>10.3f} {r['rho']:>11.3e} {r['inflation']:>10.4f} "
                  f"{r['artifact']:>10.4f} {r['relerr']:>11.4f}{flag}")

    print()
    print("=" * 108)
    print("v2 表 B：点积范数型相对误差的增长斜率（0.5=sqrt(K) 经典；1.0=线性）")
    print("=" * 108)
    print(f"{'dist':<12} {'Lm':>3} {'shared_blk slope':>18} {'per_element slope':>18}   shared / per_element")
    print("-" * 108)
    for dist in DISTS:
        for Lm in (3, 7):
            ss, ps, pts = dot_scaling(dist, Lm, BLOCK)
            print(f"{dist:<12} {Lm:>3} {ss:>18.3f} {ps:>18.3f}   {pts}")

    print()
    print("=" * 108)
    print("v2 结论判据")
    print("=" * 108)
    art = -1.0 / (BLOCK - 1)
    genuine_rho = [r for r in rows if abs(r["rho"] - art) > 1e-6]
    print(f"  rho 与伪影值 {art:.5f} 不同的配置：{len(genuine_rho)} / {len(rows)}")
    sig_bias = [r for r in rows if abs(r["z_bias"]) > 4]
    print(f"  偏置显著 (|z|>4) 的配置        ：{len(sig_bias)} / {len(rows)}")
    for r in sorted(sig_bias, key=lambda r: -abs(r["z_bias"]))[:8]:
        print(f"     {r['dist']:<12} Lm={r['Lm']}  z={r['z_bias']:>9.2f}  bias/|x|={r['bias_rel']:.4f}")
    bad_var = [r for r in rows if not (0.9 <= r["var_ratio"] <= 1.1)]
    print(f"  emp/pred 超出 ±10% 的配置      ：{len(bad_var)} / {len(rows)}")
    for r in sorted(bad_var, key=lambda r: r["var_ratio"])[:8]:
        print(f"     {r['dist']:<12} Lm={r['Lm']}  emp/pred={r['var_ratio']:.4f}")


if __name__ == "__main__":
    main()

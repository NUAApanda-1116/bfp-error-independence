"""
块浮点误差 pilot v4 —— 用有效估计量重做"块内相关性"（发现三）

v2/v3 的缺陷：
  用单对元素 (e_0, e_1) 估计平均协方差 rho，再算膨胀因子 1+(n-1)rho。
  该估计量的标准误被 (n-1) 放大，在 n 大时完全被噪声主导。
  v3 中出现负的膨胀因子（如 -17.487），而可交换序列下膨胀因子恒 >= 0，
  即为该缺陷的直接证据。

v4 的有效估计量：
  不估 rho，直接估待测量本身：
      inflation = Var_blocks( S ) / ( n * Var_elem(e) ),   S = sum_{i in block} e_i
  两个量都用全部数据估计，方差远低于单对估计。
  用分组法给出诚实误差棒（20 组，组内独立）。

  理论下限检查：inflation >= 0 必须成立（可交换性）。
  经典独立模型预测 inflation = 1。
"""

import numpy as np

DISTS = ["gauss", "student_t3", "lognormal", "spike_mix", "relu", "halfnormal"]
N_LIST = [8, 16, 32, 64, 128, 256, 512]
N_GROUPS = 20
NB_PER_GROUP = 4000        # 每组块数 -> 总块数 = 80000
CHUNK = 4000


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


def quantize_blocks(x, Lm):
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    return np.round(x / step) * step, step[:, 0]


def group_stats(dist, Lm, n, rng):
    """对 N_GROUPS 个组，各返回 (Var_blocks(S), Var_elem(e), E[e])"""
    out = []
    for _ in range(N_GROUPS):
        s_sum = s_sq = 0.0
        e_sum = e_sq = 0.0
        cnt = 0
        done = 0
        while done < NB_PER_GROUP:
            m = min(CHUNK, NB_PER_GROUP - done)
            x = make_samples(dist, m * n, rng).reshape(m, n)
            q, _ = quantize_blocks(x, Lm)
            err = q - x
            S = err.sum(axis=1)
            s_sum += S.sum(); s_sq += (S ** 2).sum()
            e_sum += err.sum(); e_sq += (err ** 2).sum()
            cnt += m
            done += m
        var_S = s_sq / cnt - (s_sum / cnt) ** 2
        var_e = e_sq / (cnt * n) - (e_sum / (cnt * n)) ** 2
        out.append((var_S, var_e, e_sum / (cnt * n)))
    return out


def inflation(dist, Lm, n, rng):
    gs = group_stats(dist, Lm, n, rng)
    vals = np.array([vS / (n * ve) for vS, ve, _ in gs])
    # E[e] 的显著性用第一组的块均值近似（仅用于偏置报告）
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


def main():
    rng = np.random.default_rng(20260916)
    print("=" * 106)
    print("v4：方差膨胀因子 1+(n-1)rho 的有效估计（分组均值 ± 标准误，20 组 x 4000 块）")
    print("  理论下限：inflation >= 0（可交换性）；经典独立模型预测 inflation = 1")
    print("=" * 106)

    for Lm in (2, 4):
        print()
        print(f"--- Lm = {Lm} ---")
        print(f"{'dist':<12} " + " ".join(f"{'n='+str(n):>14}" for n in N_LIST))
        print("-" * 106)
        for dist in DISTS:
            cells = []
            for n in N_LIST:
                m, se = inflation(dist, Lm, n, rng)
                cells.append(f"{m:>7.3f}±{se:.3f}")
            print(f"{dist:<12} " + " ".join(f"{c:>14}" for c in cells))

    print()
    print("=" * 106)
    print("判读规则")
    print("=" * 106)
    print("  inflation 与 1 在误差棒内一致      -> 独立模型成立，发现三不成立")
    print("  inflation 显著 > 1 且随 n 增长     -> 存在正相关，发现三成立")
    print("  inflation 显著 < 1                 -> 存在负相关（对求和误差有利）")
    print("  出现 < 0 的值                      -> 估计量仍有问题（可交换性禁止）")


if __name__ == "__main__":
    main()

"""
块浮点误差 pilot v5 —— 消除初稿 §5.4：幂律膨胀是否有上界？
阶段：2026-02

v4 测得对数正态下膨胀 ~ n^0.64~0.75（n <= 512）。本脚本把 n 推到 4096，
并给出局部对数斜率 d log(inflation) / d log(n)，以判断：
    局部斜率保持正且不衰减 -> 幂律持续，n->inf 膨胀无界（强结论）
    局部斜率随 n 下降趋零  -> 存在饱和，膨胀有界（弱结论，退化为常数修正）

对照组：高斯（对称，预期恒为 1）、ReLU 与半正态（v4 中为阈值型）。
种子为固定实验标识符，用于复现归档输出。
"""

import numpy as np

DISTS = ["gauss", "lognormal", "relu", "halfnormal"]
N_LIST = [32, 64, 128, 256, 512, 1024, 2048, 4096]
LM_LIST = [2, 4]
N_GROUPS = 20
NB_PER_GROUP = 1000
CHUNK = 1000


def make_samples(name, n, rng):
    if name == "gauss":
        return rng.standard_normal(n)
    if name == "lognormal":
        return rng.lognormal(0.0, 1.0, n)
    if name == "relu":
        return np.maximum(rng.standard_normal(n), 0.0)
    if name == "halfnormal":
        return np.abs(rng.standard_normal(n))
    raise ValueError(name)


def quantize_blocks(x, Lm):
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    return np.round(x / step) * step


def inflation(dist, Lm, n, rng):
    vals = []
    for _ in range(N_GROUPS):
        s_sum = s_sq = e_sum = e_sq = 0.0
        cnt = 0
        done = 0
        while done < NB_PER_GROUP:
            m = min(CHUNK, NB_PER_GROUP - done)
            x = make_samples(dist, m * n, rng).reshape(m, n)
            err = quantize_blocks(x, Lm) - x
            S = err.sum(axis=1)
            s_sum += S.sum(); s_sq += (S ** 2).sum()
            e_sum += err.sum(); e_sq += (err ** 2).sum()
            cnt += m; done += m
        var_S = s_sq / cnt - (s_sum / cnt) ** 2
        var_e = e_sq / (cnt * n) - (e_sum / (cnt * n)) ** 2
        vals.append(var_S / (n * var_e))
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


def main():
    rng = np.random.default_rng(20260917)
    for Lm in LM_LIST:
        print("=" * 110)
        print(f"v5：方差膨胀因子随块大小 n 的走势  (Lm = {Lm}, {N_GROUPS} 组 x {NB_PER_GROUP} 块)")
        print("  局部斜率 = d log(inflation)/d log(n)。趋零 => 饱和；保持正 => 幂律持续")
        print("=" * 110)
        print(f"{'dist':<12} " + " ".join(f"{'n='+str(n):>15}" for n in N_LIST))
        print("-" * 110)
        table = {}
        for dist in DISTS:
            cells, means = [], []
            for n in N_LIST:
                m, se = inflation(dist, Lm, n, rng)
                means.append(m)
                cells.append(f"{m:>8.3f}±{se:.3f}")
            table[dist] = np.array(means)
            print(f"{dist:<12} " + " ".join(f"{c:>15}" for c in cells))

        print()
        print(f"{'dist':<12} " + " ".join(f"{'->'+str(N_LIST[i+1]):>10}" for i in range(len(N_LIST) - 1)))
        print("-" * 110)
        for dist in DISTS:
            v = table[dist]
            local = []
            for i in range(len(N_LIST) - 1):
                s = (np.log(v[i + 1]) - np.log(v[i])) / (np.log(N_LIST[i + 1]) - np.log(N_LIST[i]))
                local.append(s)
            print(f"{dist:<12} " + " ".join(f"{s:>10.3f}" for s in local))
            tail = np.mean(local[-3:])
            if dist == "gauss":
                verdict = "对照：恒为 1，符合预期"
            elif tail > 0.6:
                verdict = f"尾部斜率 {tail:.3f} -> 幂律持续，膨胀似无界"
            elif tail > 0.25:
                verdict = f"尾部斜率 {tail:.3f} -> 仍在增长但减速"
            else:
                verdict = f"尾部斜率 {tail:.3f} -> 趋于饱和，膨胀有界"
            print(f"{'':<12} {verdict}")
        print()


if __name__ == "__main__":
    main()

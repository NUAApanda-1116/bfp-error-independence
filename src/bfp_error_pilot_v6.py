"""
阶段：2026-03
块浮点误差 pilot v6 —— 检验机制的核心可证伪预言

§4.1 机制：Cov(e_i, e_j) = Var(E[e | s])   （s = 共享步长）
§4.2 猜想：该量为 0 当且仅当给定块最大值时输入条件对称。

随机舍入（SR）按构造满足 E[e | x, s] = 0（向下取整的概率 = 小数部分），
且各元素的 SR 随机性相互独立。由此推出：
        E[e | block] = 0  =>  Cov(e_i, e_j | block) = 0  =>  rho = 0
故核心预言为：

        ** 在 SR 下，对一切分布与一切块长 n，膨胀因子 I_n = 1 **

若观测到 I_n = 1，机制得到独立的强支持。
若 SR 下 I_n != 1，则 §4.1 的分解或 §4.2 的猜想有误，机制需修正。
同一配置下并跑 RNE 作为对照。
"""

import numpy as np

DISTS = ["gauss", "lognormal", "relu", "halfnormal"]
N_LIST = [32, 512, 4096]
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


def quantize_blocks(x, Lm, mode, rng):
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    y = x / step
    if mode == "rne":
        q = np.round(y)
    elif mode == "sr":
        lo = np.floor(y)
        q = lo + (rng.random(y.shape) < (y - lo))
    else:
        raise ValueError(mode)
    return q * step


def inflation(dist, Lm, n, mode, rng):
    vals = []
    for _ in range(N_GROUPS):
        s_sum = s_sq = e_sum = e_sq = 0.0
        cnt = 0
        done = 0
        while done < NB_PER_GROUP:
            m = min(CHUNK, NB_PER_GROUP - done)
            x = make_samples(dist, m * n, rng).reshape(m, n)
            err = quantize_blocks(x, Lm, mode, rng) - x
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
    rng = np.random.default_rng(20260918)
    print("=" * 108)
    print("v6：机制的核心可证伪预言 —— 随机舍入下是否恒有 I_n = 1？")
    print("  预言：SR 使 E[e|block]=0 且块内条件独立 => Cov=0 => I_n = 1（对所有分布、所有 n）")
    print("=" * 108)

    for Lm in LM_LIST:
        print()
        print(f"--- Lm = {Lm} ---")
        print(f"{'dist':<12} {'mode':<5} " + " ".join(f"{'n='+str(n):>16}" for n in N_LIST))
        print("-" * 108)
        for dist in DISTS:
            for mode in ("rne", "sr"):
                cells = []
                for n in N_LIST:
                    m, se = inflation(dist, Lm, n, mode, rng)
                    cells.append(f"{m:>8.3f}±{se:.3f}")
                tag = "  <== 预言值 1" if mode == "sr" else ""
                print(f"{dist:<12} {mode:<5} " + " ".join(f"{c:>16}" for c in cells) + tag)
            print("-" * 108)

    print()
    print("=" * 108)
    print("判读")
    print("=" * 108)
    print("  若 SR 行全部落在 1.000 ± 3se 内  -> 机制得到独立强支持，§4.2 猜想可保留")
    print("  若 SR 行显著偏离 1               -> §4.1 的分解或 §4.2 的猜想有误，机制需修正")


if __name__ == "__main__":
    main()

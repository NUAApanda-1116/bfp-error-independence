r"""
阶段：2026-07
块浮点误差 pilot v8 —— 两项收尾实验

(A) 有界支撑 vs 无界支撑：检验"重尾性是否为必要条件"，并检验由推论得到的判据
      有界支撑  =>  xi = floor(log2 max|x|) 以概率 1 收敛到一个常数
                =>  Var(m(xi)) -> 0（指数速度）=>  I_n -> 1
    因此预测：有界支撑的各分布一律 I_n -> 1（不管端点尾指数），
              而无界支撑的分布（指数、对数正态、Pareto）仍然产生膨胀。
    同时报告 P(xi = 众数) 以直接展示"xi 集中"这一机制。

(B) 大块长 n 到 65536：rho_n = (I_n-1)/(n-1) 是趋于正常数还是衰减？
    三种区制：
      有界支撑                    : rho_n -> 0 指数速度, I_n -> 1
      正则变化尾（Pareto alpha）  : log2 M_n 的涨落 ~ O(1)  => rho_n -> rho_inf > 0, I_n ~ n
      对数正态（Gumbel 型）       : log2 M_n 的涨落 ~ 1/(ln2*sqrt(2 ln n)) -> 0
                                    => rho_n ~ C/sqrt(ln n) -> 0 但 I_n ~ C n/sqrt(ln n) -> inf
    可检验的定量预言（对数正态）：rho_n 正比于 1/sqrt(ln n)，故
      I_65536 / I_4096 应约为 16 * sqrt(ln4096/ln65536) = 13.9，
      而"rho 趋于常数"的假设给出 16。

脚本复用 v7 的收集/分箱/退化模型代码（importlib 载入并替换 make_samples）。
"""

import importlib.util
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_v7():
    spec = importlib.util.spec_from_file_location("v7", os.path.join(_HERE, "bfp_error_pilot_v7.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v7 = _load_v7()
_V7_MAKE_SAMPLES = v7.make_samples   # 先保存原实现，避免 v8 版回退时递归


def make_samples(name, n, rng):
    """在 v7 的六类分布之外，加入四类"支撑性质"对照分布。"""
    if name == "exponential":                      # 无界、轻尾、非对称
        return rng.exponential(1.0, n)
    if name == "pareto4":                          # 无界、正则变化尾（alpha=4，方差有限）
        return rng.pareto(4.0, n) + 1.0
    if name == "beta25":                           # 有界 [0,1]，非对称，密度在 1 处 4 阶消失
        return rng.beta(2.0, 5.0, n)
    if name == "beta52":                           # 有界 [0,1]，非对称，密度在 1 处 1 阶消失
        return rng.beta(5.0, 2.0, n)
    if name == "truncnorm":                        # 有界 [0,2]，非对称（|N| 截断）
        out = np.empty(n)
        i = 0
        while i < n:
            cand = np.abs(rng.standard_normal(max(n, 64)))
            cand = cand[cand <= 2.0]
            take = min(len(cand), n - i)
            out[i:i + take] = cand[:take]
            i += take
        return out
    return _V7_MAKE_SAMPLES(name, n, rng)


v7.make_samples = make_samples
v7.SHAPE_N = 4_000_000
v7.CHUNK = 1000


def run(dist, Lm, n, mode, seed, chunk=None):
    old = v7.CHUNK
    if chunk is not None:
        v7.CHUNK = chunk
    try:
        st = v7.collect(dist, Lm, n, mode, seed)
    finally:
        v7.CHUNK = old
    out = v7.analyse(*st, n, Lm)
    out["_s2"] = st[4]
    p, mode_xi, k = xi_concentration(st[0])
    out["p_mode"], out["mode_xi"], out["n_xi"] = p, mode_xi, k
    return out


def xi_concentration(xi):
    vals, cnt = np.unique(xi, return_counts=True)
    j = int(np.argmax(cnt))
    return float(cnt[j] / cnt.sum()), int(vals[j]), len(vals)


def table_A(seed=20260921):
    dists = ["exponential", "pareto4", "beta25", "beta52", "truncnorm", "lognormal", "relu"]
    LM = [2, 4]
    NS = [32, 512, 4096]
    print("=" * 122)
    print("(A) 有界 vs 无界支撑：'重尾性是否为必要条件'")
    print("    预言：有界支撑 => xi 集中到单一取值 => Var(m)=0 => I_n = 1（与端点尾指数无关）")
    print("=" * 122)
    for Lm in LM:
        head = (f"{'dist':<12}{'n':>6}{'I_n 实测':>11}{'I_n 模型':>11}{'rho_n':>10}"
                f"{'Var(m)':>11}{'P(xi=众数)':>12}{'#xi':>6}{'死区':>8}")
        print(f"\n--- Lm = {Lm} ---")
        print(head)
        print("-" * len(head))
        for dist in dists:
            for n in NS:
                r = run(dist, Lm, n, "rne", seed, chunk=400 if n >= 4096 else 1000)
                r.update(v7.reduced_model(dist, Lm, n, r.pop("_s2")))
                print(f"{dist:<12}{n:>6}{r['I_direct']:>11.4f}{r['model_I']:>11.4f}"
                      f"{r['rho_var_m']:>10.5f}{r['var_m']:>11.3e}{r['p_mode']:>12.6f}"
                      f"{r['n_xi']:>6}{r['p_dz']:>8.3f}")
            print("-" * len(head))
    print()


def table_B(seed=20260922):
    """大块长：rho_n 趋于常数还是衰减。"""
    dists = ["lognormal", "relu", "halfnormal", "pareto4", "gauss"]
    NS = [4096, 16384, 65536]
    LM = [2]
    NB_GROUPS, NB_PER_GROUP = 20, 200
    print("=" * 122)
    print("(B) 大块长 n 到 65536：rho_n = (I_n-1)/(n-1) 趋于常数还是衰减？")
    print(f"    {NB_GROUPS} 组 x {NB_PER_GROUP} 块 = {NB_GROUPS * NB_PER_GROUP} 块/配置，Lm = 2")
    print("=" * 122)
    old = (v7.N_GROUPS, v7.NB_PER_GROUP)
    v7.N_GROUPS, v7.NB_PER_GROUP = NB_GROUPS, NB_PER_GROUP
    try:
        head = (f"{'dist':<12}{'n':>8}{'I_n 实测':>12}{'I_n 模型':>12}{'rho_n':>11}"
                f"{'I_n/n':>10}{'P(xi=众数)':>12}{'#xi':>6}")
        print(head)
        print("-" * len(head))
        results = {}
        for dist in dists:
            row = []
            for n in NS:
                r = run(dist, 2, n, "rne", seed, chunk=100 if n >= 16384 else 400)
                r.update(v7.reduced_model(dist, 2, n, r.pop("_s2")))
                row.append(r)
                print(f"{dist:<12}{n:>8}{r['I_direct']:>12.4f}{r['model_I']:>12.4f}"
                      f"{r['rho_var_m']:>11.5f}{r['I_direct'] / n:>10.5f}"
                      f"{r['p_mode']:>12.6f}{r['n_xi']:>6}")
            results[dist] = row
            r0, r1 = row[0], row[-1]
            obs = r1["I_direct"] / r0["I_direct"]
            d_ratio = float(np.sqrt(np.log(4096.0) / np.log(65536.0)))
            print(f"{'':<12}  I_65536/I_4096 实测 {obs:.2f} | rho~const 预言 {65536 / 4096:.2f}"
                  f" | rho~1/sqrt(ln n) 预言 {65536 / 4096 * d_ratio:.2f}")
            print("-" * len(head))
    finally:
        v7.N_GROUPS, v7.NB_PER_GROUP = old
    print()


def table_A2(seed=20260921):
    """直接展示 xi 的集中：给出众数概率随 n 的变化（有界支撑 vs 无界）。"""
    print("=" * 122)
    print("(A2) 机制直接证据：P(xi = 众数) 随 n 的变化（Lm = 2）")
    print("=" * 122)
    dists = ["beta25", "beta52", "truncnorm", "exponential", "pareto4", "lognormal"]
    print(f"{'dist':<12}" + "".join(f"{'n=' + str(n):>16}" for n in [32, 512, 4096]))
    print("-" * 70)
    for dist in dists:
        cells = []
        for n in [32, 512, 4096]:
            rng = np.random.default_rng(seed)
            st = None
            oldg, oldb, oldc = v7.N_GROUPS, v7.NB_PER_GROUP, v7.CHUNK
            v7.N_GROUPS, v7.NB_PER_GROUP, v7.CHUNK = 20, 1000, 400 if n >= 4096 else 1000
            try:
                xi, S, Q, G, s2, n0, ndz = v7.collect(dist, 2, n, "rne", seed)
            finally:
                v7.N_GROUPS, v7.NB_PER_GROUP, v7.CHUNK = oldg, oldb, oldc
            p, mode, k = xi_concentration(xi)
            cells.append(f"{p:.6f}")
        print(f"{dist:<12}" + "".join(f"{c:>16}" for c in cells))
    print()


def main():
    t0 = time.time()
    table_A2()
    table_A()
    table_B()
    print(f"总用时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

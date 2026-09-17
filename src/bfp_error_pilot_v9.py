r"""
块浮点误差 pilot v9 —— 大块长趋势的稳健性复核

v8 的 (B) 表在 4000 块/配置上给出对数正态的 rho_n 由 0.041(4096) 降到 0.018(65536)，
而 Pareto(4) 的 rho_n 稳定在 0.70（I_65536/I_4096 = 15.92 对"rho 常数"预言的 16.00）。
本脚本用两个独立种子、10000 块/配置复核这一趋势，避免把估计噪声读成渐近行为。

复用 v8 的分布定义（有界支撑与 Pareto 都在 v8 里）。
"""

import importlib.util
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"bfp_error_pilot_{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v8 = _load("v8")
v7 = v8.v7
v8.CHUNK_OVERRIDE = None


def measure(dist, n, seed, n_groups=20, nb_per_group=500):
    old = (v7.N_GROUPS, v7.NB_PER_GROUP, v7.CHUNK)
    v7.N_GROUPS, v7.NB_PER_GROUP = n_groups, nb_per_group
    v7.CHUNK = min(200, nb_per_group) if n >= 16384 else min(1000, nb_per_group)
    try:
        st = v7.collect(dist, 2, n, "rne", seed)
    finally:
        v7.N_GROUPS, v7.NB_PER_GROUP, v7.CHUNK = old
    r = v7.analyse(*st, n, 2)
    r.update(v7.reduced_model(dist, 2, n, st[4]))
    p, mode, k = v8.xi_concentration(st[0])
    r.update(p_mode=p, mode_xi=mode, n_xi=k, blocks=n_groups * nb_per_group)
    return r


def main():
    t0 = time.time()
    dists = ["lognormal", "pareto4", "halfnormal"]
    NS = [4096, 16384, 65536]
    seeds = [20260930, 20260931]
    print("=" * 118)
    print("v9：大块长趋势的稳健性复核（Lm=2，20 组 x 500 块 = 10000 块/配置，两个独立种子）")
    print("=" * 118)
    print(f"{'dist':<12}{'n':>8}{'seed':>10}{'I_n 实测':>12}{'I_n 模型':>12}"
          f"{'rho_n':>10}{'I_n/n':>9}{'P(xi=众数)':>12}")
    print("-" * 118)
    for dist in dists:
        rows = []
        for n in NS:
            for sd in seeds:
                r = measure(dist, n, sd)
                rows.append((n, sd, r))
                print(f"{dist:<12}{n:>8}{sd:>10}{r['I_direct']:>12.4f}{r['model_I']:>12.4f}"
                      f"{r['rho_var_m']:>10.5f}{r['I_direct'] / n:>9.5f}{r['p_mode']:>12.4f}")
        print("-" * 118)
        # 每个 n 上两个种子的均值与相对散布
        for n in NS:
            v = [r["I_direct"] for (nn, sd, r) in rows if nn == n]
            rr = [r["rho_var_m"] for (nn, sd, r) in rows if nn == n]
            print(f"    {dist:<12} n={n:>6}  I_n = {np.mean(v):>10.3f} "
                  f"(两种子相差 {abs(v[0] - v[1]) / np.mean(v):.1%})   rho_n = {np.mean(rr):.5f}")
        v0 = np.mean([r["I_direct"] for (nn, sd, r) in rows if nn == NS[0]])
        v1 = np.mean([r["I_direct"] for (nn, sd, r) in rows if nn == NS[-1]])
        print(f"    {dist:<12} I_65536/I_4096 = {v1 / v0:.2f}"
              f"   | rho~const 预言 {NS[-1] / NS[0]:.2f}"
              f" | 实测局部斜率 {np.log(v1 / v0) / np.log(NS[-1] / NS[0]):.3f}")
        print("=" * 118)
    print(f"总用时 {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

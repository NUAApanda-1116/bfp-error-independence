r"""
块浮点误差 pilot v7 —— 直接在共享指数 xi 上条件化，分离命题 1 的两项
阶段：2026-05

目的（对应论文局限 3 / 后续工作 2，以及第 3.2 节的 0.506）：
  1) 把命题 1 从"只测总协方差"推进到"分别测出它的两个来源"：
         Cov(e_i, e_j)  =  E[ Cov(e_i, e_j | xi) ]  +  Var( m(xi) )
                          \______ E[c(xi)] ______/     \__ 条件均值的波动 __/
     并由此直接检验假设 (A2)（即 E[c(xi)] 是否可忽略），而不是把它当作前提。
  2) 用逐 xi 分箱直接给出 Var(m(xi))，反推 rho = Var(m)/Var(e) 与 I_n = 1+(n-1)rho，
     与 v5/v6 的直接估计量相互对照（三条独立路径必须一致）。
  3) 把 E[e^2] = E[s^2 * gamma(xi)] 分解出来，定量解释第 3.2 节实测/预测 = 0.506：
     ReLU 有约一半元素精确为零（"概率原子"），这些元素的量化误差恒为 0，
     而经典公式 E[s^2]/12 对每个元素都计入 s^2/12。
  4) 输出逐箱的 delta(xi) = m(xi)/s 与死区比例 p_dz(xi) = P(|x_i| < s/2 | xi)，
     用以给出 L_m 依赖与标度行为的结构性解释。

关键方法学点（这是 v1 伪影的正解）：
  估计条件协方差 c(xi) 时，中心化必须用 **分箱（条件）均值 m_b**，
  而**不能**用块内均值。用块内均值中心化会强制 sum_i (e_i - ebar)=0，
  使块内平均两两协方差必然等于 -Var(e)/(n-1)，与数据无关——这就是 v1 的伪影。
  本脚本用恒等式
      sum_{i != j} (e_i - m_b)(e_j - m_b) = (S - n*m_b)^2 - sum_i (e_i - m_b)^2
  只需每块的 (S, Q, xi) 三个量即可精确计算，无需保存元素级数据。

同时核对若干严格结论（含一条被本脚本否定的错误猜想）：
      0 <= I_n <= n                                 （Cauchy-Schwarz）
      Var(e)  = E[Var(e|xi)] + Var(m(xi))           （全方差公式，精确）
      Var(ebar) = Var(m) + E[Var(e|xi)]/n           （块均值的精确分解）
      Cov(e_i,e_j) = E[c(xi)] + Var(m(xi))          （全协方差公式，精确）
  被否定者：Cov(e_i,e_j) >= (E[e])^2（即"偏置给出膨胀下界"）。
  对数正态 Lm=2 n=4096 处 n*(E[e])^2/Var(e) = 1586 >> I_n = 175.8，故该不等式不成立。
"""

import numpy as np

DISTS = ["gauss", "student_t3", "lognormal", "spike_mix", "relu", "halfnormal"]
N_LIST = [32, 512, 4096]
LM_LIST = [2, 4]
N_GROUPS = 20
NB_PER_GROUP = 1000
CHUNK = 1000
MIN_BIN_BLOCKS = 20


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


def block_sufficient_stats(x, Lm, mode, rng):
    """返回每块的 (xi, S, Q, G, s2, n0, ndz)。

    S = sum_i e_i, Q = sum_i e_i^2, G = sum_i g(u_i)^2, s2 = s^2,
    n0 = #{x_i == 0}（可精确表示的概率原子）, ndz = #{|x_i| < s/2}（死区）
    """
    mx = np.max(np.abs(x), axis=1, keepdims=True)
    e = np.floor(np.log2(np.maximum(mx, np.finfo(float).tiny)))
    step = 2.0 ** (e - Lm)
    u = x / step
    if mode == "rne":
        g = np.round(u) - u
    elif mode == "sr":
        lo = np.floor(u)
        g = (lo + (rng.random(u.shape) < (u - lo))) - u
    else:
        raise ValueError(mode)
    err = g * step
    return (
        e[:, 0].astype(np.int64),
        err.sum(axis=1),
        (err ** 2).sum(axis=1),
        (g ** 2).sum(axis=1),
        (step[:, 0] ** 2).copy(),
        (x == 0.0).sum(axis=1).astype(np.int64),
        (np.abs(u) < 0.5).sum(axis=1).astype(np.int64),
    )


def collect(dist, Lm, n, mode, seed):
    rng = np.random.default_rng(seed)
    total = N_GROUPS * NB_PER_GROUP
    xi = np.empty(total, dtype=np.int64)
    S = np.empty(total)
    Q = np.empty(total)
    G = np.empty(total)
    s2 = np.empty(total)
    n0 = np.empty(total, dtype=np.int64)
    ndz = np.empty(total, dtype=np.int64)
    done = 0
    while done < total:
        m = min(CHUNK, total - done)
        x = make_samples(dist, m * n, rng).reshape(m, n)
        st = block_sufficient_stats(x, Lm, mode, rng)
        xi[done:done + m], S[done:done + m], Q[done:done + m], \
            G[done:done + m], s2[done:done + m], \
            n0[done:done + m], ndz[done:done + m] = st
        done += m
    return xi, S, Q, G, s2, n0, ndz


def analyse(xi, S, Q, G, s2, n0, ndz, n, Lm, keep_bins=False):
    """全部估计量都只用到每块的充分统计量。"""
    Ntot = len(S)
    Ne = Ntot * n
    out = {}

    # ---------- 全局量：方差、直接膨胀因子、全局配对协方差 ----------
    mu = S.sum() / Ne                                   # E[e]
    var_e = Q.sum() / Ne - mu ** 2                      # Var(e)
    muS = S.sum() / Ntot
    var_S = (S ** 2).sum() / Ntot - muS ** 2            # Var(sum e)
    out["I_direct"] = var_S / (n * var_e)
    out["var_S"] = var_S
    cov_g = (((S - n * mu) ** 2 - (Q - 2 * mu * S + n * mu ** 2)).sum()) / (Ntot * n * (n - 1))
    out["cov_global"] = cov_g
    out["rho_global"] = cov_g / var_e
    out["I_from_rho"] = 1.0 + (n - 1) * out["rho_global"]
    out["Ee"] = mu
    out["var_e"] = var_e
    out["Ee_sq_over_var"] = mu ** 2 / var_e

    # ---------- 按 xi 分箱 ----------
    vals, inv, cnt = np.unique(xi, return_inverse=True, return_counts=True)
    keep = cnt >= MIN_BIN_BLOCKS
    if int(keep.sum()) >= 2:
        lut = {int(v): j for j, v in enumerate(vals[keep])}   # xi 可为负，不能直接做下标
        newinv = np.fromiter((lut.get(int(v), -1) for v in xi), dtype=np.int64, count=len(xi))
        sel = newinv >= 0
        inv = newinv[sel]
        S, Q, G, s2, n0, ndz = S[sel], Q[sel], G[sel], s2[sel], n0[sel], ndz[sel]
        cnt = np.bincount(inv, minlength=len(lut))
        bin_xi = np.array(sorted(lut, key=lambda k: lut[k]), dtype=np.int64)
    else:
        inv = np.zeros_like(xi)
        cnt = np.array([len(xi)])
        bin_xi = np.array([int(xi[0])])
    nb = len(cnt)
    w = cnt / float(cnt.sum())

    Sb = np.bincount(inv, weights=S, minlength=nb)
    Qb = np.bincount(inv, weights=Q, minlength=nb)
    Gb = np.bincount(inv, weights=G, minlength=nb)
    s2b = np.bincount(inv, weights=s2, minlength=nb)
    n0b = np.bincount(inv, weights=n0.astype(float), minlength=nb)
    ndzb = np.bincount(inv, weights=ndz.astype(float), minlength=nb)
    SSb = np.bincount(inv, weights=S ** 2, minlength=nb)       # 每块 (sum e)^2 按箱求和

    m_b = Sb / (cnt * n)                                        # m(xi) 的箱估计
    m_bar = float((w * m_b).sum())
    # 箱内元素方差 E[(e-m_b)^2 | xi]
    v_b = (Qb - 2 * m_b * Sb + cnt * n * m_b ** 2) / (cnt * n)
    # 箱内配对协方差 c(xi) = E[(e_i-m_b)(e_j-m_b) | xi]
    c_b = (SSb - 2 * n * m_b * Sb + cnt * n ** 2 * m_b ** 2
           - (Qb - 2 * m_b * Sb + cnt * n * m_b ** 2)) / (cnt * n * (n - 1))
    s_b = np.sqrt(s2b / cnt)                                    # 该箱的 s（箱内为常数）
    p_dz_b = ndzb / (cnt * n)                                   # 死区比例
    p0_b = n0b / (cnt * n)                                      # 精确零比例

    var_m_raw = float((w * (m_b - m_bar) ** 2).sum())
    noise = float((w * v_b / (cnt * n)).sum())                  # 箱均值的估计噪声
    var_m = var_m_raw - noise
    # 只在"保留的箱"上做严格恒等式核对：丢弃的稀疏箱会破坏与全局量的恒等
    kept_mass = float(cnt.sum()) / Ntot
    var_e_kept = Qb.sum() / (cnt.sum() * n) - m_bar ** 2
    var_S_kept = SSb.sum() / cnt.sum() - (Sb.sum() / cnt.sum()) ** 2

    out.update(dict(
        nbins=nb, min_bin=int(cnt.min()), max_bin=int(cnt.max()),
        m_bar=m_bar, var_m=var_m, var_m_raw=var_m_raw, var_m_noise=noise,
        E_var_within=float((w * v_b).sum()),
        E_cov_within=float((w * c_b).sum()),
        # 下面三条用未修正的 Var(m)_raw，才是严格恒等式的左端
        cov_from_decomp=float((w * c_b).sum()) + var_m_raw,
        var_e_from_decomp=float((w * v_b).sum()) + var_m_raw,
        # Var(ebar) = Var(m) + E[Var(e|xi)]/n + (n-1)E[c(xi)]/n   （最后一项即 A2 违背的贡献）
        var_ebar_decomp=var_m_raw + float((w * v_b).sum()) / n
        + (n - 1) * float((w * c_b).sum()) / n,
        I_from_var_m=1.0 + (n - 1) * var_m / var_e,
        rho_var_m=var_m / var_e,
        # 块均值的精确分解： Var(ebar) = Var(m) + E[Var(e|xi)]/n（在保留箱上）
        var_ebar=var_S_kept / n ** 2, var_ebar_global=var_S / n ** 2,
        kept_mass=kept_mass, var_e_kept=var_e_kept,
    ))

    # ---------- 方差公式与"概率原子" ----------
    Es2 = s2.sum() / Ntot
    Es2_g2 = (s2 * G).sum() / Ne
    out["gamma_bar"] = Es2_g2 / Es2                      # E[s^2 gamma]/E[s^2]
    out["E2_ratio"] = Es2_g2 / (Es2 / 12.0)              # E[e^2] / (E[s^2]/12)
    out["var_ratio"] = var_e / (Es2 / 12.0)              # Var(e) / (E[s^2]/12)  ← 与 v2 同定义
    out["Es2"] = Es2
    out["p0"] = n0b.sum() / Ne                           # 可精确表示原子的比例
    out["p_dz"] = ndzb.sum() / Ne                        # 死区比例
    out["atom_pred"] = 1.0 - out["p0"]                   # 原子解释给出的比值预测

    # ---------- 严格不等式核对 ----------
    out["bound_hi"] = float(n)                           # Cauchy-Schwarz 上界
    out["false_bound"] = n * mu ** 2 / var_e             # 被否定的"偏置下界"

    if keep_bins:
        out["bins"] = [
            dict(xi=int(bin_xi[j]), nb=int(cnt[j]), s=float(s_b[j]), m=float(m_b[j]),
                 delta=float(m_b[j] / s_b[j]) if s_b[j] > 0 else float("nan"),
                 v=float(v_b[j]), c=float(c_b[j]), p_dz=float(p_dz_b[j]), p0=float(p0_b[j]))
            for j in range(nb)
        ]
    return out


def run(dist, Lm, n, mode, seed, keep_bins=False):
    st = collect(dist, Lm, n, mode, seed)
    out = analyse(*st, n, Lm, keep_bins=keep_bins)
    out["_s2"] = st[4]
    return out


# ---------------- 退化模型：delta(s) 与 gamma(s) 的一维查表（按 (dist,s) 缓存） ----------------
_SHAPE_CACHE = {}
SHAPE_N = 4_000_000


def shape_functions(dist, s):
    key = (dist, round(float(s), 12))
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]
    rng = np.random.default_rng(11)
    x = make_samples(dist, SHAPE_N, rng)
    r = np.round(x / s) - x / s
    val = (float(r.mean()), float((r * r).mean()))
    _SHAPE_CACHE[key] = val
    return val


def reduced_model(dist, Lm, n, s2, n_grid=4096):
    """退化模型预言： m(xi)=s*delta(s), E[e^2|xi]=s^2*gamma(s)，s=2^{floor(log2 M)-Lm}。

    s 的分布按块最大值的分布（即次序统计量）取，delta/gamma 只依赖 F 与 s。
    """
    s = np.sqrt(s2)
    grid, cnts = np.unique(np.round(s, 12), return_counts=True)
    w = cnts / cnts.sum()
    mu = 0.0
    Es2g = 0.0
    Em2 = 0.0
    for p, sv in zip(w, grid):
        d, g = shape_functions(dist, sv)
        mu += p * sv * d
        Es2g += p * sv * sv * g
        Em2 += p * (sv * d) ** 2
    var_m = Em2 - mu ** 2
    var_e = max(Es2g - mu ** 2, 1e-300)
    return dict(model_I=1.0 + (n - 1) * var_m / var_e, model_mu=mu,
                model_var_e=var_e, model_var_m=var_m, model_Es2g=Es2g)


def main():
    seed = 20260920
    print("=" * 116)
    print("v7：条件化于共享指数 xi 的直接分解   Cov(e_i,e_j) = E[Cov(e_i,e_j|xi)] + Var(m(xi))")
    print(f"  {N_GROUPS} 组 x {NB_PER_GROUP} 块；分箱：xi 的整数值；箱内块数 >= {MIN_BIN_BLOCKS}")
    print("=" * 116)

    rows = []
    for Lm in LM_LIST:
        for mode in ("rne", "sr"):
            print()
            print(f"########## Lm = {Lm}   舍入 = {mode.upper()} ##########")
            head = (f"{'dist':<11}{'n':>6}{'bins':>5}{'I_direct':>10}{'I_rho':>10}{'I_var_m':>10}"
                    f"{'rho':>9}{'Var(m)':>11}{'E[Var|xi]':>11}{'E[cov|xi]':>11}"
                    f"{'Var(e)':>10}{'ratio':>8}{'p0':>7}{'p_dz':>7}")
            print(head)
            print("-" * len(head))
            for dist in DISTS:
                for n in N_LIST:
                    r = run(dist, Lm, n, mode, seed, keep_bins=(n == N_LIST[-1] and mode == "rne"))
                    if mode == "rne":
                        r.update(reduced_model(dist, Lm, n, r.pop("_s2")))
                        rows.append(dict(dist=dist, Lm=Lm, mode=mode, n=n, **r))
                        print(f"{dist:<11}{n:>6}{r['nbins']:>5}{r['I_direct']:>10.4f}{r['I_from_rho']:>10.4f}"
                              f"{r['I_from_var_m']:>10.4f}{r['rho_var_m']:>9.5f}{r['var_m']:>11.3e}"
                              f"{r['E_var_within']:>11.3e}{r['E_cov_within']:>11.3e}{r['var_e']:>10.3e}"
                              f"{r['var_ratio']:>8.4f}{r['p0']:>7.4f}{r['p_dz']:>7.4f}")
                    else:
                        r.pop("_s2", None)
                        rows.append(dict(dist=dist, Lm=Lm, mode=mode, n=n, **r))
                        print(f"{dist:<11}{n:>6}{r['nbins']:>5}{r['I_direct']:>10.4f}{r['I_from_rho']:>10.4f}"
                              f"{r['I_from_var_m']:>10.4f}{r['rho_var_m']:>9.5f}{r['var_m']:>11.3e}"
                              f"{r['E_var_within']:>11.3e}{r['E_cov_within']:>11.3e}{r['var_e']:>10.3e}"
                              f"{r['var_ratio']:>8.4f}{r['p0']:>7.4f}{r['p_dz']:>7.4f}")
                print("-" * len(head))

    # ---------------- 判据 ----------------
    print()
    print("=" * 116)
    print("判据 1：三条独立路径给出的 I_n 是否一致？")
    print("=" * 116)
    print(f"  max |I_direct - I_(1+(n-1)rho)|     = "
          f"{max(abs(r['I_direct'] - r['I_from_rho']) for r in rows):.4f}")
    print(f"  max |I_direct - I_(由 Var(m) 反推)|  = "
          f"{max(abs(r['I_direct'] - r['I_from_var_m']) for r in rows):.4f}")

    print()
    print("=" * 116)
    print("判据 2：三条精确恒等式的数值闭合误差（用未做噪声修正的 Var(m)，规模按 Var(e) 归一）")
    print("=" * 116)
    e1 = max(abs(r["var_e_from_decomp"] - r["var_e_kept"]) / r["var_e_kept"] for r in rows)
    e2 = max(abs(r["var_ebar_decomp"] - r["var_ebar"]) / r["var_ebar"] for r in rows)
    e3 = max(abs(r["cov_from_decomp"] - r["cov_global"]) / r["var_e_kept"] for r in rows)
    e4 = max(abs(r["var_e_kept"] / r["var_e"] - 1.0) for r in rows)
    e6 = min(r["kept_mass"] for r in rows)
    e5 = max(abs(r["var_m_noise"] / r["var_e_kept"]) for r in rows)
    print(f"  Var(e)      = E[Var(e|xi)] + Var(m(xi))          : {e1:.3e}  （在保留箱上恒等）")
    print(f"  Var(ebar)   = Var(m) + E[Var|xi]/n + (n-1)E[c]/n   : {e2:.3e}")
    print(f"  Cov(e_i,e_j)= E[cov(e_i,e_j|xi)] + Var(m(xi))     : {e3:.3e}")
    print(f"  保留箱的最小质量占比                              : {e6:.4f}")
    print(f"  丢弃稀疏箱带来的偏差（Var(e) 相对 / Var(ebar) 相对）: {e4:.3e} / "
          f"{max(abs(r['var_ebar'] / r['var_ebar_global'] - 1.0) for r in rows):.3e}")
    print(f"  箱均值噪声修正量的规模 Var(m)_noise/Var(e)        : {e5:.3e}")

    print()
    print("=" * 116)
    print("判据 3：严格上界 0 <= I_n <= n 是否满足；以及被否定的'偏置下界'")
    print("=" * 116)
    print(f"  违反 0 <= I_n 的配置：{sum(1 for r in rows if r['I_direct'] < -1e-6)} / {len(rows)}")
    print(f"  违反 I_n <= n 的配置：{sum(1 for r in rows if r['I_direct'] > r['n'] + 1e-6)} / {len(rows)}")
    print("  若'Cov >= (E[e])^2'成立，则 I_n >= n*(E[e])^2/Var(e)。反例：")
    bad = sorted([r for r in rows if r["mode"] == "rne" and r["false_bound"] > r["I_direct"]],
                 key=lambda r: -(r["false_bound"] / r["I_direct"]))[:6]
    print(f"  {'dist':<11}{'Lm':>4}{'n':>6}{'I_n':>10}{'伪下界':>12}{'伪下界/I_n':>11}{'Var(m)':>11}{'(E[e])^2':>11}")
    for r in bad:
        print(f"  {r['dist']:<11}{r['Lm']:>4}{r['n']:>6}{r['I_direct']:>10.4f}{r['false_bound']:>12.2f}"
              f"{r['false_bound'] / r['I_direct']:>11.2f}{r['var_m']:>11.3e}{r['Ee'] ** 2:>11.3e}")
    print(f"  共有 {sum(1 for r in rows if r['mode'] == 'rne' and r['false_bound'] > r['I_direct'])}"
          f" / {sum(1 for r in rows if r['mode'] == 'rne')} 个 RNE 配置的反例（按双精度比较容差 1e-9）")

    print()
    print("=" * 116)
    print("判据 4：谁贡献了块内相关？E[Cov(e_i,e_j|xi)]（即假设 A2 的违背量）对 Var(m(xi))")
    print("=" * 116)
    print(f"{'dist':<11}{'Lm':>4}{'n':>6}{'Cov 总计':>13}{'E[cov|xi]':>13}{'Var(m)':>13}"
          f"{'A2 占比':>10}{'rho':>9}")
    print("-" * 82)
    for r in rows:
        if r["mode"] != "rne":
            continue
        tot = r["cov_from_decomp"]
        share = r["E_cov_within"] / tot if abs(tot) > 1e-15 else float("nan")
        print(f"{r['dist']:<11}{r['Lm']:>4}{r['n']:>6}{tot:>13.3e}{r['E_cov_within']:>13.3e}"
              f"{r['var_m']:>13.3e}{share:>10.3f}{r['rho_var_m']:>9.5f}")

    print()
    print("=" * 116)
    print("判据 5：E[s^2]/12 偏差的原子解释（实测比值是否命中 1 - p0？）")
    print("=" * 116)
    print(f"{'dist':<11}{'Lm':>4}{'mode':>5}{'n':>6}{'实测/预测':>12}{'1-p0':>9}{'gamma_bar':>11}"
          f"{'差':>9}{'p_dz':>8}")
    print("-" * 82)
    for r in rows:
        print(f"{r['dist']:<11}{r['Lm']:>4}{r['mode']:>5}{r['n']:>6}{r['var_ratio']:>12.4f}"
              f"{r['atom_pred']:>9.4f}{r['gamma_bar']:>11.5f}{r['var_ratio'] - r['atom_pred']:>9.4f}"
              f"{r['p_dz']:>8.4f}")

    print()
    print("=" * 116)
    print("判据 6：逐箱诊断 —— delta(xi)=m(xi)/s 是否近似为常数？死区比例是否解释 L_m 依赖？")
    print("=" * 116)
    for Lm in LM_LIST:
        for dist in DISTS:
            r = next(x for x in rows if x["dist"] == dist and x["Lm"] == Lm
                     and x["n"] == N_LIST[-1] and x["mode"] == "rne")
            print(f"\n  --- {dist}  Lm={Lm}  n={N_LIST[-1]}   I_n={r['I_direct']:.3f}  "
                  f"Var(m)={r['var_m']:.3e}  E[cov|xi]={r['E_cov_within']:.3e} ---")
            print(f"  {'xi':>4}{'s':>10}{'块数':>8}{'m(xi)':>12}{'delta=m/s':>12}"
                  f"{'Var(e|xi)':>12}{'c(xi)':>12}{'p_dz':>8}{'p0':>8}")
            for b in r["bins"]:
                print(f"  {b['xi']:>4}{b['s']:>10.4f}{b['nb']:>8}{b['m']:>12.4e}{b['delta']:>12.4f}"
                      f"{b['v']:>12.3e}{b['c']:>12.3e}{b['p_dz']:>8.3f}{b['p0']:>8.3f}")


    print()
    print("=" * 116)
    print("判据 7：退化模型的无拟合预言  I_n 与 E[e] 是否被判据 6 的机制再现？")
    print("    模型： m(xi)=s*delta(s)，E[e^2|xi]=s^2*gamma(s)，")
    print("           delta(s)=E[RN(x/s)-x/s]、gamma(s)=E[(RN(x/s)-x/s)^2]，x~F（一维，可按求积算）")
    print("           Var(e)=E[s^2 gamma]-(E[s delta])^2，Cov=Var(s*delta)，s 的分布取次序统计量 max|x|")
    print("=" * 116)
    print(f"{'dist':<11}{'Lm':>4}{'n':>6}{'I 实测':>11}{'I 模型':>11}{'相对误差':>11}"
          f"{'E[e] 实测':>13}{'E[e] 模型':>13}{'Var(e) 实测':>13}{'Var(e) 模型':>13}")
    print("-" * 96)
    rel = []
    for r in rows:
        if r["mode"] != "rne":
            continue
        err = (r["model_I"] - r["I_direct"]) / r["I_direct"]
        rel.append(abs(err))
        print(f"{r['dist']:<11}{r['Lm']:>4}{r['n']:>6}{r['I_direct']:>11.4f}{r['model_I']:>11.4f}"
              f"{err:>11.2%}{r['Ee']:>13.4e}{r['model_mu']:>13.4e}"
              f"{r['var_e']:>13.4e}{r['model_var_e']:>13.4e}")
    print("-" * 96)
    print(f"  |I_model/I_measured - 1| 的最大值：{max(rel):.2%}   中位数：{np.median(rel):.2%}")
    big = [r for r in rows if r["mode"] == "rne" and r["I_direct"] > 1.2]
    print(f"  仅在 I_n > 1.2 的 {len(big)} 个（真正有膨胀的）配置上：")
    print(f"     最大相对误差 {max(abs((r['model_I'] - r['I_direct']) / r['I_direct']) for r in big):.2%}")


if __name__ == "__main__":
    main()

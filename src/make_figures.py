"""
生成论文图表。数据取自 math_pilot/bfp_error_pilot_v2~v6.py 的实际运行结果。
所有数值均为 20 组 x 1000 块（或 4e4 块）实测，见论文附录的可复现说明。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

DISTS = ["Gaussian", "Student-$t_3$", "Spike mixture", "Lognormal", "Half-normal", "ReLU"]
SYM = [True, True, True, False, False, False]
COL = ["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52", "#DD8452", "#937860"]
LMS = [2, 3, 4, 7]

# ---- 表: 偏置 z 值 (4e4 块) ----
Z = {
    "Gaussian":       [1.63, -0.93, -0.48, 0.98],
    "Student-$t_3$":  [0.39, -1.40, -1.55, 0.35],
    "Spike mixture":  [0.10, -0.31, -0.28, 0.90],
    "Lognormal":      [-126.80, -61.69, -19.92, 0.27],
    "Half-normal":    [-60.57, -30.96, -15.22, -1.80],
    "ReLU":           [-39.48, -19.45, -9.26, -1.31],
}

# ---- 表: emp/pred 方差比 ----
VR = {
    "Gaussian":       [0.999, 0.999, 0.999, 1.001],
    "Student-$t_3$":  [0.808, 0.913, 0.914, 1.002],
    "Spike mixture":  [0.093, 0.148, 0.319, 1.002],
    "Lognormal":      [0.854, 1.011, 1.043, 1.002],
    "Half-normal":    [0.996, 0.999, 1.000, 1.001],
    "ReLU":           [0.506, 0.507, 0.506, 0.506],
}

# ---- 表: 膨胀因子 I_n (v5, 20x1000 块) ----
NS = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
INFL = {
    2: {
        "Gaussian":    [1.000, 0.990, 0.985, 1.004, 0.997, 1.009, 1.026, 1.013],
        "Lognormal":   [3.075, 5.330, 9.809, 18.953, 31.341, 51.480, 102.555, 176.996],
        "ReLU":        [1.037, 1.016, 1.007, 1.028, 1.124, 1.520, 2.679, 6.132],
        "Half-normal": [1.015, 1.000, 1.043, 1.120, 1.441, 2.657, 6.321, 14.989],
    },
    4: {
        "Gaussian":    [0.996, 0.967, 0.984, 0.997, 1.009, 0.987, 1.013, 0.989],
        "Lognormal":   [1.328, 2.047, 3.511, 7.265, 15.601, 34.748, 76.063, 165.025],
        "ReLU":        [1.005, 1.016, 0.986, 1.005, 1.026, 1.030, 1.084, 1.300],
        "Half-normal": [0.996, 0.993, 1.020, 0.991, 1.013, 1.096, 1.284, 1.834],
    },
}

# ---- 表: RNE vs SR, n=4096 (v6) ----
SRV = {
    2: {"Gaussian": (1.004, 0.988), "Lognormal": (176.051, 0.988),
        "ReLU": (6.111, 1.009), "Half-normal": (14.871, 0.994)},
    4: {"Gaussian": (1.009, 0.975), "Lognormal": (168.274, 0.991),
        "ReLU": (1.290, 1.000), "Half-normal": (1.775, 0.991)},
}

# ---- 表: 舍入规则敏感性, n=32, Lm=4 (v3) ----
MODES = ["RNE", "Half-away", "Truncate", "Stochastic"]
MODE_Z = {
    "Lognormal":   [-25.33, -24.68, -422.71, -0.98],
    "ReLU":        [-11.29, -13.16, -603.77, -0.81],
    "Half-normal": [-21.08, -18.23, -982.14, 0.97],
}


def fig1_bias():
    fig, ax = plt.subplots(figsize=(6.2, 2.05))
    x = np.arange(len(DISTS))
    w = 0.2
    for k, Lm in enumerate(LMS):
        vals = [abs(Z[d][k]) for d in DISTS]
        ax.bar(x + (k - 1.5) * w, vals, w, label=f"$L_m={Lm}$",
               color=plt.cm.viridis(k / 3.5), edgecolor="k", linewidth=0.3)
    ax.axhline(4, color="crimson", ls="--", lw=1.0, zorder=5)
    ax.text(len(DISTS) - 0.45, 5.2, "significance threshold $|z|=4$",
            color="crimson", fontsize=7.5, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 900)
    ax.set_xticks(x)
    ax.set_xticklabels(DISTS, fontsize=8)
    for t, s in zip(ax.get_xticklabels(), SYM):
        t.set_color("#4C72B0" if s else "#C44E52")
    ax.set_ylabel("$|z|$ of bias", fontsize=8)
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    fig.savefig(f"{OUT}/fig1_bias.pdf")
    plt.close(fig)


def fig2_var():
    fig, ax = plt.subplots(figsize=(6.2, 2.05))
    x = np.arange(len(DISTS))
    w = 0.2
    for k, Lm in enumerate(LMS):
        vals = [VR[d][k] for d in DISTS]
        ax.bar(x + (k - 1.5) * w, vals, w, label=f"$L_m={Lm}$",
               color=plt.cm.viridis(k / 3.5), edgecolor="k", linewidth=0.3)
    ax.axhline(1, color="k", ls="--", lw=1.0, zorder=5)
    ax.text(len(DISTS) - 0.45, 1.03, "classical prediction", fontsize=7.5, ha="right")
    ax.set_ylim(0, 1.30)
    ax.set_xticks(x)
    ax.set_xticklabels(DISTS, fontsize=8)
    for t, s in zip(ax.get_xticklabels(), SYM):
        t.set_color("#4C72B0" if s else "#C44E52")
    ax.set_ylabel("empirical / predicted var.", fontsize=8)
    ax.legend(ncol=4, fontsize=7, loc="lower left")
    fig.savefig(f"{OUT}/fig2_variance.pdf")
    plt.close(fig)


def fig3_inflation():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, Lm in zip(axes, (2, 4)):
        for d in ["Gaussian", "Lognormal", "ReLU", "Half-normal"]:
            c = "#4C72B0" if d == "Gaussian" else COL[DISTS.index(d)]
            ls = "--" if d == "Gaussian" else "-"
            ax.plot(NS, INFL[Lm][d], marker="o", ms=3, lw=1.2, color=c, ls=ls, label=d)
        ax.plot(NS, NS / NS[0] * 1.0, color="gray", lw=0.8, ls=":", zorder=0)
        ax.plot(NS, (NS / NS[0]) ** 0.5, color="gray", lw=0.8, ls="-.", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("block size $n$")
        ax.set_title(f"$L_m = {Lm}$", fontsize=9)
    axes[0].set_ylabel("variance inflation factor $\\mathcal{I}_n$")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].text(40, 250, "$\\propto n$", color="gray", fontsize=7)
    axes[1].text(1500, 1.6, "$\\propto\\sqrt{n}$", color="gray", fontsize=7, ha="right")
    fig.savefig(f"{OUT}/fig3_inflation.pdf")
    plt.close(fig)


def fig4_sr():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    names = ["Gaussian", "Lognormal", "ReLU", "Half-normal"]
    x = np.arange(len(names))
    for ax, Lm in zip(axes, (2, 4)):
        rne = [SRV[Lm][d][0] for d in names]
        sr = [SRV[Lm][d][1] for d in names]
        ax.bar(x - 0.19, rne, 0.36, label="RNE", color="#C44E52",
               edgecolor="k", linewidth=0.3)
        ax.bar(x + 0.19, sr, 0.36, label="stochastic rounding", color="#55A868",
               edgecolor="k", linewidth=0.3)
        ax.axhline(1, color="k", ls="--", lw=1.0, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 400)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=12)
        ax.set_title(f"$L_m = {Lm}$,  $n = 4096$", fontsize=9)
    axes[0].set_ylabel("$\\mathcal{I}_n$")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.savefig(f"{OUT}/fig4_sr.pdf")
    plt.close(fig)


def fig5_modes():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    x = np.arange(len(MODES))
    w = 0.26
    for i, (d, c) in enumerate(zip(["Lognormal", "ReLU", "Half-normal"],
                                   ["#C44E52", "#937860", "#DD8452"])):
        ax.bar(x + (i - 1) * w, [abs(v) for v in MODE_Z[d]], w, label=d,
               color=c, edgecolor="k", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3000)
    ax.set_xticks(x)
    ax.set_xticklabels(MODES, fontsize=8)
    ax.set_ylabel("$|z|$ of block-mean bias")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.savefig(f"{OUT}/fig5_modes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1_bias(); fig2_var(); fig3_inflation(); fig4_sr(); fig5_modes()
    for f in sorted(os.listdir(OUT)):
        print(f, os.path.getsize(os.path.join(OUT, f)), "bytes")

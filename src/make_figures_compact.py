"""
Layout variants of the paper figures, used to reduce float area without
dropping content.  Panels are the same data as make_figures.py.

Outputs (written into scripts/src/figures/):
  fig_biasvar.pdf   fig1 (|z| of bias) over fig2 (empirical/predicted
                    variance), stacked in one two-panel float
  fig_srmodes.pdf   fig4 (I_n under RNE vs SR) beside fig5 (|z| under four
                    rounding rules)

Both are redrawn from the same hard-coded arrays as make_figures.py; no
experiment is re-run.

Usage:
    python3 make_figures_compact.py                  # writes next to this script
    python3 make_figures_compact.py <outdir>         # writes into <outdir>
The manuscript uses the second form with the package's top-level figures/
directory, which is where paper.tex looks for them.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures as M

plt.rcParams.update(M.plt.rcParams)


def out_dir():
    """Directory the two PDFs are written to (argv[1] if given)."""
    d = sys.argv[1] if len(sys.argv) > 1 else M.OUT
    os.makedirs(d, exist_ok=True)
    return d


def fig_biasvar():
    out = out_dir()
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.1))
    x = np.arange(len(M.DISTS))
    w = 0.2

    ax = axes[0]
    for k, Lm in enumerate(M.LMS):
        ax.bar(x + (k - 1.5) * w, [abs(M.Z[d][k]) for d in M.DISTS], w,
               label=f"$L_m={Lm}$", color=plt.cm.viridis(k / 3.5),
               edgecolor="k", linewidth=0.3)
    ax.axhline(4, color="crimson", ls="--", lw=1.0, zorder=5)
    ax.text(len(M.DISTS) - 0.45, 5.4, "significance threshold $|z|=4$",
            color="crimson", fontsize=7, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 1200)
    ax.set_ylabel("$|z|$ of bias", fontsize=8)
    ax.legend(ncol=4, fontsize=7, loc="upper left")

    ax = axes[1]
    for k, Lm in enumerate(M.LMS):
        ax.bar(x + (k - 1.5) * w, [M.VR[d][k] for d in M.DISTS], w,
               label=f"$L_m={Lm}$", color=plt.cm.viridis(k / 3.5),
               edgecolor="k", linewidth=0.3)
    ax.axhline(1, color="k", ls="--", lw=1.0, zorder=5)
    ax.text(len(M.DISTS) - 0.45, 1.03, "classical prediction", fontsize=7, ha="right")
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("empirical / predicted var.", fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(M.DISTS, fontsize=8, rotation=12)
        for t, s in zip(ax.get_xticklabels(), M.SYM):
            t.set_color("#4C72B0" if s else "#C44E52")

    fig.subplots_adjust(hspace=0.42)
    fig.savefig(os.path.join(out, "fig_biasvar.pdf"))
    plt.close(fig)


def fig_srmodes():
    out = out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    names = ["Gaussian", "Lognormal", "ReLU", "Half-normal"]

    ax = axes[0]
    x = np.arange(len(names))
    rne = [M.SRV[2][d][0] for d in names]
    sr = [M.SRV[2][d][1] for d in names]
    ax.bar(x - 0.19, rne, 0.36, label="RNE", color="#C44E52",
           edgecolor="k", linewidth=0.3)
    ax.bar(x + 0.19, sr, 0.36, label="stochastic rounding", color="#55A868",
           edgecolor="k", linewidth=0.3)
    ax.axhline(1, color="k", ls="--", lw=1.0, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 400)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=12)
    ax.set_ylabel("$\\mathcal{I}_n$", fontsize=8)
    ax.set_title("$L_m = 2$,  $n = 4096$", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    x = np.arange(len(M.MODES))
    w = 0.26
    for i, (d, c) in enumerate(zip(["Lognormal", "ReLU", "Half-normal"],
                                   ["#C44E52", "#937860", "#DD8452"])):
        ax.bar(x + (i - 1) * w, [abs(v) for v in M.MODE_Z[d]], w, label=d,
               color=c, edgecolor="k", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3000)
    ax.set_xticks(x)
    ax.set_xticklabels(M.MODES, fontsize=8, rotation=12)
    ax.set_ylabel("$|z|$ of bias", fontsize=8)
    ax.set_title("$n = 32$,  $L_m = 4$", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")

    fig.subplots_adjust(wspace=0.30)
    fig.savefig(os.path.join(out, "fig_srmodes.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_biasvar()
    fig_srmodes()
    d = out_dir()
    for f in ("fig_biasvar.pdf", "fig_srmodes.pdf"):
        print(f, "written to", d)

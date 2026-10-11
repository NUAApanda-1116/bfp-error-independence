"""Optional scientific figure from retained coefficient and precision maps."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((OUT/"highcontrast_diffusion_prespecification.json").read_text(encoding="utf-8"))
    saved = np.load(OUT/"highcontrast_diffusion_samples.npz")
    mesh = manifest["mesh"]
    maps = []
    for policy in ("source", "energy"):
        cfg = next(cfg for cfg in manifest["configurations"] if cfg["id"] == f"b32_contiguous_{policy}")
        values = np.zeros(mesh*mesh, dtype=int)
        for ids, level in zip(cfg["block_indices"], cfg["L_by_block"]):
            values[ids] = level
        maps.append(values.reshape(mesh, mesh))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 1.8), constrained_layout=True)
    extent = (.5/(mesh+1), (mesh+.5)/(mesh+1), .5/(mesh+1), (mesh+.5)/(mesh+1))
    cmap = ListedColormap(["#ededed", "#1764a0"])
    norm = BoundaryNorm([2.5, 3.5, 4.5], cmap.N)
    axes[0].imshow(np.log10(saved["conductivity"].T), origin="lower", extent=extent,
                   cmap=ListedColormap(["#666666", "#eeeeee"]), vmin=-2, vmax=0., interpolation="nearest")
    axes[0].set_title(r"$\log_{10}\alpha$")
    axes[0].text(.71, .30, r"$-2$", color="white", ha="center", va="center", fontsize=7)
    axes[0].text(.25, .28, r"$0$", ha="center", fontsize=7)
    for ax, values, title in zip(axes[1:], maps, ("Source scale placement", "Full energy placement")):
        ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
    for ax in axes:
        ax.add_patch(Circle((.7, .3), np.sqrt(.05), fill=False, ls="--", lw=.8, edgecolor="#c1453f"))
        ax.add_patch(Circle((.3, .65), np.sqrt(.02*np.log(2)), fill=False, ls=":", lw=.9,
                           edgecolor="#1f7a35", path_effects=[pe.withStroke(linewidth=1.6, foreground="white")]))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.set_xlabel("$x$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$y$")
    handles = [Line2D([0], [0], color="#c1453f", lw=.8, ls="--", label=r"$\alpha=.01$ region"),
               Line2D([0], [0], color="#1f7a35", lw=.9, ls=":", label="Source half amplitude"),
               Line2D([0], [0], color="#1764a0", lw=4, label=r"$L=4$"),
               Line2D([0], [0], color="#ededed", lw=4, label=r"$L=3$")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               fontsize=6.5, handlelength=1.8, columnspacing=1.)
    fig.savefig(OUT/"highcontrast_precision_maps.pdf")
    fig.savefig(OUT/"highcontrast_precision_maps.png", dpi=240)
    plt.close(fig)
    print("Generated optional coefficient/precision-map figure from retained declarations.")


if __name__ == "__main__":
    main()

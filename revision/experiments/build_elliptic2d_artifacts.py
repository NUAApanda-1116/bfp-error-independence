"""Build concise manuscript artifacts from retained primary results only."""
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read_csv(OUT/"elliptic2d_primary_results.csv")
    comparisons = read_csv(OUT/"elliptic2d_primary_paired_comparisons.csv")
    manifest = json.loads((OUT/"elliptic2d_replication_manifest.json").read_text(encoding="utf-8"))
    original = json.loads((OUT/"elliptic2d_manifest.json").read_text(encoding="utf-8"))
    table = []
    for profile in ("smooth", "hotspot"):
        for size, policy in ((8, "uniform"), (32, "uniform"), (32, "source"),
                             (32, "energy"), (128, "uniform"), (128, "energy")):
            row = next(row for row in rows if int(row["mesh"]) == 32 and row["profile"] == profile
                       and row["id"] == f"b{size}_contiguous_{policy}")
            label = "Smooth" if profile == "smooth" else "Hotspot"
            policy_label = {"uniform": "Uniform", "source": "Source", "energy": "Energy"}[policy]
            values = [100*float(row[key+"_relative_rms"]) for key in ("exact", "independent", "uniform")]
            low, high = 100*float(row["mc_relative_rms_low"]), 100*float(row["mc_relative_rms_high"])
            table.append(f"{label} & {size} & {policy_label} & {values[0]:.3f} & {values[1]:.3f} & {values[2]:.3f} & [{low:.3f},{high:.3f}] " + r"\\")
    table_tex = r"""\begin{table}[tb]
\centering
\caption{Two-dimensional heterogeneous source compression, $32^2$ unknowns,
contiguous blocks, and exactly $7168$ payload bits. Source and Energy allocate
the extra coefficient bits by source scale and the full energy prediction,
respectively. Relative energy RMS values are percentages. Ind. retains exact
marginal moments and bias; Unif. uses independent uniform-cell noise.}
\label{tab:revision-elliptic2d}
\begingroup\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrlrrrr}
\toprule
Profile & $|B|$ & Placement & Exact & Ind. & Unif. & MC $95\%$ interval\\
\midrule
__TABLE__
\bottomrule
\end{tabular}\endgroup
\end{table}
""".replace("__TABLE__", "\n".join(table))
    (OUT.parent/"table_elliptic2d.tex").write_text(table_tex, encoding="utf-8")
    chosen = next(row for row in comparisons if row["mesh"] == "32" and row["profile"] == "smooth"
                  and row["baseline"] == "b32_contiguous_uniform" and row["candidate"] == "b32_contiguous_energy")
    exact_vs_source = next(row for row in comparisons if row["mesh"] == "32" and row["profile"] == "hotspot"
                          and row["baseline"] == "b32_contiguous_source" and row["candidate"] == "b32_contiguous_energy")
    uniform_choice = next(row for row in comparisons if row["mesh"] == "32" and row["profile"] == "hotspot"
                          and row["baseline"] == "b8_contiguous_uniform" and row["candidate"] == "b32_contiguous_energy")
    largest_covariance_gap = max(rows, key=lambda row: float(row["exact_energy_mse"])/float(row["independent_energy_mse"]))
    percentage = lambda value: f"{100*float(value):.2f}"
    section = r"""\subsection{Spatially heterogeneous sources in an elliptic solve}
\label{sec:revision-elliptic2d}
Consider $-\Delta u=f$ on $(0,1)^2$ with homogeneous Dirichlet conditions.
On an $m\times m$ interior grid, let $h=(m+1)^{-1}$,
$A=h^{-2}(I\otimes T+T\otimes I)$ with
$T=\operatorname{tridiag}(-1,2,-1)$, and $G=A^{-1}$.
We compress the cell sources $b_i=s_i\exp(Z_i)$, where the $Z_i$ are
independent $N(0,1)$ variables. The prescribed spatial scales are
\begin{align*}
 s_{\rm sm}(x,y)&=1+3\sin^2(\pi x)\sin^2(\pi y),\\
 s_{\rm hot}(x,y)&=1+7\exp\!\left(-\frac{(x-0.3)^2+(y-0.65)^2}{0.02}\right).
\end{align*}
These profiles test the nonidentical-input formula with smooth variation
and a localized source hotspot.

Write $e=Q(b)-b$, $\delta u=Ge$, $\mu=\mathbb E e$, and
$\Sigma_B=\operatorname{Cov}(e_B)$ for a partition into disjoint blocks.
For any symmetric positive definite $A$, define $\|z\|_{A,h}^2=h^2z^TAz$.
Independence across blocks gives
\begin{equation}\label{eq:revision-elliptic2d-energy}
 J:=\mathbb E\|\delta u\|_{A,h}^2
 =h^2\left[\sum_B\operatorname{tr}(G_{BB}\Sigma_B)
                 +\mu^TG\mu\right].
\end{equation}
The bias term couples different spatial blocks. Relative RMS is
$\sqrt{J/\{h^2\mathbb E[b^TGb]\}}$, with the denominator evaluated
from the source laws. The baseline denoted Ind. retains every exact
marginal mean and variance while dropping off-diagonal error covariances;
Unif. uses independent zero-mean uniform-cell errors.

All codecs pack exactly $7m^2$ payload bits: an 8-bit signed exponent per
block, $L_B+3$ signed bits per coefficient, and any remaining padding.
Block lengths are $8,32,128$, with contiguous and interleaved layouts.
The largest common precision is supplemented by prescribed extra-bit
blocks. Their locations are fixed codec maps, so no data-dependent flags
are needed. At $m=32$, we compare evenly spaced locations (Uniform),
descending $\sum_{i\in B}s_i^2$ (Source), and a greedy full-energy rule
(Energy). For a candidate precision increment, embed its mean change
$d$ in the full vector and let $\Delta\Sigma_B$ be its block covariance
change. The greedy rule chooses the smallest
\begin{equation}\label{eq:revision-elliptic2d-increment}
 \Delta J=h^2\left[\operatorname{tr}(G_{BB}\Delta\Sigma_B)
                     +2d^TG\mu+d^TGd\right].
\end{equation}
Ties are resolved by block index. The implemented codec rejects
out-of-range exponents; none occurred, and the per-vector probability
bound is below $10^{-1632}$.

We retain all $40$ prescribed configurations, including coincident
precision maps, and use $20\,000$ paired source vectors per profile for
$m=16$ and $60\,000$ for $m=32$, with $300$ whole-vector bootstrap
replicates. The latter pools the initial $10\,000$ vectors and a declared
independent $50\,000$-vector replication for both profiles. All streams,
configuration records, and interval misses remain in the package.
The primary nominal intervals contain __COVERAGE__ of $40$ population RMS predictions;
the largest RMS discrepancy is __MAXERR__\%.

\input{revision/table_elliptic2d}

For the smooth profile at $m=32$, length-$32$ Energy placement reduces
measured RMS by __PLACEMENT__\% relative to Uniform placement
(paired $95\%$ interval $[__PLACEMENTLO__,__PLACEMENTHI__]\%$).
For the hotspot, it improves on Source placement by __SOURCE__\%
($[__SOURCELO__,__SOURCEHI__]\%$). These are gains from the full-model
placement objective. The uniform-cell model selects length $32$ with
Energy placement for the hotspot, whereas the exact and marginal-moment
models select length $8$. The former choice has __CHOICE__\% higher
measured RMS ($[__CHOICELO__,__CHOICEHI__]\%$), so the population model
also supports a concrete accuracy decision at the same storage budget.
The largest covariance gap occurs for the $16^2$ smooth case with
length-$128$ contiguous blocks: exact, Ind., and measured RMS are
__LARGESTEXACT__\%, __LARGESTIND__\%, and __LARGESTMC__\%, respectively;
the diagonal-moment model underestimates energy MSE by a factor __LARGESTFACTOR__.

For each codec, $100$ paired original and compressed sources are also
solved by conjugate gradients at relative tolerance $10^{-11}$.
For $r=A\widehat{\delta u}-e$, the solver discrepancy satisfies
$\|\widehat{\delta u}-\delta u\|_{A,h}\le h\|r\|_2/
\sqrt{\lambda_{\min}(A)}$; its maximum relative certificate is
$2.8\times10^{-9}$, and the CG and spectral energy values differ by
at most $1.3\times10^{-10}$ relatively.
"""
    replacements = {
        "__COVERAGE__": str(manifest["primary_exact_rms_ci_coverage"]),
        "__MAXERR__": percentage(manifest["primary_max_relative_rms_prediction_error"]),
        "__PLACEMENT__": percentage(chosen["mc_fractional_rms_reduction"]),
        "__PLACEMENTLO__": percentage(chosen["mc_fractional_rms_reduction_low"]),
        "__PLACEMENTHI__": percentage(chosen["mc_fractional_rms_reduction_high"]),
        "__SOURCE__": percentage(exact_vs_source["mc_fractional_rms_reduction"]),
        "__SOURCELO__": percentage(exact_vs_source["mc_fractional_rms_reduction_low"]),
        "__SOURCEHI__": percentage(exact_vs_source["mc_fractional_rms_reduction_high"]),
        "__CHOICE__": percentage(-float(uniform_choice["mc_fractional_rms_reduction"])),
        "__CHOICELO__": percentage(-float(uniform_choice["mc_fractional_rms_reduction_high"])),
        "__CHOICEHI__": percentage(-float(uniform_choice["mc_fractional_rms_reduction_low"])),
        "__LARGESTEXACT__": f"{100*float(largest_covariance_gap['exact_relative_rms']):.3f}",
        "__LARGESTIND__": f"{100*float(largest_covariance_gap['independent_relative_rms']):.3f}",
        "__LARGESTMC__": f"{100*float(largest_covariance_gap['mc_relative_rms']):.3f}",
        "__LARGESTFACTOR__": f"{float(largest_covariance_gap['exact_energy_mse'])/float(largest_covariance_gap['independent_energy_mse']):.2f}"}
    for key, value in replacements.items():
        section = section.replace(key, value)
    (OUT.parent/"elliptic2d_application.tex").write_text(section, encoding="utf-8")
    with (OUT/"elliptic2d_artifact_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(dict(primary_coverage=manifest["primary_exact_rms_ci_coverage"],
            first_coverage=manifest["first_exact_rms_ci_coverage"],
            max_rms_discrepancy=manifest["primary_max_relative_rms_prediction_error"],
            smooth_energy_vs_uniform=chosen, hotspot_energy_vs_source=exact_vs_source,
            max_cg_energy_relative_error=original["maximum_cg_relative_energy_error"],
            max_cg_norm_certificate=original["maximum_cg_relative_energy_norm_certificate"]), stream, indent=2)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(5.15, 2.5), constrained_layout=True)
    for ax, profile in zip(axes, ("smooth", "hotspot")):
        for policy, marker, color, label in (("uniform", "s", "#1764a0", "Uniform placement"),
            ("source", "^", "#b65c12", "Source scale placement"),
            ("energy", "o", "#28824d", "Full energy placement")):
            group = [row for row in rows if int(row["mesh"]) == 32 and row["profile"] == profile
                     and row["layout"] == "contiguous" and row["precision_policy"] == policy]
            group.sort(key=lambda row: int(row["block_size"]))
            x = np.array([int(row["block_size"]) for row in group])
            y = np.array([100*float(row["mc_relative_rms"]) for row in group])
            lo = np.array([100*float(row["mc_relative_rms_low"]) for row in group])
            hi = np.array([100*float(row["mc_relative_rms_high"]) for row in group])
            ax.plot(x, [100*float(row["exact_relative_rms"]) for row in group],
                    color=color, lw=1.1, label=label)
            ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt=marker, color=color, ms=3, capsize=2, lw=.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 128], ["8", "32", "128"])
        ax.set_xlabel("Block length; equal 7-bit payload per cell")
        ax.set_ylabel("Relative energy RMS (%)")
        ax.set_title("Smooth scale" if profile == "smooth" else "Localized hotspot")
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(OUT/"elliptic2d_application.pdf")
    fig.savefig(OUT/"elliptic2d_application.png", dpi=220)
    plt.close(fig)
    print("Generated 2Dtable, optionalfigure, and keypairedcomparisons from allretainedprimaryvectors.")


if __name__ == "__main__":
    main()

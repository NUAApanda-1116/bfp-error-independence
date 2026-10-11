"""Fixed-budget BFP compression of independent cell sources in a Poisson solve.

Run: python poisson_application.py; rebuild without resampling: --from-saved.
The fixed codec returns a format failure outside its signed int8 exponent
range. It never saturates or silently adds an unbudgeted escape payload.
All prescribed layouts and block lengths are retained; inputs are paired
across codecs and independent of all earlier application experiments.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy.fft import dst, idst
from scipy.special import log_ndtr, logsumexp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_revision as core

OUT = Path(__file__).resolve().parent
LAWS = ("exponential", "lognormal")
SIZES = (128, 512)
BLOCK_SIZES = (4, 8, 16, 32, 64, 128)
BITS_PER_ENTRY = 7
EXPONENT_BITS = 8
EXPONENT_MIN, EXPONENT_MAX = -128, 127
DEFAULT_TRIALS = {128: 40000, 512: 20000}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def source_moments(law):
    if law == "exponential":
        return 1., 1., 2., 24.
    if law == "lognormal":
        return math.exp(.5), math.exp(2)-math.e, math.exp(2), math.exp(8)
    raise ValueError(law)


def geometry(n):
    h = 1 / (n + 1)
    i = np.arange(1, n + 1)
    # Green matrix of A=h^-2 tridiag(-1,2,-1), checked against DST below.
    green = h*h * np.minimum.outer(i, i) * (n + 1 - np.maximum.outer(i, i)) / (n + 1)
    eigenvalues = 4 / (h*h) * np.sin(np.pi * i / (2*(n+1)))**2
    return h, green, eigenvalues


def configurations(n):
    configurations = []
    budget = BITS_PER_ENTRY*n
    for size in BLOCK_SIZES:
        count = n // size
        base = (budget - EXPONENT_BITS*count)//n - 3
        extra = (budget - EXPONENT_BITS*count - n*(base+3))//size
        # Locations are part of the prescribed codec, requiring no flags.
        high = set(np.floor((np.arange(extra)+.5)*count/extra).astype(int)) if extra else set()
        levels = np.array([base + (j in high) for j in range(count)], dtype=int)
        payload = int(np.sum(size*(levels+3)) + EXPONENT_BITS*count)
        padding = budget-payload
        if base < 0 or extra > count or not 0 <= padding < size:
            raise RuntimeError("invalid prescribed storage allocation")
        for layout in ("contiguous", "interleaved"):
            if count == 1 and layout == "interleaved":
                continue
            ids = (np.arange(n).reshape(count, size) if layout == "contiguous"
                   else np.arange(n).reshape(size, count).T)
            configurations.append(dict(id=f"b{size}_{layout}", n=n, block_size=size,
                blocks=count, layout=layout, base_L=int(base), higher_L_blocks=int(extra),
                L_by_block=levels.tolist(), block_indices=ids.tolist(),
                coefficient_bits=int(size*np.sum(levels+3)),
                exponent_bits=EXPONENT_BITS*count, padding_bits=padding,
                total_bits=payload+padding, bits_per_entry=BITS_PER_ENTRY))
    return configurations


def probability_audit(law, config, h, eigenvalues):
    """Union-bound format failure and omitted energy moment, in log10 units."""
    n, size = config["n"], config["block_size"]
    lower_log = EXPONENT_MIN*math.log(2)
    upper_log = (EXPONENT_MAX+1)*math.log(2)
    if law == "lognormal":
        log_lower = size*float(log_ndtr(lower_log))
        log_upper_single = float(log_ndtr(-upper_log))
    else:
        log_lower = size*math.log(-math.expm1(-math.exp(lower_log)))
        log_upper_single = -math.exp(upper_log)
    log_failure = min(0., math.log(config["blocks"]) +
                      float(logsumexp([log_lower, math.log(size)+log_upper_single])))
    _, _, second, fourth = source_moments(law)
    input_fourth_norm = n*fourth+n*(n-1)*second*second
    # ||e||^2 <= b_max 2^-2(Lmin+1) ||b||^2; Cauchy bounds the
    # full-vector energy on failure, including unaffected blocks.
    factor = h/eigenvalues[0]*size/2**(2*(config["base_L"]+1))
    log_energy = math.log(factor)+.5*(math.log(input_fourth_norm)+log_failure)
    return dict(failure_probability_log10_bound=log_failure/math.log(10),
                omitted_energy_mse_log10_bound=log_energy/math.log(10))


def population(law, config, h, green, eigenvalues, cache):
    n, size = config["n"], config["block_size"]
    means = np.zeros(n)
    exact_variance, independent_variance, uniform = 0., 0., 0.
    minimum_retained = 1.
    for ids, level in zip(config["block_indices"], config["L_by_block"]):
        key = (law, size, level)
        if key not in cache:
            cache[key] = core.exact_model(law, size, level, compute_reduced=False)
        model = cache[key]
        mu, v, kappa = model["mean"], model["var"], model["cov"]
        ids = np.array(ids)
        block_green = green[np.ix_(ids, ids)]
        trace = np.trace(block_green)
        means[ids] = mu
        exact_variance += (v-kappa)*trace+kappa*block_green.sum()
        independent_variance += v*trace
        mean_step2 = sum(t["p"]*t["s"]**2 for t in model["layers"])/model["retained_probability"]
        uniform += mean_step2/12*trace
        minimum_retained = min(minimum_retained, model["retained_probability"])
    bias = float(means @ green @ means)
    mean, variance, _, _ = source_moments(law)
    reference = h*(variance*np.trace(green)+mean*mean*green.sum())
    exact = h*(exact_variance+bias)
    independent = h*(independent_variance+bias)
    uniform *= h
    row = {k: v for k, v in config.items() if k not in ("block_indices", "L_by_block")}
    row.update(distribution=law, exact_energy_mse=exact,
               exact_energy_variance=h*exact_variance, energy_squared_bias=h*bias,
               independent_energy_mse=independent, uniform_energy_mse=uniform,
               reference_expected_energy=reference,
               exact_relative_rms=math.sqrt(exact/reference),
               independent_relative_rms=math.sqrt(independent/reference),
               uniform_relative_rms=math.sqrt(uniform/reference),
               minimum_retained_layer_probability=minimum_retained,
               **probability_audit(law, config, h, eigenvalues))
    return row


def bounded_quantize(inputs, config):
    ids = np.array(config["block_indices"])
    z = inputs[:, ids]
    maximum = np.max(np.abs(z), axis=2)
    # frexp extracts the exact binary64 binade, including the predecessor
    # of a power of two, where rounded log2 can misclassify the maximum.
    exponent = np.frexp(np.where(maximum > 0, maximum, 1))[1].astype(int)-1
    exponent = np.where(maximum > 0, exponent, 0)
    failed = (exponent < EXPONENT_MIN) | (exponent > EXPONENT_MAX)
    if np.any(failed):
        raise OverflowError(f"bounded codec format failure in {config['id']}: "
                            f"{np.count_nonzero(failed)} block(s); no resampling")
    exponent_code = exponent.astype(np.int8)
    if not np.array_equal(exponent, exponent_code.astype(int)):
        raise RuntimeError("int8 exponent encoding differs")
    levels = np.array(config["L_by_block"])
    step = np.exp2(exponent-levels[None, :])
    coefficients = np.rint(z/step[:, :, None]).astype(np.int16)
    max_positive = np.exp2(levels+2)-1
    min_negative = -np.exp2(levels+2)
    if np.any(coefficients > max_positive[None, :, None]) or np.any(coefficients < min_negative[None, :, None]):
        raise RuntimeError("signed L+3 coefficient width insufficient")
    # Construct the declared fixed-length payload. Coefficients use their
    # prescribed two's-complement widths, rather than a free int16 payload.
    payload_bits = np.zeros((len(inputs), config["total_bits"]), dtype=np.uint8)
    cursor = 0
    for j, level in enumerate(levels):
        width = int(level+3)
        exp_unsigned = exponent_code[:, j].view(np.uint8)
        payload_bits[:, cursor:cursor+8] = (exp_unsigned[:, None] >> np.arange(7, -1, -1)) & 1
        cursor += 8
        unsigned = coefficients[:, j, :] & (2**width-1)
        coded = ((unsigned[:, :, None] >> np.arange(width-1, -1, -1)) & 1).reshape(len(inputs), -1)
        payload_bits[:, cursor:cursor+config["block_size"]*width] = coded
        cursor += config["block_size"]*width
    if cursor+config["padding_bits"] != config["total_bits"]:
        raise RuntimeError("payload accounting mismatch")
    payload = np.packbits(payload_bits, axis=1, bitorder="big")
    if payload.shape[1]*8 != config["total_bits"]:
        raise RuntimeError("fixed payload byte width mismatch")
    # Independently decode the first complete payload in every batch.
    bits = np.unpackbits(payload[:1], axis=1, bitorder="big")[0]
    cursor = 0
    for j, level in enumerate(levels):
        width = int(level+3)
        exp_value = int(bits[cursor:cursor+8] @ (2**np.arange(7, -1, -1)))
        if exp_value >= 128:
            exp_value -= 256
        cursor += 8
        count = config["block_size"]*width
        unsigned = bits[cursor:cursor+count].reshape(-1, width) @ (2**np.arange(width-1, -1, -1))
        signed = np.where(unsigned >= 2**(width-1), unsigned-2**width, unsigned)
        cursor += count
        if exp_value != exponent[0, j] or not np.array_equal(signed, coefficients[0, j]):
            raise RuntimeError("packed signed-coefficient/exponent roundtrip mismatch")
    decoded = coefficients*step[:, :, None]
    residual = np.zeros_like(inputs)
    residual[:, ids.reshape(-1)] = (decoded-z).reshape(len(inputs), -1)
    return residual, int(exponent.min()), int(exponent.max())


def bootstrap_rms(values, reference, labels, replicates):
    generator = core.rng_for("poisson", "bootstrap", *labels)
    estimates = np.empty(replicates)
    for i in range(replicates):
        estimates[i] = math.sqrt(np.mean(values[generator.integers(len(values), size=len(values))])/reference)
    return np.percentile(estimates, [2.5, 97.5])


def ranking(rows):
    rankings = []
    for n in SIZES:
        for law in LAWS:
            selected = [r for r in rows if int(r["n"]) == n and r["distribution"] == law]
            for method in ("exact", "independent", "uniform", "mc"):
                order = sorted(selected, key=lambda r: (r[method+"_relative_rms"], r["id"]))
                winner = order[0]
                rankings.append(dict(n=n, distribution=law, method=method,
                    chosen_codec=winner["id"], predicted_relative_rms=winner[method+"_relative_rms"],
                    observed_relative_rms=winner["mc_relative_rms"],
                    observed_low=winner["mc_relative_rms_low"],
                    observed_high=winner["mc_relative_rms_high"],
                    ordered_codecs=";".join(r["id"] for r in order)))
    return rankings


def layout_comparisons(rows, manifest):
    comparisons = []
    for n in SIZES:
        for law in LAWS:
            saved = np.load(OUT/f"poisson_samples_{law}_n{n}.npz")
            energies, ids = saved["energy"], saved["codec_ids"].tolist()
            for size in BLOCK_SIZES:
                cont_id, inter_id = f"b{size}_contiguous", f"b{size}_interleaved"
                if inter_id not in ids:
                    continue
                contiguous = energies[:, ids.index(cont_id)]
                interleaved = energies[:, ids.index(inter_id)]
                generator = core.rng_for("poisson", "paired_layout_bootstrap", law, n, size)
                values = []
                for _ in range(manifest["bootstrap_replicates"]):
                    index = generator.integers(len(energies), size=len(energies))
                    values.append(1-math.sqrt(interleaved[index].mean()/contiguous[index].mean()))
                low, high = np.percentile(values, [2.5, 97.5])
                cont = next(r for r in rows if int(r["n"]) == n and r["distribution"] == law and r["id"] == cont_id)
                inter = next(r for r in rows if int(r["n"]) == n and r["distribution"] == law and r["id"] == inter_id)
                comparisons.append(dict(n=n, distribution=law, block_size=size,
                    exact_fractional_rms_reduction=1-inter["exact_relative_rms"]/cont["exact_relative_rms"],
                    independent_fractional_rms_reduction=1-inter["independent_relative_rms"]/cont["independent_relative_rms"],
                    uniform_fractional_rms_reduction=1-inter["uniform_relative_rms"]/cont["uniform_relative_rms"],
                    mc_fractional_rms_reduction=1-math.sqrt(interleaved.mean()/contiguous.mean()),
                    mc_fractional_rms_reduction_low=float(low), mc_fractional_rms_reduction_high=float(high),
                    bootstrap_replicates=manifest["bootstrap_replicates"], trials=len(energies)))
    write_csv(OUT/"poisson_layout_comparisons.csv", comparisons)
    return comparisons


def make_figure(rows):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(5.125, 2.35), constrained_layout=True)
    for ax, law in zip(axes, LAWS):
        rr = [r for r in rows if int(r["n"]) == 128 and r["distribution"] == law and r["layout"] == "contiguous"]
        rr.sort(key=lambda r: r["block_size"])
        x = np.arange(len(rr))
        for method, color, ls, label in (("exact", "#1764a0", "-", "Exact covariance and bias"),
            ("independent", "#b65c12", "--", "Independent, exact bias"),
            ("uniform", "#28824d", ":", "Independent uniform cell")):
            ax.plot(x, [100*r[method+"_relative_rms"] for r in rr], color=color, ls=ls, lw=1.1, label=label)
        y = np.array([100*r["mc_relative_rms"] for r in rr])
        lo = np.array([100*r["mc_relative_rms_low"] for r in rr])
        hi = np.array([100*r["mc_relative_rms_high"] for r in rr])
        ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="o", color="#333333", ms=3, capsize=2, lw=.7, label="MC, 95% CI")
        ax.set_xticks(x, [str(int(r["block_size"])) for r in rr])
        ax.set_xlabel("Block length; equal 7-bit payload per entry")
        ax.set_ylabel("Relative energy RMS (%)")
        ax.set_title(core.NAMES[law])
        ax.set_yscale("log")
        ax.grid(alpha=.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(OUT/"poisson_application.pdf")
    fig.savefig(OUT/"poisson_application.png", dpi=200)
    plt.close(fig)


def write_section(rows, rankings, comparisons, manifest):
    table = []
    for law in LAWS:
        for size in BLOCK_SIZES:
            r = next(r for r in rows if int(r["n"]) == 128 and r["distribution"] == law and int(r["block_size"]) == size and r["layout"] == "contiguous")
            higher, count, base = int(r["higher_L_blocks"]), int(r["blocks"]), int(r["base_L"])
            level = str(base) if not higher else f"{base}/{base+1} ({higher}/{count})"
            label = "Exp." if law == "exponential" else "Logn."
            table.append(f"{label} & {size} & {level} & {int(r['padding_bits'])} & {100*r['exact_relative_rms']:.3f} & {100*r['independent_relative_rms']:.3f} & {100*r['uniform_relative_rms']:.3f} & [{100*r['mc_relative_rms_low']:.3f},{100*r['mc_relative_rms_high']:.3f}] " + r"\\")
    largest = next(r for r in rows if int(r["n"]) == 128 and r["distribution"] == "lognormal" and int(r["block_size"]) == 128)
    exp_choice = next(r for r in rankings if int(r["n"]) == 128 and r["distribution"] == "exponential" and r["method"] == "exact")
    ln_choice = next(r for r in rankings if int(r["n"]) == 128 and r["distribution"] == "lognormal" and r["method"] == "exact")
    coverage = sum(r["exact_in_mc_interval"] for r in rows)
    pbound = max(r["failure_probability_log10_bound"] for r in rows)
    momentbound = max(r["omitted_energy_mse_log10_bound"] for r in rows)
    layout_rows = [next(r for r in comparisons if int(r["n"]) == n and r["distribution"] == "lognormal" and int(r["block_size"]) == 8) for n in SIZES]
    section = r"""\subsection{A Poisson solve with an equal storage budget}
\label{sec:revision-poisson}
As a scientific-computing application, consider the steady diffusion problem
$-u''=f$ on $(0,1)$ with homogeneous Dirichlet conditions. On $N$ interior
points, put $h=(N+1)^{-1}$ and $A=h^{-2}\operatorname{tridiag}(-1,2,-1)$.
The cell sources $b_i$ are independent $\operatorname{Exp}(1)$ or
$\exp(N(0,1))$ variables; $u=A^{-1}b$. We compress $b$ and compute
$\delta u=A^{-1}(Q(b)-b)$ with the same fixed operator in binary64.
This tests source compression in a prescribed elliptic solve; it does not
model a correlated random field or arithmetic inside an iterative solver.

For a partition $\{B_j\}$, let its block moments be $\mu_j,v_j,\kappa_j$,
and set $m_i=\mu_j$ for $i\in B_j$. With $\|z\|_{A,h}^2=h z^TAz$,
the block covariance formulas give
\begin{equation}\label{eq:revision-poisson-energy}
\begin{aligned}
 h^{-1}\mathbb E\|\delta u\|_{A,h}^2
 &=\sum_j\left[(v_j-\kappa_j)
    \operatorname{tr}((A^{-1})_{B_jB_j})
 +\kappa_j\mathbf1_{B_j}^{T}A^{-1}\mathbf1_{B_j}\right]\\
 &\quad +m^TA^{-1}m.
\end{aligned}
\end{equation}
The reported relative RMS divides the square root of this expectation by
the deterministic population reference
$\{h[\operatorname{Var}(b_i)\operatorname{tr}(A^{-1})+
(\mathbb E b_i)^2\mathbf1^TA^{-1}\mathbf1]\}^{1/2}$.
In particular, the bias term includes interactions across different blocks.

Each codec has exactly $7N$ payload bits on representable inputs: an 8-bit
signed exponent per block, $L_j+3$ signed bits per grid coefficient, and
explicit padding. Block lengths are $4,8,16,32,64,128$; the largest feasible
common $L$ is supplemented by one extra coefficient bit in as many evenly
spaced, prescribed blocks as the budget allows. These locations are part of
the codec and require no data-dependent flags. Table~\ref{tab:revision-poisson}
lists the $N=128$ allocations. We test contiguous and interleaved layouts,
where the latter assigns every $N/|B_j|$th coordinate to the same block.

The actual encoder rejects exponents outside $[-128,127]$; it does not
silently saturate or add escape data. No failure occurred in the retained
samples. For each configuration the per-vector union-bound failure probability is
below $10^{__PBOUND__}$; a fourth-moment/Cauchy bound puts the omitted
unconditional energy MSE below $10^{__MBOUND__}$. The population formula
uses the unsaturated model, so this audit makes its finite-range discrepancy
explicit. This is a fully specified model codec, not a claim about MXFP4.

We retain all __CONFIGS__ configurations at $N=128,512$, using respectively
__TRIAL128__ and __TRIAL512__ independent held-out source vectors per law and
__BOOT__ vector-bootstrap replicates. Each vector is reused across codecs.
The configuration and encoding policy were recorded before sampling; the
analytical pilot used to design this experiment is disclosed in the manifest.
The measured intervals cover __COVERAGE__ of __CONFIGS__ exact RMS predictions.
Both resolutions and every layout, interval, and population ranking are in
the reproduction package.

\begin{table}[tb]
\centering
\caption{Equal-$896$-bit Poisson source compression: both source laws,
$N=128$, contiguous blocks. A mixed $L/(L+1)$ entry gives the number of
higher-precision blocks in parentheses. Relative energy RMS values are
percentages. ``Ind.'' keeps exact marginal moments and bias; ``Unif.'' uses
independent zero-mean uniform-cell errors. All six candidates are retained.}
\label{tab:revision-poisson}
\begingroup\footnotesize\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lrlrrrrr}
\toprule
Law & $|B_j|$ & $L_j$ & Pad & Exact & Ind. & Unif. & MC $95\%$ interval\\
\midrule
__TABLE__
\bottomrule
\end{tabular}\endgroup
\end{table}

For the single length-$128$ lognormal block, the bias-aware independent
prediction underestimates energy MSE by a factor __MSEFACTOR__, and the
uniform-cell prediction by __UNIFORMFACTOR__. The storage constraint also
prevents treating smaller blocks as a free improvement. At $N=128$ the exact
minimum among the prescribed codecs is __EXPCHOICE__ for exponential sources
and __LNCHOICE__ for lognormal sources. The marginal-only and exact models
choose the same block lengths here. At length $8$, interleaving reduces the
measured lognormal energy RMS by __LAYOUT128__\% at $N=128$ (paired $95\%$
interval $[__LAYOUT128LOW__,__LAYOUT128HIGH__]\%$) and __LAYOUT512__\% at $N=512$
($[__LAYOUT512LOW__,__LAYOUT512HIGH__]\%$). The marginal-only models tie these
layouts, whereas the exact covariance predicts their difference.
The comparison establishes quantitative error and layout predictions; it
does not establish a general optimizer or a block-length ranking reversal.
"""
    choice_label = lambda r: "$|B_j|="+r["chosen_codec"].split("_")[0][1:]+"$ with "+r["chosen_codec"].split("_")[1]+" blocks"
    substitutions = {"__TABLE__": "\n".join(table), "__PBOUND__": str(math.ceil(pbound)),
        "__MBOUND__": str(math.ceil(momentbound)), "__CONFIGS__": str(len(rows)),
        "__TRIAL128__": f"{manifest['trials_by_n']['128']:,}".replace(",", r"\,"),
        "__TRIAL512__": f"{manifest['trials_by_n']['512']:,}".replace(",", r"\,"),
        "__BOOT__": str(manifest["bootstrap_replicates"]), "__COVERAGE__": str(coverage),
        "__MSEFACTOR__": f"{largest['exact_energy_mse']/largest['independent_energy_mse']:.2f}",
        "__UNIFORMFACTOR__": f"{largest['exact_energy_mse']/largest['uniform_energy_mse']:.2f}",
        "__EXPCHOICE__": choice_label(exp_choice), "__LNCHOICE__": choice_label(ln_choice)}
    for n, row in zip(SIZES, layout_rows):
        substitutions[f"__LAYOUT{n}__"] = f"{100*row['mc_fractional_rms_reduction']:.2f}"
        substitutions[f"__LAYOUT{n}LOW__"] = f"{100*row['mc_fractional_rms_reduction_low']:.2f}"
        substitutions[f"__LAYOUT{n}HIGH__"] = f"{100*row['mc_fractional_rms_reduction_high']:.2f}"
    for marker, replacement in substitutions.items():
        section = section.replace(marker, replacement)
    (OUT.parent/"poisson_application.tex").write_text(section, encoding="utf-8")


def rebuild(rows, manifest):
    rankings = ranking(rows)
    write_csv(OUT/"poisson_rankings.csv", rankings)
    comparisons = layout_comparisons(rows, manifest)
    make_figure(rows)
    write_section(rows, rankings, comparisons, manifest)
    files = ["poisson_application.py", "poisson_results.csv", "poisson_rankings.csv",
             "poisson_layout_comparisons.csv", "poisson_application.pdf",
             "poisson_application.png", "poisson_prespecification.json", "poisson_numerical_source.py"]
    files += [p.name for p in sorted(OUT.glob("poisson_samples_*.npz"))]
    if (OUT/"poisson_binade_audit.json").exists():
        files.append("poisson_binade_audit.json")
    manifest["files"] = {name: sha(OUT/name) for name in files}
    manifest["generated_manuscript_section"] = {"../poisson_application.tex": sha(OUT.parent/"poisson_application.tex")}
    (OUT/"poisson_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--trials-128", type=int, default=DEFAULT_TRIALS[128])
    parser.add_argument("--trials-512", type=int, default=DEFAULT_TRIALS[512])
    parser.add_argument("--from-saved", action="store_true")
    args = parser.parse_args()
    if args.from_saved:
        manifest = json.loads((OUT/"poisson_manifest.json").read_text(encoding="utf-8"))
        manifest["numerical_source_snapshot"] = "poisson_numerical_source.py"
        if sha(OUT/"poisson_numerical_source.py") != manifest["numerical_source_sha256"]:
            raise RuntimeError("executed numerical source snapshot hash mismatch")
        manifest["artifact_rebuild"] = "From saved CSVs and paired energy arrays; no new source vectors or original RMS bootstrap. Paired layout intervals computed from retained vectors; numerical source dependency hashes preserved."
        rebuild(read_rows("poisson_results.csv"), manifest)
        print("Rebuilt Poisson figures, section, rankings, and hashes from saved CSVs.")
        return
    started = time.time()
    (OUT/"poisson_numerical_source.py").write_bytes(Path(__file__).read_bytes())
    trials = {128: args.trials_128, 512: args.trials_512}
    presets = {n: configurations(n) for n in SIZES}
    manifest = dict(master_seed=core.MASTER_SEED, rng="PCG64 via SHA256-labeled SeedSequence",
        date="2026-10-10 Asia/Shanghai", laws={"exponential": "Exp(1)", "lognormal": "exp(N(0,1))"},
        trials_by_n={str(n): trials[n] for n in SIZES}, bootstrap_replicates=args.bootstrap,
        bootstrap_unit="independent complete source vectors, paired across prescribed codecs",
        confidence_level=.95, bits_per_entry=BITS_PER_ENTRY,
        encoder=dict(rounding="RNE ties-to-even", coefficient_width="L+3 signed bits",
            exponent_width=EXPONENT_BITS, exponent_range=[EXPONENT_MIN, EXPONENT_MAX],
            zero_block="exponent zero and all coefficients zero", outside_range="explicit failure; no resampling, no silent escape",
            padding="explicit unused bits to common 7N payload", precision_metadata="fixed codec locations; no data-dependent flags"),
        scope="independent cell sources; fixed unquantized 1D Dirichlet Poisson operator; binary64 residual propagation; no solver arithmetic",
        reference="population expected solution energy; not sample-wise relative errors",
        configurations={str(n): presets[n] for n in SIZES},
        selection_record=[
            "A read-only analytical pilot inspected all six block lengths at nominal budgets 4,5,6,7,8 bits/entry for N=128 before the held-out source samples.",
            "Seven bits/entry and both N=128,512 resolutions were then fixed. All six lengths and all distinct contiguous/interleaved layouts are retained, including padding and deterministic mixed precisions.",
            "No population block-length ranking reversal was found in the pilot; the experiment targets error magnitude and layout effects, and reports rankings honestly.",
            "Held-out input stream label is poisson|heldout|input|law|N. No sample-dependent threshold, seed search, or configuration deletion."],
        source_dependencies={"run_revision.py": sha(OUT/"run_revision.py")},
        numerical_source_sha256=sha(Path(__file__)), numerical_source_snapshot="poisson_numerical_source.py",
        python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
        platform=platform.platform())
    # This durable record precedes all population computation and new sampling.
    (OUT/"poisson_prespecification.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    cache, rows = {}, []
    maximum_identity_error = 0.
    for n in SIZES:
        h, green, eigenvalues = geometry(n)
        cfgs = presets[n]
        for law in LAWS:
            predictions = [population(law, cfg, h, green, eigenvalues, cache) for cfg in cfgs]
            generator = core.rng_for("poisson", "heldout", "input", law, n)
            energies = np.empty((trials[n], len(cfgs)))
            minima = np.full(len(cfgs), 100000, dtype=int)
            maxima = np.full(len(cfgs), -100000, dtype=int)
            batch = max(1, core.CHUNK_ELEMS//n)
            for start in range(0, trials[n], batch):
                inputs = core.draw(law, (min(batch, trials[n]-start), n), generator)
                for j, cfg in enumerate(cfgs):
                    residual, emin, emax = bounded_quantize(inputs, cfg)
                    transformed = dst(residual, type=1, norm="ortho", axis=1)
                    energy = h*np.sum(transformed*transformed/eigenvalues, axis=1)
                    if start == 0:
                        check = h*np.einsum("bi,ij,bj->b", residual[:8], green, residual[:8])
                        discrepancy = float(np.max(np.abs(check-energy[:8])/(1+np.abs(check))))
                        maximum_identity_error = max(maximum_identity_error, discrepancy)
                        solved = idst(transformed[:8]/eigenvalues, type=1, norm="ortho", axis=1)
                        direct = h*np.sum(residual[:8]*solved, axis=1)
                        if not np.allclose(direct, check, atol=2e-16, rtol=2e-13):
                            raise RuntimeError("DST solve/Green energy identity mismatch")
                    energies[start:start+len(inputs), j] = energy
                    minima[j] = min(minima[j], emin)
                    maxima[j] = max(maxima[j], emax)
            np.savez_compressed(OUT/f"poisson_samples_{law}_n{n}.npz", energy=energies,
                codec_ids=np.array([cfg["id"] for cfg in cfgs]), exponent_min=minima, exponent_max=maxima)
            for j, row in enumerate(predictions):
                rms = math.sqrt(float(energies[:, j].mean())/row["reference_expected_energy"])
                low, high = bootstrap_rms(energies[:, j], row["reference_expected_energy"], (law, n, row["id"]), args.bootstrap)
                row.update(trials=trials[n], mc_energy_mse=float(energies[:, j].mean()),
                    mc_relative_rms=rms, mc_relative_rms_low=float(low), mc_relative_rms_high=float(high),
                    exact_in_mc_interval=bool(low <= row["exact_relative_rms"] <= high),
                    relative_prediction_error=(rms/row["exact_relative_rms"]-1),
                    observed_exponent_min=int(minima[j]), observed_exponent_max=int(maxima[j]), format_failures=0)
                rows.append(row)
            print(f"Completed {law}, N={n}: {trials[n]} paired vectors, {len(cfgs)} codecs.", flush=True)
    write_csv(OUT/"poisson_results.csv", rows)
    manifest.update(duration_seconds=time.time()-started,
        maximum_DST_Green_energy_identity_error=maximum_identity_error,
        exact_rms_ci_coverage=sum(r["exact_in_mc_interval"] for r in rows), configurations_count=len(rows),
        maximum_relative_rms_prediction_error=max(abs(r["relative_prediction_error"]) for r in rows),
        format_failures=0,
        failure_probability_log10_bound=max(r["failure_probability_log10_bound"] for r in rows),
        omitted_energy_mse_log10_bound=max(r["omitted_energy_mse_log10_bound"] for r in rows))
    rebuild(rows, manifest)
    print(f"Retained {len(rows)} configurations; exact RMS CI coverage {manifest['exact_rms_ci_coverage']}/{len(rows)}; "
          f"largest RMS discrepancy {100*manifest['maximum_relative_rms_prediction_error']:.2f}%; no exponent failures.")


if __name__ == "__main__":
    main()

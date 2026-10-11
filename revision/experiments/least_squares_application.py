"""Fixed-design least-squares RHS compression: exact BFP error propagation.

Only iid entries of b are quantized.  A is deterministic and unquantized;
coefficient errors are evaluated as A^dagger (Q(b)-b) in binary64.  This is
not a simulation of solver arithmetic, non-iid regression residuals, or MXFP4.
Run: python least_squares_application.py
All configurations and all sampled coefficients are retained.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_revision as core

OUT = Path(__file__).resolve().parent
N = 128
LM = 2
LAWS = ("exponential", "lognormal")
COORDINATES = ("intercept", "centered_slope", "through_origin_slope")
LABELS = ("Intercept", "Slope with intercept", "Slope through origin")
PRECISIONS = (2, 3, 4, 5)
RMS_TOLERANCE = .10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_saved_csv(name):
    with (OUT/name).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            elif value == "":
                row[key] = None
            elif key in ("n", "L", "blocks", "selected_L"):
                row[key] = int(value)
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def designs():
    t = np.linspace(0., 1., N)
    z = t - .5
    a = np.column_stack((np.ones(N), z))
    q, r = np.linalg.qr(a, mode="reduced")
    inverse = np.linalg.solve(r, q.T)
    ao = t[:, None]
    qo, ro = np.linalg.qr(ao, mode="reduced")
    origin = np.linalg.solve(ro, qo.T)
    weights = np.vstack((inverse, origin))
    # Independent identities verify the fixed QR operator.
    analytic = np.vstack((np.ones(N)/N, z/(z@z), t/(t@t)))
    if not np.allclose(weights, analytic, atol=2e-16, rtol=3e-14):
        raise RuntimeError("QR operator differs from analytic design")
    g = weights @ np.ones(N)
    if not np.allclose(g[:2], [1., 0.], atol=2e-15):
        raise RuntimeError("constant-direction geometry check failed")
    return t, weights, g


def statistics(samples):
    return {"mean": samples.mean(0), "variance": samples.var(0, ddof=1),
            "mse": np.mean(samples*samples, axis=0),
            "covariance": np.cov(samples, rowvar=False, ddof=1)}


def bootstrap(samples, law, replicates):
    generator = core.rng_for("least_squares", "bootstrap", law)
    values = {k: [] for k in statistics(samples)}
    for _ in range(replicates):
        x = samples[generator.integers(len(samples), size=len(samples))]
        for key, value in statistics(x).items():
            values[key].append(value)
    return {k: np.percentile(v, [2.5, 97.5], axis=0) for k, v in values.items()}


def figure(rows):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 9,
                         "legend.fontsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(5.125, 2.25), constrained_layout=True)
    for ax, law in zip(axes, LAWS):
        rr = [r for r in rows if r["distribution"] == law]
        x = np.arange(len(rr))
        pred = np.array([r["exact_variance_ratio"] for r in rr])
        obs = np.array([r["mc_variance_ratio"] for r in rr])
        lo = np.array([r["mc_variance_ratio_low"] for r in rr])
        hi = np.array([r["mc_variance_ratio_high"] for r in rr])
        ax.axhline(1., color="#666666", ls="--", lw=1., label="Independent errors")
        ax.plot(x, pred, "s", ms=4, color="#1764a0", label="Exact covariance")
        ax.errorbar(x+.06, obs, yerr=[obs-lo, hi-obs], fmt="o", ms=3,
                    lw=.8, capsize=2, color="#b65c12", label="MC, 95% CI")
        ax.set_xticks(x, ["Intercept", "Slope with\nintercept", "Slope through\norigin"])
        ax.set_title(core.NAMES[law])
        ax.set_ylabel("Coefficient variance / independent prediction")
        ax.grid(axis="y", alpha=.15)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(OUT/"least_squares_application.pdf")
    fig.savefig(OUT/"least_squares_application.png", dpi=200)
    plt.close(fig)


def precision_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(5.125, 2.35), constrained_layout=True)
    for ax, law in zip(axes, LAWS):
        rr = [r for r in rows if r["distribution"] == law]
        precisions = np.array([r["L"] for r in rr])
        for method, color, linestyle, label in (
            ("exact", "#1764a0", "-", "Exact covariance and bias"),
            ("independent", "#b65c12", "--", "Independent, exact bias"),
            ("uniform", "#28824d", ":", "Independent uniform cell")):
            ax.plot(precisions, [r[method+"_rms"] for r in rr], color=color,
                    ls=linestyle, lw=1.1, label=label)
        observed = np.array([r["mc_rms"] for r in rr])
        low = np.array([r["mc_rms_low"] for r in rr])
        high = np.array([r["mc_rms_high"] for r in rr])
        ax.errorbar(precisions, observed, yerr=[observed-low, high-observed],
                    fmt="o", ms=3, color="#333333", lw=.7, capsize=2,
                    label="MC, 95% CI")
        ax.axhline(RMS_TOLERANCE, color="#777777", ls="-.", lw=.9,
                   label="Illustrative tolerance")
        ax.set_yscale("log")
        ax.set_xticks(PRECISIONS)
        ax.set_xlabel("Grid precision $L$")
        ax.set_ylabel("Intercept compression RMS")
        ax.set_title(core.NAMES[law])
        ax.grid(alpha=.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2,
               frameon=False, fontsize=7)
    fig.savefig(OUT/"least_squares_precision.pdf")
    fig.savefig(OUT/"least_squares_precision.png", dpi=200)
    plt.close(fig)


def write_section(rows, precision_rows, decisions, replicates):
    table = []
    for law in LAWS:
        for row in (r for r in precision_rows if r["distribution"] == law):
            table.append("{} & {} & {:.4f} & {:.4f} & {:.4f} & [{:.4f}, {:.4f}] \\\\".format(
                "Exp(1)" if law == "exponential" else "Lognormal", row["L"],
                row["exact_rms"], row["independent_rms"], row["uniform_rms"],
                row["mc_rms_low"], row["mc_rms_high"]))
    section = r"""\subsection{Fixed-design least squares with a quantized right-hand side}
\label{sec:revision-ls}
The covariance formula also predicts errors in coefficients of a prescribed
linear approximation. Let $A\in\mathbb R^{n\times p}$ be deterministic and
have full column rank, and let $b_i$ be i.i.d. observations. Define
$\theta=A^\dagger b$ and $\widetilde\theta=A^\dagger Q(b)$, where the
right-hand side is quantized as one block and
$A^\dagger=(A^TA)^{-1}A^T$. These are projections of the same observations;
we measure the additional coefficient error due to data compression.

\begin{corollary}[Coefficient error and the constant direction]
\label{cor:revision-ls}
Put $g=A^\dagger\mathbf1$ and $\delta\theta=\widetilde\theta-\theta$. Then
\begin{align}
 \mathbb E[\delta\theta]&=\mu g,\qquad
 \operatorname{Cov}(\delta\theta)
 =(v-\kappa)(A^TA)^{-1}+\kappa gg^T,\label{eq:revision-ls-cov}\\
 \mathbb E[\|\delta\theta\|_2^2]
 &=(v-\kappa)\operatorname{tr}((A^TA)^{-1})
       +(\kappa+\mu^2)\|g\|_2^2.\label{eq:revision-ls-mse}
\end{align}
If $A=[\mathbf1,Z]$ has full column rank, then $g=(1,0,\ldots,0)^T$.
If instead $A^T\mathbf1=0$, then $g=0$.
\end{corollary}
\begin{proof}
Since $\delta\theta=A^\dagger e$, multiply
\eqref{eq:revision-spectrum} on both sides by $A^\dagger$ and its transpose,
and use $A^\dagger(A^\dagger)^T=(A^TA)^{-1}$. The second identity is the
trace of the covariance plus the squared mean. For the stated designs,
$A(1,0,\ldots,0)^T=\mathbf1$ or $A^T\mathbf1=0$, respectively.
\end{proof}

Thus the inverse Gram matrix controls ordinary geometric amplification,
while $g$ determines how the shared component and bias enter the fitted
coefficients. An intercept confines this component to the intercept
coefficient; omitting the intercept can transmit it to a slope. The centered
slope still has variance proportional to $v-\kappa$, rather than $v$.

We use $n=128$ and $t_i=(i-1)/(n-1)$. The first design fits
$\alpha+\beta(t_i-1/2)$, and the second fits $\gamma t_i$ through the
origin. Inputs are either $\operatorname{Exp}(1)$ or
$\exp(N(0,1))$; there is no deterministic trend added to these i.i.d.
right-hand sides. Each law uses $__BLOCKS__$ independent blocks and $__BOOT__$
block bootstrap replicates. At $L=__L__$, the intercept biases are $__EXPBIAS__$
and $__LNBIAS__$, respectively. For lognormal inputs, retaining the exact
marginal $\mu,v$ but setting $\kappa=0$ underpredicts the intercept and
through-origin slope variances by factors $__LNINTERCEPT__$ and $__LNORIGIN__$,
while overpredicting the centered slope variance by $__LNSLOPEPERCENT__\%$.
The mean, variance, and mean-square-error intervals cover $__MEANCOV__$,
$__VARCOV__$, and $__MSECOV__$ of $__COUNT__$ predictions, respectively.
__MISSES__ Full coefficient results are in the package.

For an accuracy decision, $\alpha=\overline b$ has predicted mean-square
compression error
$[v+(n-1)\kappa]/n+\mu^2$. Table~\ref{tab:revision-ls} uses the same
input blocks at $L=__PRECISIONS__$. Besides the bias-aware independent baseline
$v/n+\mu^2$, it reports the classical independent, zero-mean uniform-cell
prediction $\mathbb E[s^2]/(12n)$. We choose an illustrative RMS tolerance
$__TOLERANCE__$ after inspecting the initial experiment; this is a transparent
design example, not a preregistered threshold or a precision optimizer.

\begin{table}[tb]
\centering
\caption{RMS compression error in the fitted intercept $\alpha$.
``Ind.'' retains the exact bias and marginal variance but drops pair
covariance; ``Unif.'' assumes independent, zero-mean uniform-cell errors.
The last column gives $95\%$ block bootstrap intervals for the measured
RMS. All tested precisions are retained.}
\label{tab:revision-ls}
\begingroup\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrr}
\toprule
Law & $L$ & Exact & Ind. & Unif. & MC RMS interval\\
\midrule
__TABLE__
\bottomrule
\end{tabular}\endgroup
\end{table}

For lognormal inputs, among the tested precisions, the exact prediction
selects $L=__LNEXACTL__$, whereas the bias-aware independent model selects
$L=__LNINDL__$ and the uniform-cell model selects $L=__LNUNIFL__$.
__DECISION_VALIDATION__ For exponential inputs all three models select
$L=__EXPL__$; the measured RMS at this precision is $__EXPMC__$.
At fixed block length, increasing
$L$ by one increases a sufficient signed integer coefficient width by
one bit while preserving one shared exponent; this is a model-level
storage tradeoff, not a hardware format comparison. Only the right-hand
side is quantized. The designs are fixed and their QR pseudoinverses are
applied to the residual in binary64, isolating input compression from
subsequent solver arithmetic and from non-i.i.d. regression models.
"""
    section = section.replace("__TABLE__", "\n".join(table))
    def coordinate(law, name):
        return next(r for r in rows if r["distribution"] == law and r["coordinate"] == name)
    def decision(law, method):
        return next(d for d in decisions if d["distribution"] == law and d["method"] == method)
    missed = [r for r in rows if not r["exact_variance_in_ci"]]
    coef_labels = {"intercept": "intercept", "centered_slope": "centered slope",
                   "through_origin_slope": "through-origin slope"}
    if missed:
        miss_details = []
        for r in missed:
            miss_details.append("{} {} (exact ratio ${:.6f}$, interval $[{:.6f},{:.6f}]$)".format(
                core.NAMES[r["distribution"]], coef_labels[r["coordinate"]],
                r["exact_variance_ratio"], r["mc_variance_ratio_low"], r["mc_variance_ratio_high"]))
        miss_text = "The retained variance-interval misses are: "+"; ".join(miss_details)+"."
    else:
        miss_text = "Every coefficient variance interval covers its population prediction."
    ld = [decision("lognormal", method) for method in ("exact", "independent", "uniform")]
    validation = []
    for d, label in zip(ld, ("exact", "bias-aware independent", "uniform-cell")):
        outcome = "meets" if d["mc_meets_tolerance_at_selected"] else "fails"
        validation.append("the {} selection {} the tolerance (measured RMS ${:.5f}$)".format(
            label, outcome, d["mc_rms_at_selected"]))
    replacements = {
        "__N__": str(N), "__L__": str(LM),
        "__BLOCKS__": f"{rows[0]['blocks']:,}".replace(",", r"\,"), "__BOOT__": str(replicates),
        "__EXPBIAS__": f"{coordinate('exponential','intercept')['exact_mean']:.5f}",
        "__LNBIAS__": f"{coordinate('lognormal','intercept')['exact_mean']:.5f}",
        "__LNINTERCEPT__": f"{coordinate('lognormal','intercept')['exact_variance_ratio']:.2f}",
        "__LNORIGIN__": f"{coordinate('lognormal','through_origin_slope')['exact_variance_ratio']:.2f}",
        "__LNSLOPEPERCENT__": f"{100*(1/coordinate('lognormal','centered_slope')['exact_variance_ratio']-1):.1f}",
        "__MEANCOV__": str(sum(r["exact_mean_in_ci"] for r in rows)),
        "__VARCOV__": str(sum(r["exact_variance_in_ci"] for r in rows)),
        "__MSECOV__": str(sum(r["exact_mse_in_ci"] for r in rows)), "__COUNT__": str(len(rows)),
        "__MISSES__": miss_text, "__PRECISIONS__": ",".join(map(str, PRECISIONS)),
        "__TOLERANCE__": f"{RMS_TOLERANCE:.2f}",
        "__LNEXACTL__": str(ld[0]["selected_L"]), "__LNINDL__": str(ld[1]["selected_L"]),
        "__LNUNIFL__": str(ld[2]["selected_L"]),
        "__DECISION_VALIDATION__": "Monte Carlo shows that "+"; ".join(validation)+".",
        "__EXPL__": str(decision("exponential", "exact")["selected_L"]),
        "__EXPMC__": f"{decision('exponential', 'exact')['mc_rms_at_selected']:.5f}"}
    section = section.replace("$n=128$", "$n=__N__$")
    for key, value in replacements.items():
        section = section.replace(key, value)
    (OUT.parent/"application.tex").write_text(section, encoding="utf-8")


def update_artifacts_from_saved():
    """Rebuild figures/text/hashes from saved result CSVs; never resample."""
    rows = read_saved_csv("least_squares_results.csv")
    precision_rows = read_saved_csv("least_squares_precision.csv")
    decisions = read_saved_csv("least_squares_precision_decisions.csv")
    manifest = json.loads((OUT/"least_squares_manifest.json").read_text(encoding="utf-8"))
    figure(rows)
    precision_figure(precision_rows)
    write_section(rows, precision_rows, decisions, manifest["bootstrap_replicates"])
    manifest["selection_record"][1] = (
        "The initial L=2 coefficient run reused n=128 and L=2 from the existing weighted experiment, with no parameter-grid or random-seed search. A subsequent L=2,3,4,5 accuracy sweep reused exactly the same RHS blocks and retained every precision.")
    manifest["files"] = {name: sha(OUT/name) for name in manifest["files"]}
    manifest["generated_manuscript_section"] = {"../application.tex": sha(OUT.parent/"application.tex")}
    manifest["artifact_rebuild"] = "Generated from saved result CSVs without sampling or bootstrap computation; source dependency hashes for the numerical run are preserved."
    (OUT/"least_squares_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Rebuilt application text, figures, and hashes from saved CSVs; no sampling or bootstrap computation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=80000)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--from-saved", action="store_true",
                        help="rebuild manuscript section, figures, and hashes from saved CSVs; no simulation or bootstrap")
    args = parser.parse_args()
    if args.from_saved:
        update_artifacts_from_saved()
        return
    started = time.time()
    t, weights, g = designs()
    gram_inverse = weights @ weights.T
    rows, covrows, precision_rows, decisions = [], [], [], []
    output_files = []
    check_max = 0.
    for law in LAWS:
        model = core.exact_model(law, N, LM)
        mu, v, kappa = model["mean"], model["var"], model["cov"]
        exact_mean = mu*g
        exact_cov = (v-kappa)*gram_inverse + kappa*np.outer(g, g)
        independent_cov = v*gram_inverse
        generator = core.rng_for("least_squares", "input", law, N, LM)
        chunks, exponents, precision_chunks = [], [], []
        batch = max(1, core.CHUNK_ELEMS//N)
        for start in range(0, args.blocks, batch):
            b = core.draw(law, (min(batch, args.blocks-start), N), generator)
            error, step, exponent = core.quantize(b, LM, "rne", generator)
            delta = error @ weights.T
            if start == 0:
                direct = (b[:32]+error[:32]) @ weights.T - b[:32] @ weights.T
                deviation = float(np.max(np.abs(direct-delta[:32])))
                check_max = max(check_max, deviation)
                if deviation > 1e-13:
                    raise RuntimeError("direct fit difference disagrees with residual propagation")
            chunks.append(delta)
            exponents.append(exponent[:, 0])
            sweep = [delta[:, 0]]
            for precision in PRECISIONS[1:]:
                ep, _, _ = core.quantize(b, precision, "rne", generator)
                sweep.append(ep @ weights[0])
            precision_chunks.append(np.column_stack(sweep))
        samples = np.concatenate(chunks)
        exponents = np.concatenate(exponents)
        point = statistics(samples)
        interval = bootstrap(samples, law, args.bootstrap)
        raw_name = f"least_squares_samples_{law}.csv"
        np.savetxt(OUT/raw_name, np.column_stack((exponents, samples)),
                   delimiter=",", header="K,intercept,centered_slope,through_origin_slope",
                   comments="", fmt=["%d", "%.17g", "%.17g", "%.17g"])
        output_files.append(raw_name)
        precision_samples = np.concatenate(precision_chunks)
        precision_point = statistics(precision_samples)
        precision_interval = bootstrap(precision_samples, "precision_"+law, args.bootstrap)
        raw_precision_name = f"least_squares_precision_samples_{law}.csv"
        np.savetxt(OUT/raw_precision_name, precision_samples, delimiter=",",
                   header="intercept_L2,intercept_L3,intercept_L4,intercept_L5",
                   comments="", fmt="%.17g")
        output_files.append(raw_precision_name)
        for j, precision in enumerate(PRECISIONS):
            pm = core.exact_model(law, N, precision)
            exact_mse = pm["mse_sum"]/N**2
            independent_mse = pm["var"]/N+pm["mean"]**2
            uniform_mse = pm["var"]/(N*pm["variance_ratio"])
            prow = dict(distribution=law, n=N, L=precision, blocks=args.blocks,
                        error_mean=pm["mean"], error_variance=pm["var"],
                        error_pair_covariance=pm["cov"],
                        expected_step_squared=12*pm["var"]/pm["variance_ratio"],
                        exact_mse=exact_mse, independent_mse=independent_mse,
                        uniform_mse=uniform_mse, exact_rms=float(np.sqrt(exact_mse)),
                        independent_rms=float(np.sqrt(independent_mse)),
                        uniform_rms=float(np.sqrt(uniform_mse)),
                        mc_mse=float(precision_point["mse"][j]),
                        mc_mse_low=float(precision_interval["mse"][0, j]),
                        mc_mse_high=float(precision_interval["mse"][1, j]),
                        mc_rms=float(np.sqrt(precision_point["mse"][j])),
                        mc_rms_low=float(np.sqrt(precision_interval["mse"][0, j])),
                        mc_rms_high=float(np.sqrt(precision_interval["mse"][1, j])),
                        rms_tolerance=RMS_TOLERANCE)
            for method in ("exact", "independent", "uniform", "mc"):
                prow[method+"_passes"] = prow[method+"_rms"] <= RMS_TOLERANCE
            precision_rows.append(prow)
        for method in ("exact", "independent", "uniform", "mc"):
            candidates = [r for r in precision_rows if r["distribution"] == law and r[method+"_passes"]]
            selected = min(r["L"] for r in candidates) if candidates else None
            at_selected = next((r for r in candidates if r["L"] == selected), None)
            decisions.append(dict(distribution=law, method=method, tolerance=RMS_TOLERANCE,
                                  selected_L=selected,
                                  mc_rms_at_selected=None if at_selected is None else at_selected["mc_rms"],
                                  mc_rms_high_at_selected=None if at_selected is None else at_selected["mc_rms_high"],
                                  mc_meets_tolerance_at_selected=None if at_selected is None else at_selected["mc_passes"]))
        for j, coordinate in enumerate(COORDINATES):
            baseline = float(independent_cov[j, j])
            row = dict(distribution=law, n=N, L=LM, coordinate=coordinate,
                       blocks=args.blocks, weight_sum=float(g[j]),
                       weight_norm_squared=float(gram_inverse[j, j]),
                       error_mean=mu, error_variance=v, error_pair_covariance=kappa,
                       exact_mean=float(exact_mean[j]),
                       exact_variance=float(exact_cov[j, j]),
                       independent_variance=baseline,
                       exact_mse=float(exact_cov[j, j]+exact_mean[j]**2),
                       independent_mse=float(baseline+exact_mean[j]**2),
                       exact_variance_ratio=float(exact_cov[j, j]/baseline))
            for key in ("mean", "variance", "mse"):
                row["mc_"+key] = float(point[key][j])
                row["mc_"+key+"_low"] = float(interval[key][0, j])
                row["mc_"+key+"_high"] = float(interval[key][1, j])
            for suffix in ("", "_low", "_high"):
                row["mc_variance_ratio"+suffix] = row["mc_variance"+suffix]/baseline
            row["exact_mean_in_ci"] = row["mc_mean_low"] <= row["exact_mean"] <= row["mc_mean_high"]
            row["exact_variance_in_ci"] = row["mc_variance_low"] <= row["exact_variance"] <= row["mc_variance_high"]
            row["exact_mse_in_ci"] = row["mc_mse_low"] <= row["exact_mse"] <= row["mc_mse_high"]
            rows.append(row)
        for j in range(3):
            for k in range(j+1, 3):
                covrows.append(dict(distribution=law, coordinate_1=COORDINATES[j],
                                    coordinate_2=COORDINATES[k],
                                    exact_covariance=float(exact_cov[j, k]),
                                    independent_covariance=float(independent_cov[j, k]),
                                    mc_covariance=float(point["covariance"][j, k]),
                                    mc_covariance_low=float(interval["covariance"][0, j, k]),
                                    mc_covariance_high=float(interval["covariance"][1, j, k])))
    write_csv(OUT/"least_squares_results.csv", rows)
    write_csv(OUT/"least_squares_covariance.csv", covrows)
    write_csv(OUT/"least_squares_precision.csv", precision_rows)
    write_csv(OUT/"least_squares_precision_decisions.csv", decisions)
    figure(rows)
    precision_figure(precision_rows)
    write_section(rows, precision_rows, decisions, args.bootstrap)
    output_files += ["least_squares_results.csv", "least_squares_covariance.csv",
                     "least_squares_application.pdf", "least_squares_application.png",
                     "least_squares_precision.csv", "least_squares_precision_decisions.csv",
                     "least_squares_precision.pdf", "least_squares_precision.png"]
    manifest = dict(master_seed=core.MASTER_SEED,
                    rng="NumPy PCG64; SeedSequence with SHA256 labels, inherited rng_for",
                    date="2026-10-10 Asia/Shanghai", n=N, L=LM,
                    laws={"exponential": "Exp(1)", "lognormal": "exp(N(0,1))"},
                    blocks_per_law=args.blocks, bootstrap_replicates=args.bootstrap,
                    confidence_level=.95, bootstrap_unit="independent complete RHS blocks",
                    design="t_i=i/(n-1), i=0,...,n-1; A=[1,t-1/2] and A_origin=t",
                    quantizer="RNE ties-to-even; step=2^(floor(log2 max|b_i|)-L); uniform unsaturated grid",
                    scope="iid RHS compression; deterministic unquantized design; binary64 application of a fixed QR pseudoinverse; solver arithmetic excluded; not MXFP4",
                    baseline="same exact marginal mean and variance, kappa=0",
                    selection_record=[
                        "Two laws chosen from the existing grid before generating application samples: exponential (memoryless control C=0) and lognormal (C nonzero).",
                        "The initial L=2 coefficient run reused n=128 and L=2 from the existing weighted experiment, with no parameter-grid or random-seed search. A subsequent L=2,3,4,5 accuracy sweep reused exactly the same RHS blocks and retained every precision.",
                        "Inspected only analytic population moments to verify the prescribed designs' constant-direction geometry and effect sizes.",
                        "All two laws and all three coefficient coordinates are retained, including every missed confidence interval; seed and labeled streams are fixed."],
                    precision_sweep=dict(precisions=list(PRECISIONS),
                                         tolerance=RMS_TOLERANCE,
                                         observation_stream="exactly the same RHS blocks as the original L=2 application run; input label retains L=2",
                                         purpose="illustrative accuracy decision, not an optimization algorithm",
                                         threshold_provenance="chosen after the initial coefficient experiment and analytic inspection, not preregistered; every tested L is retained",
                                         decisions=decisions),
                    design_weight_sums=g.tolist(),
                    direct_fit_difference_check_max=check_max,
                    exact_mean_ci_coverage=sum(r["exact_mean_in_ci"] for r in rows),
                    exact_variance_ci_coverage=sum(r["exact_variance_in_ci"] for r in rows),
                    exact_mse_ci_coverage=sum(r["exact_mse_in_ci"] for r in rows),
                    configurations=len(rows), duration_seconds=time.time()-started,
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    matplotlib=matplotlib.__version__, platform=platform.platform(),
                    source_dependencies={"run_revision.py": sha(OUT/"run_revision.py")},
                    files={name: sha(OUT/name) for name in output_files+[Path(__file__).name]})
    manifest["generated_manuscript_section"] = {"../application.tex": sha(OUT.parent/"application.tex")}
    (OUT/"least_squares_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    for row in rows:
        print(row["distribution"], row["coordinate"],
              "bias", row["exact_mean"], "MC", row["mc_mean"],
              "variance ratio", row["exact_variance_ratio"],
              "CI", [row["mc_variance_ratio_low"], row["mc_variance_ratio_high"]])
    print("coverage mean/variance/mse:", manifest["exact_mean_ci_coverage"],
          manifest["exact_variance_ci_coverage"], manifest["exact_mse_ci_coverage"],
          "of", len(rows), "; seconds:", manifest["duration_seconds"])


if __name__ == "__main__":
    main()

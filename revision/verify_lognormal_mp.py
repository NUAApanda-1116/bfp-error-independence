"""Independent high-precision audit of lognormal finite-block moments.

Run: python revision/verify_lognormal_mp.py

Cell moments are integrated with analytical lognormal truncated raw moments
in mpmath at 60 and 90 decimal digits. Unconditional first, second and pair
moments are computed by subtraction over the two nested maximum events,
without the normalized negative-covariance formula used by the producer.
The producer is imported only for the final numerical comparison.
"""
from pathlib import Path
import hashlib
import json
import sys
import mpmath as mp


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
import run_revision as core

CONFIGURATIONS = [(8, 3), (32, 3), (32, 4), (128, 3)]
DIGITS = (60, 90)
STATE_THRESHOLD = "1e-40"


def survival_raw(power, threshold):
    """E[X**power; X>=threshold], for log(X)~N(0,1)."""
    if threshold == 0:
        return mp.exp(mp.mpf(power)**2 / 2)
    z = (mp.log(threshold) - power) / mp.sqrt(2)
    return mp.exp(mp.mpf(power)**2 / 2) * mp.erfc(z) / 2


def cdf(threshold):
    return mp.erfc(-mp.log(threshold) / mp.sqrt(2)) / 2


def integrate_cells(step, upper):
    """Integrate the residual and its square from zero to upper exactly in law."""
    first = second = mp.mpf(0)
    # The endpoint/step ratio is a power of two, so this cell count is exact.
    cells = int(upper / step)
    for integer in range(cells + 1):
        center = integer * step
        lower = max(mp.mpf(0), center - step / 2)
        higher = min(upper, center + step / 2)
        if higher <= lower:
            continue
        raw = [survival_raw(power, lower) - survival_raw(power, higher)
               for power in (0, 1, 2)]
        first += center * raw[0] - raw[1]
        second += center**2 * raw[0] - 2 * center * raw[1] + raw[2]
    return first, second


def population(n, level, digits, requested_states=None):
    with mp.workdps(digits):
        candidates = []
        for exponent in range(-30, 50):
            lower, upper = mp.power(2, exponent), mp.power(2, exponent + 1)
            p, q = cdf(lower), cdf(upper)
            probability = q**n - p**n
            if requested_states is not None:
                keep = exponent in requested_states
            else:
                keep = probability >= mp.mpf(STATE_THRESHOLD)
            if keep:
                candidates.append((exponent, lower, upper, p, q, probability))
        assert candidates
        mass = first = second = pair = conditional_mean_square = mp.mpf(0)
        layers = []
        for exponent, lower, upper, p, q, probability in candidates:
            step = mp.power(2, exponent - level)
            first_lower, second_lower = integrate_cells(step, lower)
            first_upper, second_upper = integrate_cells(step, upper)
            # Independent product-event subtraction: no r/(1-r) expression.
            first_numerator = q**(n - 1) * first_upper - p**(n - 1) * first_lower
            second_numerator = q**(n - 1) * second_upper - p**(n - 1) * second_lower
            pair_numerator = q**(n - 2) * first_upper**2 - p**(n - 2) * first_lower**2
            mean = first_numerator / probability
            moment2 = second_numerator / probability
            moment12 = pair_numerator / probability
            mass += probability
            first += first_numerator
            second += second_numerator
            pair += pair_numerator
            conditional_mean_square += probability * mean**2
            layers.append(dict(k=exponent, probability=probability, mean=mean,
                               moment2=moment2, moment12=moment12,
                               lower_first=first_lower, lower_second=second_lower,
                               upper_first=first_upper, upper_second=second_upper))
        mean = first / mass
        variance = second / mass - mean**2
        covariance = pair / mass - mean**2
        between = conditional_mean_square / mass - mean**2
        conditional = (pair - conditional_mean_square) / mass
        minimum, maximum = min(layer["k"] for layer in layers), max(layer["k"] for layer in layers)
        lower, upper = mp.power(2, minimum), mp.power(2, maximum + 1)
        lower_probability = cdf(lower)**n
        # Upper omitted second-moment bound uses |e_1|<=|X_1| and independence.
        upper_second_bound = survival_raw(2, upper) + (n - 1) * mp.exp(2) * survival_raw(0, upper)
        lower_second_bound = lower**2 * lower_probability
        return dict(mean=mean, var=variance, cov=covariance, between=between,
                    conditional=conditional, retained_mass=mass, layers=layers,
                    state_min=minimum, state_max=maximum,
                    omitted_probability_bound=lower_probability + n * survival_raw(0, upper),
                    omitted_marginal_second_moment_bound=lower_second_bound + upper_second_bound)


def relative_error(left, right):
    return abs(left - right) / max(abs(right), mp.mpf("1e-80"))


def text_number(value):
    return mp.nstr(value, 45)


def energy(n_grid, size, levels, layout, models):
    """Propagate audited moments with an independently constructed rational Green matrix."""
    h = mp.mpf(1) / (n_grid + 1)
    green = [[h**2 * min(i, j) * (n_grid + 1 - max(i, j)) / (n_grid + 1)
              for j in range(1, n_grid + 1)] for i in range(1, n_grid + 1)]
    count = n_grid // size
    if layout == "contiguous":
        blocks = [list(range(j * size, (j + 1) * size)) for j in range(count)]
    else:
        blocks = [list(range(j, n_grid, count)) for j in range(count)]
    assert len(levels) == count
    means = [mp.mpf(0)] * n_grid
    variance_energy = mp.mpf(0)
    for ids, level in zip(blocks, levels):
        model = models[(size, level)]
        mean, variance, covariance = (mp.mpf(model[key]) for key in ("mean", "var", "cov"))
        for i in ids:
            means[i] = mean
        diagonal = sum(green[i][i] for i in ids)
        block_sum = sum(green[i][j] for i in ids for j in ids)
        variance_energy += (variance - covariance) * diagonal + covariance * block_sum
    bias_energy = sum(means[i] * green[i][j] * means[j]
                      for i in range(n_grid) for j in range(n_grid))
    reference = h * ((mp.exp(2) - mp.e) * sum(green[i][i] for i in range(n_grid))
                     + mp.e * sum(map(sum, green)))
    mse = h * (variance_energy + bias_energy)
    return dict(energy_mse=mse, relative_rms=mp.sqrt(mse / reference),
                centered_energy=h * variance_energy, coherent_bias_energy=h * bias_energy)


def main():
    core_models, mp_models, rows = {}, {}, []
    for n, level in CONFIGURATIONS:
        producer = core.exact_model("lognormal", n, level, compute_reduced=False)
        core_models[(n, level)] = producer
        high = population(n, level, DIGITS[1])
        low = population(n, level, DIGITS[0])
        states = {layer["k"] for layer in producer["layers"]}
        same_states = population(n, level, DIGITS[1], states)
        mp_models[(n, level)] = high
        with mp.workdps(DIGITS[1]):
            precision_errors = {key: relative_error(low[key], high[key])
                                for key in ("mean", "var", "cov")}
            producer_errors = {key: relative_error(mp.mpf(producer[key]), high[key])
                               for key in ("mean", "var", "cov")}
            same_state_errors = {key: relative_error(mp.mpf(producer[key]), same_states[key])
                                 for key in ("mean", "var", "cov")}
            assert max(precision_errors.values()) < mp.mpf("1e-30")
            assert max(producer_errors.values()) < mp.mpf("1e-8")
            # Independently compare every retained rounding-cell integral pair.
            cells = []
            for layer in same_states["layers"]:
                exponent = layer["k"]
                for side, threshold in (("lower", 2.**exponent), ("upper", 2.**(exponent + 1))):
                    binary_first, binary_second = core.residual_integrals("lognormal", 2.**(exponent - level), threshold)
                    cells.append(dict(k=exponent, endpoint=side,
                                      mp_first=text_number(layer[side + "_first"]),
                                      mp_second=text_number(layer[side + "_second"]),
                                      binary64_first=binary_first, binary64_second=binary_second,
                                      absolute_first_difference=text_number(abs(mp.mpf(binary_first) - layer[side + "_first"])),
                                      absolute_second_difference=text_number(abs(mp.mpf(binary_second) - layer[side + "_second"]))))
            rows.append(dict(n=n, L=level, status="PASS", digits=list(DIGITS),
                             state_range=[high["state_min"], high["state_max"]],
                             binary64_state_range=[min(states), max(states)],
                             high_precision={key: text_number(high[key]) for key in ("mean", "var", "cov", "between", "conditional")},
                             binary64={key: producer[key] for key in ("mean", "var", "cov", "between", "conditional")},
                             precision_comparison_relative={key: text_number(value) for key, value in precision_errors.items()},
                             binary64_comparison_relative={key: text_number(value) for key, value in producer_errors.items()},
                             same_retained_states_comparison_relative={key: text_number(value) for key, value in same_state_errors.items()},
                             omitted_probability_bound=text_number(high["omitted_probability_bound"]),
                             omitted_marginal_second_moment_bound=text_number(high["omitted_marginal_second_moment_bound"]),
                             cell_integrals=cells))
        print(f"Checked lognormal n={n}, L={level} at {DIGITS[0]}/{DIGITS[1]} digits.", flush=True)

    plans = [(8, [3] * 16, "contiguous"), (8, [3] * 16, "interleaved"),
             (32, [4, 3, 4, 4], "contiguous"), (32, [4, 3, 4, 4], "interleaved"),
             (128, [3], "contiguous")]
    energy_rows = []
    with mp.workdps(DIGITS[1]):
        for size, levels, layout in plans:
            high_energy = energy(128, size, levels, layout, mp_models)
            producer_energy = energy(128, size, levels, layout, core_models)
            errors = {key: relative_error(producer_energy[key], high_energy[key]) for key in high_energy}
            assert max(errors.values()) < mp.mpf("1e-8")
            energy_rows.append(dict(N=128, block_size=size, L_by_block=levels, layout=layout,
                                    high_precision={key: text_number(value) for key, value in high_energy.items()},
                                    binary64_moment_propagation={key: text_number(value) for key, value in producer_energy.items()},
                                    relative_difference={key: text_number(value) for key, value in errors.items()}))
        maximum = max(mp.mpf(value) for row in rows for value in row["binary64_comparison_relative"].values())
        maximum_energy = max(mp.mpf(value) for row in energy_rows for value in row["relative_difference"].values())
        report = dict(status="PASS", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      producer_sha256=hashlib.sha256((ROOT / "experiments" / "run_revision.py").read_bytes()).hexdigest(),
                      mpmath_version=mp.__version__, digits=list(DIGITS), configurations=len(rows),
                      method="Analytical lognormal cell raw moments plus direct nested-event unconditional joint moments",
                      maximum_binary64_moment_relative_difference=text_number(maximum),
                      maximum_propagated_energy_relative_difference=text_number(maximum_energy),
                      interpretation="The population integration discrepancy is far below the percent-scale Monte Carlo miss; the noncovering interval remains reported without removal or adjustment.",
                      moment_results=rows, poisson_energy_results=energy_rows)
    (ROOT / "lognormal_precision_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: four lognormal moment configurations and five Poisson energy propagations.")
    print("Maximum relative moment / energy differences:", report["maximum_binary64_moment_relative_difference"], report["maximum_propagated_energy_relative_difference"])


if __name__ == "__main__":
    main()

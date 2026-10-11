"""Independent high-precision audit of heterogeneous lognormal block moments.

The producer is imported only for the final comparison. This evaluator uses
analytical cell raw moments and direct product-event subtraction for every
unconditional first/second/pair moment, rather than the producer's normalized
conditional covariance formula. All retained states are recomputed at 60 and
90 decimal digits; the broader population uses a 1e-40 state threshold.
"""
from pathlib import Path
import hashlib
import json
import sys

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
import elliptic2d_application as producer
import verify_poisson2d_nonidentical as rational_geometry

DIGITS = (60, 90)
THRESHOLD = "1e-40"


def tail_raw(power, threshold, scale, sigma):
    total = scale**power * mp.exp(power**2 * sigma**2 / 2)
    if threshold == 0:
        return total
    z = (mp.log(threshold / scale) - power * sigma**2) / (sigma * mp.sqrt(2))
    return total * mp.erfc(z) / 2


def cdf(threshold, scale, sigma):
    return mp.erfc(-mp.log(threshold / scale) / (sigma * mp.sqrt(2))) / 2


def cells(step, upper, scale, sigma):
    first = second = mp.mpf(0)
    for integer in range(int(upper / step) + 1):
        center = integer * step
        lower, higher = max(mp.mpf(0), center - step / 2), min(upper, center + step / 2)
        if higher <= lower:
            continue
        raw = [tail_raw(power, lower, scale, sigma) - tail_raw(power, higher, scale, sigma)
               for power in (0, 1, 2)]
        first += center * raw[0] - raw[1]
        second += center**2 * raw[0] - 2 * center * raw[1] + raw[2]
    return first, second


def evaluate(scales, level, digits, sigma=1, requested_states=None):
    with mp.workdps(digits):
        scales = [mp.mpf(str(value)) for value in scales]
        sigma = mp.mpf(str(sigma))
        n = len(scales)
        first = [mp.mpf(0)] * n
        raw = [[mp.mpf(0)] * n for _ in range(n)]
        mass = step_second = mp.mpf(0)
        retained_states = []
        minimum = maximum = None
        for k in range(-40, 36):
            a, b, step = mp.power(2, k), mp.power(2, k + 1), mp.power(2, k - level)
            p = [cdf(a, scale, sigma) for scale in scales]
            q = [cdf(b, scale, sigma) for scale in scales]
            lower_product, upper_product = mp.fprod(p), mp.fprod(q)
            probability = upper_product - lower_product
            keep = (k in requested_states if requested_states is not None
                    else probability >= mp.mpf(THRESHOLD))
            if not keep:
                continue
            lower_cells = [cells(step, a, scale, sigma) for scale in scales]
            upper_cells = [cells(step, b, scale, sigma) for scale in scales]
            for i in range(n):
                lower_except = mp.fprod(p[j] for j in range(n) if j != i)
                upper_except = mp.fprod(q[j] for j in range(n) if j != i)
                first[i] += upper_cells[i][0] * upper_except - lower_cells[i][0] * lower_except
                raw[i][i] += upper_cells[i][1] * upper_except - lower_cells[i][1] * lower_except
                for j in range(i):
                    lower_except_pair = mp.fprod(p[r] for r in range(n) if r not in (i, j))
                    upper_except_pair = mp.fprod(q[r] for r in range(n) if r not in (i, j))
                    value = upper_cells[i][0] * upper_cells[j][0] * upper_except_pair
                    value -= lower_cells[i][0] * lower_cells[j][0] * lower_except_pair
                    raw[i][j] += value
                    raw[j][i] += value
            mass += probability
            step_second += probability * step**2
            retained_states.append(k)
            minimum = k if minimum is None else min(minimum, k)
            maximum = k if maximum is None else max(maximum, k)
        covariance = [[raw[i][j] - first[i] * first[j] for j in range(n)] for i in range(n)]
        lower_limit, upper_limit = mp.power(2, minimum), mp.power(2, maximum + 1)
        omitted_probability = mp.fprod(cdf(lower_limit, scale, sigma) for scale in scales)
        omitted_probability += sum(tail_raw(0, upper_limit, scale, sigma) for scale in scales)
        # |e_i|<=|X_i|. On the lower event the norm is <=n*a².
        omitted_norm = n * lower_limit**2 * mp.fprod(
            cdf(lower_limit, scale, sigma) for scale in scales)
        total_second = [scale**2 * mp.exp(2 * sigma**2) for scale in scales]
        tails = [tail_raw(0, upper_limit, scale, sigma) for scale in scales]
        omitted_norm += sum(tail_raw(2, upper_limit, scales[i], sigma)
                            + total_second[i] * sum(tails[j] for j in range(n) if j != i)
                            for i in range(n))
        return dict(mean=first, cov=covariance, second=[raw[i][i] for i in range(n)],
                    mean_step2=step_second, retained_probability=mass,
                    states=retained_states, omitted_probability_bound=omitted_probability,
                    omitted_error_second_norm_bound=omitted_norm)


def formatted(value):
    return mp.nstr(value, 45)


def differences(left, right):
    keys = ("mean", "second", "cov")
    out = {}
    for key in keys:
        a = np.asarray(left[key], dtype=object).ravel()
        b = np.asarray(right[key], dtype=object).ravel()
        scale = max(abs(mp.mpf(value)) for value in b)
        absolute = max(abs(mp.mpf(x) - mp.mpf(y)) for x, y in zip(a, b))
        out[key] = dict(maximum_absolute=absolute,
                        maximum_absolute_relative_to_largest_entry=absolute / scale)
    out["mean_step2"] = dict(maximum_absolute=abs(mp.mpf(left["mean_step2"]) - right["mean_step2"]),
                            maximum_absolute_relative_to_largest_entry=
                            abs(mp.mpf(left["mean_step2"]) - right["mean_step2"]) / right["mean_step2"])
    return out


def propagate_energy(scales, model, grid, digits):
    """Independent exact-rational 2D inverse, converted at working precision."""
    hx, hy, rational_green = rational_geometry.geometry(grid, grid)
    with mp.workdps(digits):
        area = mp.mpf(hx.numerator) / hx.denominator * mp.mpf(hy.numerator) / hy.denominator
        green = [[mp.mpf(entry.numerator) / entry.denominator for entry in row]
                 for row in rational_green]
        n = len(scales)
        mean = [mp.mpf(entry) for entry in model["mean"]]
        cov = [[mp.mpf(entry) for entry in row] for row in model["cov"]]
        centered = area * sum(green[i][j] * cov[i][j] for i in range(n) for j in range(n))
        bias = area * sum(mean[i] * green[i][j] * mean[j] for i in range(n) for j in range(n))
        source_mean = [mp.mpf(str(scale)) * mp.exp(mp.mpf(".5")) for scale in scales]
        reference = area * (sum(mp.mpf(str(scales[i]))**2 * (mp.exp(2) - mp.e) * green[i][i]
                                for i in range(n))
                            + sum(source_mean[i] * green[i][j] * source_mean[j]
                                  for i in range(n) for j in range(n)))
        return dict(energy_mse=centered + bias, centered_energy=centered,
                    coherent_bias_energy=bias, relative_rms=mp.sqrt((centered + bias) / reference))


def main():
    profiles = [([.7, 1., 1.9, 4.], level, "four_distinct_scales") for level in (1, 3, 4)]
    profiles.append((producer.source_scales(4, "smooth").tolist(), 3, "sixteen_spatial_scales"))
    records = []
    for scales, level, name in profiles:
        binary = producer.block_moments(np.array(scales), level)
        high = evaluate(scales, level, DIGITS[1])
        low = evaluate(scales, level, DIGITS[0])
        same = evaluate(scales, level, DIGITS[1], requested_states={row["k"] for row in binary["layers"]})
        with mp.workdps(DIGITS[1]):
            errors = differences(binary, high)
            precision = differences(low, high)
            matched = differences(binary, same)
            assert max(value["maximum_absolute_relative_to_largest_entry"] for value in precision.values()) < mp.mpf("1e-25")
            assert max(value["maximum_absolute_relative_to_largest_entry"] for value in errors.values()) < mp.mpf("1e-8")
            grid = int(len(scales)**.5)
            high_energy = propagate_energy(scales, high, grid, DIGITS[1])
            binary_energy = propagate_energy(scales, binary, grid, DIGITS[1])
            energy_error = {key: abs(binary_energy[key] - high_energy[key]) / abs(high_energy[key])
                            for key in high_energy}
            assert max(energy_error.values()) < mp.mpf("1e-8")
            records.append(dict(profile=name, scales=scales, level=level, status="PASS",
                digits=list(DIGITS), producer_states=[row["k"] for row in binary["layers"]],
                high_precision_states=high["states"],
                binary64_comparison={key: {metric: formatted(value) for metric, value in error.items()}
                                     for key, error in errors.items()},
                matched_state_comparison={key: {metric: formatted(value) for metric, value in error.items()}
                                          for key, error in matched.items()},
                precision_comparison={key: {metric: formatted(value) for metric, value in error.items()}
                                      for key, error in precision.items()},
                high_precision_mean=[formatted(value) for value in high["mean"]],
                high_precision_covariance=[[formatted(value) for value in row] for row in high["cov"]],
                energy_propagation=dict(grid=[grid, grid],
                    method="Exact rational 2D inverse obtained by independent Gaussian elimination",
                    high_precision={key: formatted(value) for key, value in high_energy.items()},
                    binary64_moment_propagation={key: formatted(value) for key, value in binary_energy.items()},
                    relative_difference={key: formatted(value) for key, value in energy_error.items()}),
                omitted_probability_bound=formatted(high["omitted_probability_bound"]),
                omitted_error_second_norm_bound=formatted(high["omitted_error_second_norm_bound"])))
        print(f"PASS: {name}, n={len(scales)}, L={level}, {DIGITS[0]}/{DIGITS[1]} digits", flush=True)
    maximum = max(mp.mpf(error["maximum_absolute_relative_to_largest_entry"])
                  for record in records for error in record["binary64_comparison"].values())
    path = Path(__file__).resolve()
    report = dict(status="PASS", configurations=len(records), digits=list(DIGITS),
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  producer_sha256=hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest(),
                  method="Analytical scaled-lognormal cell moments and direct unconditional product-event subtraction",
                  maximum_binary64_scaled_error=formatted(maximum), results=records)
    path.with_name("nonidentical_lognormal_precision_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(records)} heterogeneous moments configurations; largest scaled binary64 error {formatted(maximum)}")


if __name__ == "__main__":
    main()

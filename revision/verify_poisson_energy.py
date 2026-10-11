"""Independent exact-rational audit of the partitioned Poisson energy formula.

Run from SISC_submission: python revision/verify_poisson_energy.py

This oracle imports neither the population evaluator nor the Poisson experiment.
It enumerates complete independent finite-support right-hand sides, quantizes
each declared block using rational arithmetic, solves with the exact discrete
Green matrix, and evaluates energy through differences of the solution. A
second calculation propagates independently enumerated block moments. Cases
cover contiguous/interleaved partitions, unequal block sizes, varying precision,
different block input laws and biases, covariance signs, zeros and ties.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json


LAWS = {
    "deflation": ((F(13, 10), F(1, 2)), (F(41, 20), F(1, 2))),
    "common_error": ((F(6, 5), F(1, 2)), (F(11, 5), F(1, 2))),
    "zero_ties": ((F(0), F(1, 4)), (F(1, 8), F(1, 4)),
                  (F(9, 4), F(1, 2))),
    "symmetric": ((F(-7, 8), F(1, 4)), (F(0), F(1, 2)),
                  (F(7, 8), F(1, 4))),
    "endpoints": ((F(1), F(1, 4)), (F(3, 2), F(1, 4)),
                  (F(2), F(1, 4)), (F(5, 2), F(1, 4))),
}


def quantize_block(values, level):
    """Find the binade by rational comparisons; implement exact ties-to-even."""
    maximum = max(map(abs, values))
    if maximum == 0:
        return [F(0) for _ in values], 0, 1
    exponent = 0
    scale = F(1)
    while maximum < scale:
        scale /= 2
        exponent -= 1
    while maximum >= 2 * scale:
        scale *= 2
        exponent += 1
    step = scale / 2**level
    errors, ties = [], 0
    for value in values:
        scaled = value / step
        lower = scaled.numerator // scaled.denominator
        remainder = scaled - lower
        if remainder == F(1, 2):
            ties += 1
            rounded = lower if lower % 2 == 0 else lower + 1
        else:
            rounded = lower + int(remainder > F(1, 2))
        errors.append(step * rounded - value)
    return errors, ties, 0


def weighted_blocks(law_names):
    for outcome in product(*(LAWS[name] for name in law_names)):
        probability = F(1)
        for _, weight in outcome:
            probability *= weight
        yield [value for value, _ in outcome], probability


def moments_by_enumeration(law, size, level):
    """No truncated-integral formula is used to obtain these block moments."""
    mean = second = pair = mass = F(0)
    for values, probability in weighted_blocks([law] * size):
        errors, _, _ = quantize_block(values, level)
        mass += probability
        mean += probability * errors[0]
        second += probability * errors[0] ** 2
        if size > 1:
            pair += probability * errors[0] * errors[1]
    assert mass == 1
    return mean, second - mean**2, pair - mean**2 if size > 1 else F(0)


def discrete_green(size):
    h = F(1, size + 1)
    green = [[h**2 * F(min(i, j) * (size + 1 - max(i, j)), size + 1)
              for j in range(1, size + 1)] for i in range(1, size + 1)]
    # Check each entry of A G=I using the tridiagonal stencil, independently
    # of the energy identities that are tested below.
    for i in range(size):
        for j in range(size):
            entry = 2 * green[i][j]
            if i:
                entry -= green[i - 1][j]
            if i + 1 < size:
                entry -= green[i + 1][j]
            assert entry / h**2 == int(i == j)
    assert sum(green[i][i] for i in range(size)) == h**2 * F(size * (size + 2), 6)
    assert sum(map(sum, green)) == h**2 * F(size * (size + 1) * (size + 2), 12)
    return h, green


def solution_energy(rhs, h, green):
    solution = [F(0)]
    solution.extend(sum(row[j] * rhs[j] for j in range(len(rhs))) for row in green)
    solution.append(F(0))
    # This is the gradient definition h sum((u_(i+1)-u_i)/h)^2, rather than
    # the e^T A^-1 e quadratic form used by the population prediction.
    return sum((right - left)**2 / h for left, right in zip(solution, solution[1:]))


def bilinear(left, matrix, right):
    return sum(left[i] * matrix[i][j] * right[j]
               for i in range(len(left)) for j in range(len(right)))


def configuration(name, blocks, laws, levels, layout):
    return dict(name=name, blocks=blocks, laws=laws, levels=levels, layout=layout)


CASES = [
    configuration("single_block", [[0, 1, 2, 3]], ["deflation"], [2], "single"),
    configuration("equal_contiguous", [[0, 1], [2, 3]], ["deflation"] * 2, [2, 2], "contiguous"),
    configuration("equal_interleaved", [[0, 2], [1, 3]], ["deflation"] * 2, [2, 2], "interleaved"),
    configuration("variable_precision_contiguous", [[0, 1], [2, 3]], ["deflation"] * 2, [1, 2], "contiguous"),
    configuration("variable_precision_interleaved", [[0, 2], [1, 3]], ["deflation"] * 2, [1, 2], "interleaved"),
    configuration("different_laws_contiguous", [[0, 1], [2, 3]], ["deflation", "common_error"], [2, 2], "contiguous"),
    configuration("different_laws_interleaved", [[0, 2], [1, 3]], ["deflation", "common_error"], [2, 2], "interleaved"),
    configuration("three_laws_contiguous", [[0, 1], [2, 3], [4, 5]],
                  ["deflation", "common_error", "zero_ties"], [2, 2, 0], "contiguous"),
    configuration("three_laws_interleaved", [[0, 3], [1, 4], [2, 5]],
                  ["deflation", "common_error", "zero_ties"], [2, 2, 0], "interleaved"),
    configuration("unequal_sizes", [[0, 1], [2, 3, 4]], ["deflation", "zero_ties"], [2, 1], "contiguous"),
    configuration("symmetry_variable_precision", [[0, 2], [1, 3]], ["symmetric"] * 2, [0, 2], "interleaved"),
    configuration("dyadic_endpoints_ties", [[0, 1], [2, 3]], ["endpoints"] * 2, [0, 1], "contiguous"),
    configuration("zero_state", [[0, 1, 2]], ["zero_ties"], [2], "single"),
]


def verify_case(config):
    blocks, laws, levels = config["blocks"], config["laws"], config["levels"]
    size = sum(map(len, blocks))
    assert sorted(i for ids in blocks for i in ids) == list(range(size))
    assert len(blocks) == len(laws) == len(levels)
    h, green = discrete_green(size)
    coordinate_laws = [None] * size
    means, covariance = [F(0)] * size, [[F(0)] * size for _ in range(size)]
    block_moments = []
    variance_energy, within_block_bias_energy = F(0), F(0)
    for ids, law, level in zip(blocks, laws, levels):
        mean, variance, pair_covariance = moments_by_enumeration(law, len(ids), level)
        block_moments.append(dict(mean=str(mean), variance=str(variance), covariance=str(pair_covariance)))
        for i in ids:
            coordinate_laws[i] = law
            means[i] = mean
            for j in ids:
                covariance[i][j] = variance if i == j else pair_covariance
        diagonal = sum(green[i][i] for i in ids)
        block_sum = sum(green[i][j] for i in ids for j in ids)
        variance_energy += h * ((variance - pair_covariance) * diagonal + pair_covariance * block_sum)
        within_block_bias_energy += h * mean**2 * block_sum
    bias_energy = h * bilinear(means, green, means)
    predicted = variance_energy + bias_energy

    source_means, source_variances = [], []
    for law in coordinate_laws:
        average = sum(value * probability for value, probability in LAWS[law])
        source_means.append(average)
        source_variances.append(sum(value**2 * probability for value, probability in LAWS[law]) - average**2)
    reference_prediction = h * (sum(source_variances[i] * green[i][i] for i in range(size))
                                + bilinear(source_means, green, source_means))

    direct_energy = direct_reference = mass = F(0)
    direct_means = [F(0)] * size
    direct_second = [[F(0)] * size for _ in range(size)]
    count = tie_count = zero_block_count = 0
    for rhs, probability in weighted_blocks(coordinate_laws):
        errors = [F(0)] * size
        for ids, level in zip(blocks, levels):
            block_errors, ties, zeros = quantize_block([rhs[i] for i in ids], level)
            tie_count += ties
            zero_block_count += zeros
            for i, error in zip(ids, block_errors):
                errors[i] = error
        mass += probability
        direct_energy += probability * solution_energy(errors, h, green)
        direct_reference += probability * solution_energy(rhs, h, green)
        for i in range(size):
            direct_means[i] += probability * errors[i]
            for j in range(size):
                direct_second[i][j] += probability * errors[i] * errors[j]
        count += 1
    assert mass == 1
    assert direct_energy == predicted
    assert direct_reference == reference_prediction
    assert direct_means == means
    for i in range(size):
        for j in range(size):
            assert direct_second[i][j] - direct_means[i] * direct_means[j] == covariance[i][j]

    return dict(**config, n=size, enumerated_rhs=count, status="PASS",
                block_moments=block_moments, direct_energy_mse=str(direct_energy),
                predicted_energy_mse=str(predicted), centered_energy=str(variance_energy),
                coherent_squared_bias_energy=str(bias_energy),
                cross_block_squared_bias_energy=str(bias_energy - within_block_bias_energy),
                reference_expected_energy=str(direct_reference),
                relative_rms_squared=str(direct_energy / direct_reference),
                energy_difference="0", reference_difference="0", covariance_difference="0",
                tie_encounters=tie_count, zero_block_encounters=zero_block_count)


def main():
    rows = [verify_case(config) for config in CASES]
    signs = {int(F(moment["covariance"]) > 0) - int(F(moment["covariance"]) < 0)
             for row in rows for moment in row["block_moments"]}
    assert signs == {-1, 0, 1}
    assert any(row["tie_encounters"] for row in rows)
    assert any(row["zero_block_encounters"] for row in rows)
    assert any(len(set(moment["mean"] for moment in row["block_moments"])) > 1
               and F(row["cross_block_squared_bias_energy"]) != 0 for row in rows)
    by_name = {row["name"]: row for row in rows}
    assert F(by_name["equal_contiguous"]["direct_energy_mse"]) == F(1, 3200)
    assert by_name["equal_contiguous"]["direct_energy_mse"] != by_name["equal_interleaved"]["direct_energy_mse"]
    source = Path(__file__).resolve()
    report = dict(status="PASS", exact_arithmetic="fractions.Fraction",
                  configurations=len(rows), enumerated_rhs=sum(row["enumerated_rhs"] for row in rows),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  independence="No imports from the prediction or experiment implementations",
                  direct_energy_method="Exact Green solve, then squared first differences with zero boundary values",
                  moment_method="Independent complete block enumeration; no truncated-integral formula",
                  checks=["A G = I and Green trace/constant-direction identities",
                          "Complete joint quantization-energy enumeration versus block-moment propagation",
                          "Complete error covariance matrix, including zero off-block covariances",
                          "Variable block precision and nonidentical block biases",
                          "Contiguous and interleaved spatial support",
                          "Unequal block sizes, both covariance signs, zeros, ties and dyadic endpoints",
                          "Reference expected energy and relative RMS squared"],
                  laws={name: [[str(value), str(probability)] for value, probability in law]
                        for name, law in LAWS.items()}, results=rows)
    destination = source.with_name("poisson_energy_audit.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(rows)} exact Poisson configurations; {report['enumerated_rhs']} enumerated RHSs")
    print("Includes variable precision, nonidentical biases, both spatial layouts, and the full covariance/reference energy checks.")


if __name__ == "__main__":
    main()

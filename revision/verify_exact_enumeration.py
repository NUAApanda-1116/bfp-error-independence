"""Independent rational-arithmetic oracle for the finite-block theory.

Enumerate every input block; compare direct joint moments with truncated
one-dimensional moment formulas. No Monte Carlo or floating-point quadrature.
Run: python revision/verify_exact_enumeration.py
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json


def binade(x):
    if x == 0:
        return None
    k = 0
    x = abs(x)
    while x < 1:
        x *= 2
        k -= 1
    while x >= 2:
        x /= 2
        k += 1
    return k


def nearest_even(x):
    j = x.numerator // x.denominator
    theta = x - j
    return j + int(theta > F(1, 2) or (theta == F(1, 2) and j % 2))


def residual(x, k, precision):
    if k is None:
        return F(0)
    step = F(2) ** (k - precision)
    return step * nearest_even(x / step) - x


def verify(atoms, probabilities, n, precision):
    zero = F(0)
    strata = defaultdict(lambda: [zero] * 4)
    mean = second = joint = weighted_second = zero
    weights = [F((-1) ** i) + F(1, 5) for i in range(n)]
    for indices in product(range(len(atoms)), repeat=n):
        values = [atoms[i] for i in indices]
        probability = F(1)
        for i in indices:
            probability *= probabilities[i]
        k = binade(max(abs(x) for x in values))
        errors = [residual(x, k, precision) for x in values]
        stratum = strata[k]
        for j, statistic in enumerate([F(1), errors[0], errors[0] ** 2,
                                       errors[0] * errors[1]]):
            stratum[j] += probability * statistic
        mean += probability * errors[0]
        second += probability * errors[0] ** 2
        joint += probability * errors[0] * errors[1]
        weighted_second += probability * sum(w * e for w, e in zip(weights, errors)) ** 2

    formula_mean = formula_second = formula_joint = mean_square_conditional = zero
    for k, (probability, m1, m2, m12) in strata.items():
        if k is None:
            assert m1 == m2 == m12 == 0
            assert probability == sum(p for x, p in zip(atoms, probabilities) if x == 0) ** n
            continue
        a, b = F(2) ** k, F(2) ** (k + 1)
        p = sum(prob for x, prob in zip(atoms, probabilities) if abs(x) < a)
        q = sum(prob for x, prob in zip(atoms, probabilities) if abs(x) < b)
        assert probability == q ** n - p ** n
        integrals = []
        for endpoint in (a, b):
            integrals.append([
                sum(prob * residual(x, k, precision) ** power
                    for x, prob in zip(atoms, probabilities) if abs(x) < endpoint)
                for power in (1, 2)])
        (A1, A2), (B1, B2) = integrals
        exact_m = (q ** (n - 1) * B1 - p ** (n - 1) * A1) / probability
        exact_second = (q ** (n - 1) * B2 - p ** (n - 1) * A2) / probability
        exact_joint = (q ** (n - 2) * B1 ** 2 - p ** (n - 2) * A1 ** 2) / probability
        assert exact_m == m1 / probability
        assert exact_second == m2 / probability
        assert exact_joint == m12 / probability
        if p:
            r, u, v = (p / q) ** n, A1 / p, B1 / q
            c = -r * (v - u) ** 2 / (1 - r) ** 2
        else:
            c = zero
        assert c == exact_joint - exact_m ** 2
        assert c <= 0
        formula_mean += probability * exact_m
        formula_second += probability * exact_second
        formula_joint += probability * exact_joint
        mean_square_conditional += probability * exact_m ** 2
    assert (formula_mean, formula_second, formula_joint) == (mean, second, joint)
    variance, covariance = second - mean ** 2, joint - mean ** 2
    B = mean_square_conditional - mean ** 2
    assert variance - covariance >= 0
    assert variance + (n - 1) * covariance >= 0
    predicted_weighted_mse = ((variance - covariance) * sum(w ** 2 for w in weights)
                              + (covariance + mean ** 2) * sum(weights) ** 2)
    assert weighted_second == predicted_weighted_mse
    inflation = None
    if variance:
        inflation = 1 + (n - 1) * covariance / variance
        upper = 1 + (n - 1) * B / variance
        assert n * B / variance <= inflation <= upper
        assert 0 <= upper - inflation <= 1 - B / variance <= 1
    return dict(n=n, L=precision, states=len(atoms) ** n,
                mean=str(mean), variance=str(variance), covariance=str(covariance),
                inflation=str(inflation) if inflation is not None else None)


if __name__ == "__main__":
    laws = {
        "manuscript_counterexample": ([F(13, 10), F(41, 20)], [F(1, 2)] * 2),
        "negative_covariance": ([F(3, 10), F(11, 10)], [F(1, 2)] * 2),
        "two_atom_asymmetric": ([F(3, 4), F(5, 4)], [F(1, 2)] * 2),
        "zero_and_dyadic_endpoints": ([F(0), F(1, 2), F(1), F(2)], [F(1, 4)] * 4),
        "symmetric_ties": ([F(-3, 4), F(0), F(3, 4)], [F(1, 4), F(1, 2), F(1, 4)]),
        "unequal_atom_weights": ([F(-1, 3), F(1, 3), F(7, 6)], [F(1, 5), F(3, 10), F(1, 2)]),
    }
    records = []
    for name, (atoms, probabilities) in laws.items():
        for n in (2, 3, 4, 6):
            for precision in (0, 1, 2, 4):
                record = verify(atoms, probabilities, n, precision)
                records.append(dict(law=name, **record))
    output = Path(__file__).with_name("exact_enumeration_audit.json")
    output.write_text(json.dumps(dict(passed=len(records),
                                    enumerated_states=sum(r["states"] for r in records),
                                    records=records), indent=2), encoding="utf-8")
    print(f"PASS: {len(records)} exact rational configurations; "
          f"{sum(r['states'] for r in records)} enumerated input blocks")
    print("Includes zero blocks, dyadic endpoints, ties to even, negative covariance, "
          "weighted MSE, and both covariance eigenvalues.")

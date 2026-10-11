"""Exact rational oracle for independent inputs with different laws.

Every joint block is enumerated directly; no conditional sampler or numerical
integration is used.  Run: python revision/verify_nonidentical.py
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json


def exponent(values):
    maximum = max(map(abs, values))
    if not maximum:
        return None
    k = 0
    while maximum < 1:
        maximum *= 2
        k -= 1
    while maximum >= 2:
        maximum /= 2
        k += 1
    return k


def error(x, k, L):
    if k is None:
        return F(0)
    s = F(2) ** (k - L)
    y = x / s
    lower = y.numerator // y.denominator
    fraction = y - lower
    rounded = lower + int(fraction > F(1, 2)
                          or (fraction == F(1, 2) and lower % 2))
    return s * rounded - x


def verify(laws, L):
    n = len(laws)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    weights = [F((-1) ** i) + F(1, 5) for i in range(n)]
    size = 1 + 2 * n + len(pairs)
    strata = defaultdict(lambda: [F(0) for _ in range(size)])
    direct_weighted_mse = F(0)
    states = 0
    for block in product(*[list(zip(*law)) for law in laws]):
        values = [x for x, _ in block]
        probability = F(1)
        for _, p in block:
            probability *= p
        k = exponent(values)
        residuals = [error(x, k, L) for x in values]
        statistics = ([F(1)] + residuals + [e * e for e in residuals]
                      + [residuals[i] * residuals[j] for i, j in pairs])
        for idx, statistic in enumerate(statistics):
            strata[k][idx] += probability * statistic
        direct_weighted_mse += probability * sum(
            w * e for w, e in zip(weights, residuals)) ** 2
        states += 1

    first = [F(0)] * n
    second = [F(0)] * n
    joint = [F(0)] * len(pairs)
    signs = set()
    zero_lower_states = 0
    positive_example = negative_example = None
    for k, statistics in strata.items():
        probability = statistics[0]
        if k is None:
            predicted_zero_mass = F(1)
            for atoms, probabilities in laws:
                predicted_zero_mass *= sum(p for x, p in zip(atoms, probabilities) if x == 0)
            assert probability == predicted_zero_mass
            assert all(x == 0 for x in statistics[1:])
            continue
        a, b = F(2) ** k, F(2) ** (k + 1)
        p = [sum(prob for x, prob in zip(*law) if abs(x) < a) for law in laws]
        q = [sum(prob for x, prob in zip(*law) if abs(x) < b) for law in laws]
        lower_mass = upper_mass = F(1)
        for pi, qi in zip(p, q):
            lower_mass *= pi
            upper_mass *= qi
        assert probability == upper_mass - lower_mass
        z = [[sum(prob * error(x, k, L) ** ell
                  for x, prob in zip(*law) if abs(x) < b) / qi
              for ell in (1, 2)] for law, qi in zip(laws, q)]
        r = lower_mass / upper_mass
        if lower_mass:
            u = [[sum(prob * error(x, k, L) ** ell
                      for x, prob in zip(*law) if abs(x) < a) / pi
                  for ell in (1, 2)] for law, pi in zip(laws, p)]
            m = [(z[i][0] - r * u[i][0]) / (1 - r) for i in range(n)]
            t = [(z[i][1] - r * u[i][1]) / (1 - r) for i in range(n)]
            cross = [(z[i][0] * z[j][0] - r * u[i][0] * u[j][0]) / (1 - r)
                     for i, j in pairs]
            covariances = [-r * (z[i][0] - u[i][0]) * (z[j][0] - u[j][0]) / (1 - r) ** 2
                           for i, j in pairs]
        else:
            zero_lower_states += 1
            m = [z[i][0] for i in range(n)]
            t = [z[i][1] for i in range(n)]
            cross = [z[i][0] * z[j][0] for i, j in pairs]
            covariances = [F(0)] * len(pairs)
        assert m == [x / probability for x in statistics[1:1 + n]]
        assert t == [x / probability for x in statistics[1 + n:1 + 2 * n]]
        assert cross == [x / probability for x in statistics[1 + 2 * n:]]
        for idx, (i, j) in enumerate(pairs):
            c = covariances[idx]
            assert c == cross[idx] - m[i] * m[j]
            signs.add((c > 0) - (c < 0))
            if c > 0 and positive_example is None:
                positive_example = dict(k=k, pair=[i, j], conditional_covariance=str(c))
            if c < 0 and negative_example is None:
                negative_example = dict(k=k, pair=[i, j], conditional_covariance=str(c))
            joint[idx] += probability * cross[idx]
        for i in range(n):
            first[i] += probability * m[i]
            second[i] += probability * t[i]
    predicted_weighted_mse = sum(w * w * t for w, t in zip(weights, second))
    predicted_weighted_mse += 2 * sum(weights[i] * weights[j] * joint[idx]
                                     for idx, (i, j) in enumerate(pairs))
    assert predicted_weighted_mse == direct_weighted_mse
    return dict(n=n, L=L, enumerated_blocks=states,
                all_zero_state_included=None in strata,
                zero_lower_states=zero_lower_states, conditional_covariance_signs=sorted(signs),
                positive_example=positive_example, negative_example=negative_example,
                weighted_mse=str(predicted_weighted_mse))


if __name__ == '__main__':
    positive = ([F(13, 10), F(41, 20)], [F(1, 2)] * 2)
    positive_unequal = (positive[0], [F(1, 3), F(2, 3)])
    positive_third = (positive[0], [F(1, 4), F(3, 4)])
    negative = ([-x for x in positive[0]], [F(1, 3), F(2, 3)])
    zeros_dyadic = ([F(0), F(1, 2), F(1), F(2)], [F(1, 4)] * 4)
    symmetric_ties = ([F(-3, 4), F(0), F(3, 4)], [F(1, 4), F(1, 2), F(1, 4)])
    mixed = ([F(-1, 3), F(1, 3), F(7, 6)], [F(1, 5), F(3, 10), F(1, 2)])
    dyadic_unequal = ([F(0), F(1), F(4)], [F(1, 10), F(2, 5), F(1, 2)])
    profiles = {
        'same_side_different_probabilities': [positive, positive_unequal, positive_third],
        'opposite_sides': [positive, negative, positive_third],
        'zero_endpoints_and_ties': [zeros_dyadic, symmetric_ties, mixed],
        'different_support_and_zero_lower_event': [zeros_dyadic, dyadic_unequal, mixed],
    }
    records = []
    for name, laws in profiles.items():
        for n in (2, 3):
            for L in (0, 1, 2, 4):
                records.append(dict(profile=name, **verify(laws[:n], L)))
    signs = {s for record in records for s in record['conditional_covariance_signs']}
    assert {-1, 0, 1} <= signs
    assert any(record['all_zero_state_included'] for record in records)
    assert any(record['zero_lower_states'] for record in records)
    audit = dict(passed=len(records), enumerated_blocks=sum(r['enumerated_blocks'] for r in records),
                 exact_arithmetic='fractions.Fraction', records=records)
    Path(__file__).with_suffix('.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f"PASS: {audit['passed']} nonidentical-law configurations; "
          f"{audit['enumerated_blocks']} exact blocks; positive, negative, and zero conditional covariance")

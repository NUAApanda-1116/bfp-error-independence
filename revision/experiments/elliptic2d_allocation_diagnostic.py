"""Post-hoc analytical diagnostic: full versus diagonal covariance placement.

No source vectors are drawn and no primary codec family is changed. This
diagnostic prevents attributing an improvement over the scale-only heuristic
specifically to covariance when a marginal-moment optimizer makes the same map.
"""
import json
from pathlib import Path

import numpy as np
import elliptic2d_application as exp

OUT = Path(__file__).resolve().parent


def main():
    h, green, eigenvalues, _ = exp.geometry(32)
    declarations = json.loads((OUT/"elliptic2d_prespecification.json").read_text(encoding="utf-8"))
    rows = []
    for profile in exp.PROFILES:
        scales = exp.source_scales(32, profile)
        cache = {}
        for cfg in declarations["configurations"][f"m32_{profile}"]:
            if cfg["precision_policy"] != "energy":
                continue
            ids_list = [np.asarray(indices) for indices in cfg["block_indices"]]
            base, extra = cfg["base_L"], cfg["higher_L_blocks"]
            models = []
            mean = np.zeros(len(scales))
            for ids in ids_list:
                pair = []
                for level in (base, base+1):
                    key = (tuple(scales[ids]), level, 1.)
                    if key not in cache:
                        cache[key] = exp.block_moments(scales[ids], level)
                    pair.append(cache[key])
                models.append(pair)
                mean[ids] = pair[0]["mean"]
            deltas = [pair[1]["mean"]-pair[0]["mean"] for pair in models]
            local_green = [green[np.ix_(ids, ids)] for ids in ids_list]
            centered_changes = [float(np.dot(np.diag(g), np.diag(pair[1]["cov"]-pair[0]["cov"])))
                                for pair, g in zip(models, local_green)]
            high = set()
            for _ in range(extra):
                action = green@mean
                improvements = {j: centered_changes[j]+2*float(deltas[j]@action[ids])
                                  +float(deltas[j]@local_green[j]@deltas[j])
                                for j, ids in enumerate(ids_list) if j not in high}
                best = min(improvements, key=lambda j: (improvements[j], j))
                high.add(best)
                mean[ids_list[best]] += deltas[best]
            levels = [base+int(j in high) for j in range(cfg["blocks"])]
            diagcfg = dict(cfg, L_by_block=levels)
            full = exp.population(scales, cfg, h, green, eigenvalues, cache)
            diag = exp.population(scales, diagcfg, h, green, eigenvalues, cache)
            rows.append(dict(profile=profile, codec=cfg["id"],
                precision_maps_equal=levels == cfg["L_by_block"],
                changed_block_count=sum(a != b for a, b in zip(levels, cfg["L_by_block"])),
                marginal_greedy_levels=levels, full_greedy_levels=cfg["L_by_block"],
                full_greedy_exact_relative_rms=full["exact_relative_rms"],
                marginal_greedy_exact_relative_rms=diag["exact_relative_rms"],
                full_fractional_rms_improvement=1-full["exact_relative_rms"]/diag["exact_relative_rms"]))
    (OUT/"elliptic2d_allocation_diagnostic.json").write_text(json.dumps(dict(
        timing="After sampling; analytical diagnostic only, no new primary codec or Monte Carlo selection",
        results=rows), indent=2)+"\n", encoding="utf-8")
    for row in rows:
        print(row["profile"], row["codec"], "changedblocks", row["changed_block_count"],
              "fullvsdiagreduction", row["full_fractional_rms_improvement"])


if __name__ == "__main__":
    main()

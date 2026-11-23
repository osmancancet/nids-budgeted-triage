"""Source of the Fisher load offset on held-out-family streams (1% budget, 20% warm-up, last 50% evaluated).

All held-out families of one dataset and seed use the same random order of the benign flows (make_stream is
seeded by the seed only), and the kNN novelty score does not depend on the detector or the held-out family.
The Fisher priority of the benign flows is therefore almost the same across families and models of one
dataset and seed, so its warm-up load is close to one draw per (dataset, seed) stream. For every run we
compare the load in the original order with the load after an independent random reordering of the same
stream (seed 1000 + run index), for the detector, novelty and Fisher priorities, and the expected load of
the same flows under random order (mean over 100 random orders), which is what Proposition 1 describes.

  python revision_offset_order.py   -> results/revision/offset_order.csv
"""
import os
import numpy as np
import pandas as pd

import revision_analysis as A

OUT = A.OUT


NPERM = 100


def load(s, n):
    w, ev = int(0.2 * n), slice(n // 2, n)
    t = np.quantile(s[:w], 0.99)
    return (s[ev] >= t).mean() / 0.01


def main():
    rows = []
    for i, (key, z) in enumerate(A.load_os()):
        ctx = A.Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        det, nov, y, un = z["det"].astype(np.float64), z["nov"].astype(np.float64), z["y"], z["unseen"]
        idx = A.make_stream(y, 0.01, np.random.default_rng(key["seed"] + 11), attack_mask=un)
        pr, _, _ = A.priorities(ctx, det[idx], nov[idx])
        n = len(idx)
        rg = np.random.default_rng(1000 + i)
        perms = [rg.permutation(n) for _ in range(NPERM)]
        r = dict(key)
        for nm in ["detector", "novelty", "fisher"]:
            r[f"load_{nm}"] = load(pr[nm], n)
            ls = [load(pr[nm][p], n) for p in perms]
            r[f"load_{nm}_reordered"] = ls[0]
            r[f"load_{nm}_expected"] = float(np.mean(ls))
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "offset_order.csv"), index=False)
    print(d.groupby("model")[[c for c in d.columns if c.startswith("load")]].median().round(3).to_string())


if __name__ == "__main__":
    main()

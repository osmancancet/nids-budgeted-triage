"""Realised queue load of the warm-up budget rule for the weighted Fisher priority (w = 0.7), next to
equal-weight Fisher and the detector, on the stored LightGBM scores of Protocols OS and XD
(streams, warm-up windows and evaluation segment as in revision_analysis.run_os / run_xd).

  python revision_wf_load.py   -> results/revision/wf_load.csv
"""
import os
import numpy as np
import pandas as pd

from revision_analysis import Ctx, priorities, make_stream, warmup_threshold, load_os, load_xd, OUT

BUDGETS = [0.01, 0.02, 0.05]
RULES = ["detector", "fisher", "wfisher0.7"]
WF = 0.2


def rows_for(key, ctx, det, nov, y, rng_seed, mask=None):
    idx = make_stream(y, 0.01, np.random.default_rng(rng_seed), attack_mask=mask)
    ys = y[idx]
    pr, _, _ = priorities(ctx, det[idx], nov[idx], extra=True)
    n = len(ys)
    ev = slice(n // 2, n)
    out = []
    for c in BUDGETS:
        for r in RULES:
            s = pr[r]
            q = s[ev] >= warmup_threshold(s, WF, c)
            out.append(dict(key, budget=c, priority=r, load=q.mean() / c, recall=q[ys[ev] == 1].mean()))
    return out


def main():
    os.environ["MODELS"] = "LGBM"
    rows = []
    for key, z in load_os():
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        rows += rows_for(dict(key, regime="Unseen family"), ctx, z["det"].astype(np.float64),
                         z["nov"].astype(np.float64), z["y"], key["seed"] + 11, z["unseen"])
    for key, z in load_xd():
        if key["model"] != "LGBM":
            continue
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        for tgt in z["targets"]:
            tgt = str(tgt)
            reg = "In-domain" if tgt == key["source"] else "Cross-dataset"
            rows += rows_for(dict(key, target=tgt, regime=reg), ctx, z[f"{tgt}__det"].astype(np.float64),
                             z[f"{tgt}__nov"].astype(np.float64), z[f"{tgt}__y"], key["seed"] + 7)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "wf_load.csv"), index=False)
    d = pd.DataFrame(rows)
    print(d[d.budget == 0.01].groupby(["regime", "priority"]).load.describe(percentiles=[.1, .5, .9]))


if __name__ == "__main__":
    main()

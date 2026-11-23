"""Weighted Fisher (w = 0.7) against the detector alone for all four model families (Protocol OS, five seeds).

The stored held-out-family runs of each model are reused with the stream of Section 3.8 (seed + 11), so
the only change from the LightGBM analysis of Section 4.5 is the detector that supplies the scores.

  python revision_models_wf.py   -> results/revision/os_models_wf.csv
"""
import os
import glob
import numpy as np
import pandas as pd

from revision_analysis import Ctx, priorities, topk_mask, make_stream, DUMP, OUT

BUDGETS = [0.01, 0.02, 0.05]


def main():
    rows = []
    for fpath in sorted(glob.glob(os.path.join(DUMP, "os", "*.npz"))):
        parts = os.path.basename(fpath)[:-4].split("__")
        z = np.load(fpath, allow_pickle=False)
        if len(parts) < 4 or "held_out" not in z.files:
            continue
        name, model, seed = parts[0], parts[-2], int(parts[-1][1:])
        yt, un = z["y"].astype(int), z["unseen"]
        idx = make_stream(yt, 0.01, np.random.default_rng(seed + 11), attack_mask=un)
        ys = yt[idx]
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        pr, _, _ = priorities(ctx, z["det"][idx], z["nov"][idx], extra=True)
        for c in BUDGETS:
            for rule in ["detector", "fisher", "wfisher0.7"]:
                q = topk_mask(pr[rule], c)
                rows.append(dict(dataset=name, held_out=str(z["held_out"]), model=model, seed=seed, budget=c,
                                 rule=rule, recall=(q & (ys == 1)).sum() / max(ys.sum(), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "os_models_wf.csv"), index=False)


if __name__ == "__main__":
    main()

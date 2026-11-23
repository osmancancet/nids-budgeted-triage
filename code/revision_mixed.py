"""Mixed held-out-family streams (Protocol OS, LightGBM, five seeds; reuses the stored scores).

Each test stream holds all benign test flows, the held-out family at prevalence PI_U and the
known attack families of the same dataset at prevalence PI_K, so the unseen family competes with
known attacks for places in the queue. The queue is the top b of the stream (as in Table 5) and,
separately, the warm-up quantile of the first 20% (evaluated on the rest).

  python revision_mixed.py   -> results/revision/os_mixed.csv
"""
import os
import glob
import numpy as np
import pandas as pd

from revision_analysis import Ctx, priorities, topk_mask, DUMP, OUT

SETTINGS = [(0.002, 0.008), (0.005, 0.005)]   # (unseen prevalence, known prevalence)
BUDGETS = [0.01, 0.02]
RULES = ["detector", "fisher", "wfisher0.7"]


def stream(y, unseen, pi_u, pi_k, rng):
    ben = np.flatnonzero(y == 0)
    u = np.flatnonzero(unseen)
    k = np.flatnonzero((y == 1) & ~unseen)
    nb = len(ben)
    n_u = min(len(u), int(round(pi_u / (1 - pi_u - pi_k) * nb)))
    n_k = min(len(k), int(round(pi_k / (1 - pi_u - pi_k) * nb)))
    idx = np.concatenate([ben, rng.choice(u, n_u, replace=False), rng.choice(k, n_k, replace=False)])
    return rng.permutation(idx)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(DUMP, "os", "*__LGBM__s*.npz"))):
        z = np.load(f, allow_pickle=False)
        seed = int(os.path.basename(f)[:-4].split("__s")[-1])
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        y, un = z["y"].astype(int), z["unseen"]
        if un.sum() < 300 or ((y == 1) & ~un).sum() == 0:
            continue
        for pi_u, pi_k in SETTINGS:
            idx = stream(y, un, pi_u, pi_k, np.random.default_rng(seed + 31))
            pr, _, _ = priorities(ctx, z["det"][idx], z["nov"][idx], extra=True)
            yy, uu = y[idx], un[idx]
            w = int(0.2 * len(idx))
            for c in BUDGETS:
                for rule in RULES:
                    s = pr[rule]
                    for mode in ["topk", "warmup"]:
                        if mode == "topk":
                            q, ys, us = topk_mask(s, c), yy, uu
                        else:
                            t = np.quantile(s[:w], 1 - c)
                            q, ys, us = s[w:] >= t, yy[w:], uu[w:]
                        ku = (ys == 1) & ~us
                        rows.append(dict(dataset=str(z["dataset"]), held_out=str(z["held_out"]), seed=seed,
                                         pi_u=pi_u, pi_k=pi_k, budget=c, rule=rule, mode=mode,
                                         recall_unseen=(q & us).sum() / max(us.sum(), 1),
                                         recall_known=(q & ku).sum() / max(ku.sum(), 1),
                                         recall_all=(q & (ys == 1)).sum() / max((ys == 1).sum(), 1),
                                         precision=(q & (ys == 1)).sum() / max(q.sum(), 1),
                                         load=q.mean() / c))
        print(os.path.basename(f), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "os_mixed.csv"), index=False)


if __name__ == "__main__":
    main()

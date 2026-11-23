"""How much of the 1% queue is decided by the tie-breaker (Protocol OS, LightGBM, five seeds).

Conformal p-values take at most m + 1 values, so flows above every benign calibration score share the
floor p = 1/(m + 1). The priorities break such ties with a negligible multiple of the raw scores. For
the detector rule and weighted Fisher (w = 0.7) this script reports, in the stream of Section 3.8:

  floor     share of queued flows at the detector p-value floor
  tied      share of the queue filled from the tie group that straddles the queue boundary
  recall    recall on the unseen family with raw-score tie-breaking (as in the paper) and with
            uniformly random tie-breaking

  python revision_ties.py   -> results/revision/os_ties.csv
"""
import os
import glob
import numpy as np
import pandas as pd

from revision_analysis import Ctx, make_stream, DUMP, OUT

C, W = 0.01, 0.7


def queue(base, tie, k):
    order = np.lexsort((-tie, -base))[:k]
    q = np.zeros(len(base), bool)
    q[order] = True
    edge = base[order[-1]]
    tied = (q & (base == edge)).sum() if (~q & (base == edge)).any() else 0
    return q, tied / k


def main():
    rows = []
    for fpath in sorted(glob.glob(os.path.join(DUMP, "os", "*__LGBM__s*.npz"))):
        z = np.load(fpath, allow_pickle=False)
        if "held_out" not in z.files:
            continue
        seed = int(os.path.basename(fpath)[:-4].split("__")[-1][1:])
        yt, un = z["y"].astype(int), z["unseen"]
        idx = make_stream(yt, 0.01, np.random.default_rng(seed + 11), attack_mask=un)
        ys = yt[idx]
        det, nov = z["det"][idx].astype(np.float64), z["nov"][idx].astype(np.float64)
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        pdt, pnv = ctx.p(det, nov)
        ld, ln = -np.log(pdt), -np.log(pnv)
        k = int(np.floor(C * len(ys)))
        rnd = np.random.default_rng(seed + 13).random(len(ys))
        floor = 1.0 / (len(ctx.cd) + 1)
        for rule, base, tie in [("detector", ld, det), ("wfisher0.7", W * ld + (1 - W) * ln, det + 1e-3 * nov)]:
            q, tied = queue(base, tie, k)
            qr, _ = queue(base, rnd, k)
            rows.append(dict(dataset=os.path.basename(fpath).split("__")[0], held_out=str(z["held_out"]),
                             seed=seed, rule=rule, floor=(q & (pdt <= floor * (1 + 1e-9))).sum() / k,
                             tied=tied, recall=(q & (ys == 1)).sum() / max(ys.sum(), 1),
                             recall_random_tie=(qr & (ys == 1)).sum() / max(ys.sum(), 1)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "os_ties.csv"), index=False)
    d = pd.DataFrame(rows)
    print(d.groupby("rule")[["floor", "tied", "recall", "recall_random_tie"]].mean())


if __name__ == "__main__":
    main()

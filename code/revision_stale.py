"""Staleness of the benign novelty reference on Protocol T1 (CSE-CIC-IDS2018, benign flows in time order).

The detector, the calibration split and the chronological test stream are those of Protocol T1 in
revision_extra.py (benign: first 50% of capture time for training, next 20% for calibration, last 30% for
testing). Only the kNN novelty reference changes. Two references of equal size are drawn from the benign
training flows:

  old      the earliest quarter of the benign training period
  recent   the latest quarter of the benign training period

Each reference gets its own conformal calibration on the same calibration flows. Reported per seed: the
share of benign test flows with novelty p-value <= 0.01 (1% expected without drift), the novelty AUROC on
the test flows, and recall in a 1% queue (exact top-k) for the novelty, Fisher and weighted Fisher rankings
on the chronological test stream at 1% attack prevalence.

  python revision_stale.py   -> results/revision/t1_staleness.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import run_experiments as R
from nidslib import LogScaler
from prepare import CIC_FEATURES
from revision_analysis import Ctx, priorities, topk_mask, pvals

OUT = os.path.join(R.ROOT, "results", "revision")
SEEDS = [0, 1, 2, 3, 4]
N_REF = 10000
K = 5


def t1_data(seed):
    """Same construction as Protocol T1 in revision_extra.run_temporal."""
    df = pd.read_parquet(os.path.join(R.PROC, "CSE-CIC-IDS2018-temporal.parquet"))
    rng = np.random.default_rng(seed)
    parts = []
    for fam_, g in df.groupby("family"):
        if fam_ != "Benign" and len(g) > 30000:
            g = g.iloc[np.sort(rng.choice(len(g), 30000, replace=False))]
        parts.append(g)
    d = pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)
    if (d.label == 0).sum() > 200000:
        bi = np.flatnonzero(d.label.to_numpy() == 0)
        d = d.drop(index=rng.choice(bi, len(bi) - 200000, replace=False)).reset_index(drop=True)
    feats = CIC_FEATURES + ["Protocol"]
    X, y, ts = d[feats].to_numpy(np.float32), d.label.to_numpy(), d.ts.to_numpy()
    part = np.empty(len(d), np.int8)
    ben = np.flatnonzero(y == 0)
    a, b = int(0.5 * len(ben)), int(0.7 * len(ben))
    part[ben[:a]], part[ben[a:b]], part[ben[b:]] = 0, 1, 2
    for _, idx in d[d.label == 1].groupby("family").indices.items():
        idx = rng.permutation(d.index[d.label == 1].to_numpy()[idx])
        a_, b_ = int(round(.5 * len(idx))), int(round(.7 * len(idx)))
        part[idx[:a_]], part[idx[a_:b_]], part[idx[b_:]] = 0, 1, 2
    return X, y, ts, part, rng


def main():
    rows = []
    for seed in SEEDS:
        X, y, ts, part, rng = t1_data(seed)
        tr, ca = part == 0, part == 1
        F = R.Fitted("LGBM", seed, X[tr], y[tr], X[ca], y[ca])
        te = np.flatnonzero(part == 2)
        te = te[np.argsort(ts[te], kind="stable")]
        det_te = F.det.score(X[te])
        det_ca = F.det.score(X[ca])
        yt, yc = y[te], y[ca]
        ben_tr = np.flatnonzero(tr & (y == 0))          # in time order
        q = len(ben_tr) // 4
        sc = LogScaler().fit(X[ben_tr])
        # chronological test stream at 1% prevalence, as in Protocol T1
        ben = np.flatnonzero(yt == 0)
        att = np.flatnonzero(yt == 1)
        n_att = min(len(att), int(round(0.01 / 0.99 * len(ben))))
        sel = np.sort(np.r_[ben, rng.choice(att, n_att, replace=False)])
        ys = yt[sel]
        for name, pool in [("old", ben_tr[:q]), ("recent", ben_tr[-q:])]:
            r_ = np.random.default_rng(seed)
            ref = r_.choice(pool, min(N_REF, len(pool)), replace=False)
            nn = NearestNeighbors(n_neighbors=K).fit(sc.transform(X[ref]))
            nov_te = nn.kneighbors(sc.transform(X[te]))[0].mean(1)
            nov_ca = nn.kneighbors(sc.transform(X[ca]))[0].mean(1)
            ctx = Ctx(det_ca, nov_ca, yc)
            p_nov = pvals(ctx.cn, nov_te)
            pr, _, _ = priorities(ctx, det_te[sel], nov_te[sel], extra=True)
            r = dict(seed=seed, reference=name, n_ref=len(ref),
                     ref_age_days=float((np.median(ts[te]) - np.median(ts[pool])) / 86400.0),
                     benign_pnov_le_001=float((p_nov[yt == 0] <= 0.01).mean()),
                     auroc_nov=roc_auc_score(yt, nov_te))
            for k in ["novelty", "detector", "fisher", "wfisher0.7"]:
                m = topk_mask(pr[k], 0.01)
                r[f"rec_{k}"] = float(m[ys == 1].mean())
            rows.append(r)
            print(r, flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "t1_staleness.csv"), index=False)


if __name__ == "__main__":
    main()

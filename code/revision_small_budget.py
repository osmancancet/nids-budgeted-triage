"""Small review budgets on the chronological CSE-CIC-IDS2018 streams (Protocols T1 and T2).

Same streams, seeds, detector and priorities as run_temporal in revision_extra.py (LightGBM,
1% attack prevalence, time order kept, first 20% of the stream as warm-up window). Budgets of
0.05%, 0.1%, 0.2% and, for reference, 1% of flows; warm-up quantile and online update
(step 0.02 times the spread between the median and the 99.9th percentile of the warm-up
priorities, the step of the main experiments). For each run the number of warm-up flows above
the threshold (r = budget x warm-up size) is recorded, since Proposition 1 makes the load
variance depend on it.

  python revision_small_budget.py   -> results/revision/extra_small_budget.csv
"""
import os
import numpy as np
import pandas as pd

import run_experiments as R
from revision_extra import fit_eval, SEEDS, OUT
from revision_analysis import priorities, online_quantile
from prepare import CIC_FEATURES

BUDGETS = [0.0005, 0.001, 0.002, 0.01]
PRIOS = ["detector", "fisher", "wfisher0.7"]


def main():
    df = pd.read_parquet(os.path.join(R.PROC, "CSE-CIC-IDS2018-temporal.parquet"))
    feats = CIC_FEATURES + ["Protocol"]
    days = sorted(df.day.unique())
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        parts = []
        for fam_, g in df.groupby("family"):
            if fam_ != "Benign" and len(g) > 30000:
                g = g.iloc[np.sort(rng.choice(len(g), 30000, replace=False))]
            parts.append(g)
        d = pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)
        if (d.label == 0).sum() > 200000:
            bi = np.flatnonzero(d.label.to_numpy() == 0)
            dropb = rng.choice(bi, len(bi) - 200000, replace=False)
            d = d.drop(index=dropb).reset_index(drop=True)
        X, y, ts = d[feats].to_numpy(np.float32), d.label.to_numpy(), d.ts.to_numpy()
        for proto in ["T1-benign-time", "T1-random", "T2-day", "T2-random"]:
            # identical partition and random draws to run_temporal (the random variants are
            # computed so that the generator state matches; only T1 and T2 are reported)
            part = np.empty(len(d), np.int8)
            if proto.startswith("T1"):
                ben = np.flatnonzero(y == 0)
                if proto == "T1-random":
                    ben = rng.permutation(ben)
                a, b = int(0.5 * len(ben)), int(0.7 * len(ben))
                part[ben[:a]], part[ben[a:b]], part[ben[b:]] = 0, 1, 2
                for _, idx in d[d.label == 1].groupby("family").indices.items():
                    idx = rng.permutation(d.index[d.label == 1].to_numpy()[idx])
                    a, b = int(round(.5 * len(idx))), int(round(.7 * len(idx)))
                    part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
            else:
                early = np.isin(d.day.to_numpy(), days[:5])
                if proto == "T2-random":
                    early = rng.random(len(d)) < early.mean()
                tr_pool = rng.permutation(np.flatnonzero(early))
                k = int(0.7 * len(tr_pool))
                part[tr_pool[:k]], part[tr_pool[k:]] = 0, 1
                part[~early] = 2
            te = np.flatnonzero(part == 2)
            te = te[np.argsort(ts[te], kind="stable")]
            yt = y[te]
            ben = np.flatnonzero(yt == 0)
            att = np.flatnonzero(yt == 1)
            n_att = min(len(att), int(round(0.01 / 0.99 * len(ben))))
            sel = np.sort(np.r_[ben, rng.choice(att, n_att, replace=False)])
            ys = yt[sel]
            if proto.endswith("random"):
                continue  # drawn only to keep the generator in step with run_temporal
            F, ctx = fit_eval("LGBM", seed, X[part == 0], y[part == 0], X[part == 1], y[part == 1])
            det, nov = F.scores(X[te])
            pr, _, _ = priorities(ctx, det[sel], nov[sel], extra=True)
            for c in BUDGETS:
                for name in PRIOS:
                    s = pr[name]
                    w = int(0.2 * len(s))
                    rest = slice(w, len(s))
                    t_w = np.quantile(s[:w], 1 - c)
                    gam = 0.02 * (np.quantile(s[:w], .999) - np.quantile(s[:w], .5))
                    fl_on = online_quantile(s, c, t_w, gam)
                    for rule, q in [("warmup", s[rest] >= t_w), ("online", fl_on[rest])]:
                        chunks = np.array_split(np.arange(len(q)), 10)
                        loads = [q[ch].mean() / c for ch in chunks]
                        rows.append(dict(seed=seed, protocol=proto, budget=c, priority=name, rule=rule,
                                         n_stream=len(s), n_warm=w, r_warm=c * w,
                                         queue=int(q.sum()), planned=c * len(q),
                                         load=q.mean() / c, load_min10=min(loads), load_max10=max(loads),
                                         recall=q[ys[rest] == 1].mean(),
                                         precision=(q & (ys[rest] == 1)).sum() / max(q.sum(), 1)))
            print(seed, proto, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_small_budget.csv"), index=False)


if __name__ == "__main__":
    main()

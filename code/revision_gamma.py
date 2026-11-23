"""Step-size sensitivity of the online quantile update on the chronological streams (CSE-CIC-IDS2018,
Protocols T1 and T2 of revision_extra.run_temporal, LightGBM, Fisher priority, 1% budget, five seeds).

The step is gamma = g * (q99.9 - q50) of the warm-up priorities with g in G; the paper uses g = 0.02.
A batched variant updates the threshold once per batch of 4,096 flows with the same total step.

  python revision_gamma.py   -> results/revision/extra_gamma.csv
"""
import os
import numpy as np
import pandas as pd

import run_experiments as R
from revision_analysis import priorities, make_stream, online_quantile
from revision_extra import fit_eval, OUT, SEEDS
from prepare import CIC_FEATURES

G = [0.005, 0.01, 0.02, 0.05, 0.1]
C = 0.01
B = 4096


def batched(s, c, t, gamma, B):
    f = np.zeros(len(s), bool)
    for i in range(0, len(s), B):
        f[i:i + B] = s[i:i + B] >= t
        t += gamma * (f[i:i + B].sum() - c * len(f[i:i + B]))
    return f


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
            d = d.drop(index=rng.choice(bi, len(bi) - 200000, replace=False)).reset_index(drop=True)
        X, y, ts = d[feats].to_numpy(np.float32), d.label.to_numpy(), d.ts.to_numpy()
        for proto in ["T1-benign-time", "T2-day"]:
            part = np.empty(len(d), np.int8)
            if proto.startswith("T1"):
                ben = np.flatnonzero(y == 0)
                a, b = int(0.5 * len(ben)), int(0.7 * len(ben))
                part[ben[:a]], part[ben[a:b]], part[ben[b:]] = 0, 1, 2
                for _, idx in d[d.label == 1].groupby("family").indices.items():
                    idx = rng.permutation(d.index[d.label == 1].to_numpy()[idx])
                    a, b = int(round(.5 * len(idx))), int(round(.7 * len(idx)))
                    part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
            else:
                early = np.isin(d.day.to_numpy(), days[:5])
                tr_pool = rng.permutation(np.flatnonzero(early))
                k = int(0.7 * len(tr_pool))
                part[tr_pool[:k]], part[tr_pool[k:]] = 0, 1
                part[~early] = 2
            F, ctx = fit_eval("LGBM", seed, X[part == 0], y[part == 0], X[part == 1], y[part == 1])
            te = np.flatnonzero(part == 2)
            te = te[np.argsort(ts[te], kind="stable")]
            det, nov = F.scores(X[te])
            yt = y[te]
            ben, att = np.flatnonzero(yt == 0), np.flatnonzero(yt == 1)
            n_att = min(len(att), int(round(0.01 / 0.99 * len(ben))))
            sel = np.sort(np.r_[ben, rng.choice(att, n_att, replace=False)])
            ys = yt[sel]
            s = priorities(ctx, det[sel], nov[sel])[0]["fisher"]
            w = int(0.2 * len(s))
            t_w = np.quantile(s[:w], 1 - C)
            spread = np.quantile(s[:w], .999) - np.quantile(s[:w], .5)
            runs = [(f"g{g}", online_quantile(s, C, t_w, g * spread)) for g in G]
            runs.append(("batched_g0.02", batched(s, C, t_w, 0.02 * spread, B)))
            for name, fl in runs:
                q = fl[w:]
                loads = [q[ch].mean() / C for ch in np.array_split(np.arange(len(q)), 10)]
                rows.append(dict(seed=seed, protocol=proto, rule=name, load=q.mean() / C,
                                 load_min10=min(loads), load_max10=max(loads),
                                 recall=q[ys[w:] == 1].mean(),
                                 precision=(q & (ys[w:] == 1)).sum() / max(q.sum(), 1)))
            print(seed, proto, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_gamma.csv"), index=False)


if __name__ == "__main__":
    main()

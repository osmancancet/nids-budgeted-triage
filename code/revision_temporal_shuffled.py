"""Shuffled-order control for the chronological protocols T1 and T2 (CSE-CIC-IDS2018, LightGBM).

The training, calibration and test flows are exactly those of Protocols T1 and T2 in revision_extra.py;
only the order of the 1%-prevalence test stream is randomised, so that Assumption T holds by construction
while the shift between source and target is kept. Rules and outputs as in extra_temporal_stream.csv.

  python revision_temporal_shuffled.py   -> results/revision/extra_temporal_shuffled.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_experiments as R
from revision_analysis import priorities, make_stream, online_quantile, rolling_quantile_flags, BUDGETS, ALPHA
from revision_extra import fit_eval, OUT, SEEDS
from prepare import CIC_FEATURES


def run_temporal_shuffled():
    """Two chronological protocols on CSE-CIC-IDS2018 (benign Bernoulli 5% sample, all attacks).

    T1 (benign drift): benign flows are split by time (first 50% train, next 20% calibration,
       last 30% test); attack flows are split at random by family as in the main protocol,
       capped at 30,000 per family. The test stream is kept in time order.
    T2 (day split): train and calibrate on the first five capture days, test on the last five.
       Every test attack family is absent from training, and benign traffic is later in time.
    For both, the random-split counterpart on the same flows is reported as reference.
    """
    df = pd.read_parquet(os.path.join(R.PROC, "CSE-CIC-IDS2018-temporal.parquet"))
    feats = CIC_FEATURES + ["Protocol"]
    rows, ns = [], []
    days = sorted(df.day.unique())
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # cap attack families as in the main protocol
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
        X, y, fam, ts = d[feats].to_numpy(np.float32), d.label.to_numpy(), d.family.to_numpy(), d.ts.to_numpy()
        for proto in ["T1-benign-time", "T2-day"]:
            part = np.empty(len(d), np.int8)
            if proto.startswith("T1"):
                ben = np.flatnonzero(y == 0)  # already in time order
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
                tr_pool = np.flatnonzero(early)
                tr_pool = rng.permutation(tr_pool)
                k = int(0.7 * len(tr_pool))
                part[tr_pool[:k]], part[tr_pool[k:]] = 0, 1
                part[~early] = 2
            F, ctx = fit_eval("LGBM", seed, X[part == 0], y[part == 0], X[part == 1], y[part == 1])
            te = np.flatnonzero(part == 2)
            te = te[np.argsort(ts[te], kind="stable")]  # chronological test stream
            det, nov = F.scores(X[te])
            yt = y[te]
            p_iso = ctx.iso.predict(det)
            from nidslib import ece
            r = dict(seed=seed, protocol=proto, n_test=len(te), attack_share=yt.mean(),
                     auroc=roc_auc_score(yt, det), ece_iso=ece(p_iso, yt),
                     fpr_inflation=(det[yt == 0] >= ctx.tau).mean() / ALPHA,
                     tpr=(det[yt == 1] >= ctx.tau).mean(),
                     auroc_nov=roc_auc_score(yt, nov))
            rows.append(r)
            # chronological stream at 1% prevalence: keep time order of all flows
            ben = np.flatnonzero(yt == 0)
            att = np.flatnonzero(yt == 1)
            n_att = min(len(att), int(round(0.01 / 0.99 * len(ben))))
            sel = np.sort(np.r_[ben, rng.choice(att, n_att, replace=False)])
            sel = np.random.default_rng(seed + 500).permutation(sel)  # same flows, random order
            ys = yt[sel]
            pr, _, _ = priorities(ctx, det[sel], nov[sel])
            src_thr = None
            for c in BUDGETS:
                for name in ["detector", "fisher"]:
                    s = pr[name]
                    w = int(0.2 * len(s))
                    rest = slice(w, len(s))
                    # source threshold from a source calibration stream (as in the main protocol)
                    rngc = np.random.default_rng(seed + 1000)
                    cidx = make_stream(ctx.cal_y, 0.01, rngc)
                    cpr, _, _ = priorities(ctx, ctx.cal_det[cidx], ctx.cal_nov[cidx])
                    t_src = np.quantile(cpr[name], 1 - c)
                    t_w = np.quantile(s[:w], 1 - c)
                    fl_on = online_quantile(s, c, t_w, 0.02 * (np.quantile(s[:w], .999) - np.quantile(s[:w], .5)))
                    fl_ro, _ = rolling_quantile_flags(s, c, window=w, step=500)
                    for rule, q in [("source", s[rest] >= t_src), ("warmup", s[rest] >= t_w),
                                    ("online", fl_on[rest]), ("rolling", fl_ro[rest])]:
                        # load per tenth of the evaluation segment (within-stream drift)
                        chunks = np.array_split(np.arange(len(q)), 10)
                        loads = [q[ch].mean() / c for ch in chunks]
                        ns.append(dict(seed=seed, protocol=proto, budget=c, priority=name, rule=rule,
                                       load=q.mean() / c, load_min10=min(loads), load_max10=max(loads),
                                       recall=q[ys[rest] == 1].mean() if (ys[rest] == 1).any() else np.nan,
                                       precision=(q & (ys[rest] == 1)).sum() / max(q.sum(), 1)))
            print("shuffled", seed, proto, flush=True)
            pd.DataFrame(ns).to_csv(os.path.join(OUT, "extra_temporal_shuffled.csv"), index=False)



if __name__ == "__main__":
    run_temporal_shuffled()

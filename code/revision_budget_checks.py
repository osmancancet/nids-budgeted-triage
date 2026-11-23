"""Checks of the budget rules on the stored LightGBM scores.

  ties     warm-up rule with and without random tie-breaking (priority + U(0, 1e-9) jitter, which
           makes priorities distinct and restores the conditions of Proposition 1), and the share of
           runs whose warm-up threshold falls inside a group of tied priorities
           (Protocols OS and XD, 20% window, as in revision_analysis)
  bound    value of bound (7), (R + gamma) / (gamma * T * b), for the online update on the same streams,
           with gamma = 0.02 (q99.9 - q50) of the warm-up priorities and R the range of the priorities
  band     coverage of the exact predictive 80% band of the load (Beta(r + 1, m - r) mixed with a
           binomial over the evaluated flows, Proposition 1(ii)) against the normal band of Eq. (5),
           over the warm-up windows of results/revision/os_warmup.csv
  decoy    budget exhaustion: after the warm-up window of an in-domain CIC stream (1% attacks), benign
           flows of another CICFlowMeter dataset, scored by the same source model, are inserted at
           random positions as decoys (0.5%, 1% and 2% of the evaluated stream); recall on the real
           attacks and the decoy share of the queue at a 1% budget, warm-up and online thresholds

  python revision_budget_checks.py   -> results/revision/budget_ties.csv, budget_band.csv, budget_decoy.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import beta as Beta, binom, norm

import run_experiments as R
from revision_analysis import Ctx, priorities, make_stream, online_quantile, load_os, load_xd, OUT

C = 0.01
WF = 0.2
RULES = ["detector", "fisher", "wfisher0.7"]


def tie_rows(key, s, ys, rng):
    n = len(ys)
    w = int(WF * n)
    ev = slice(n // 2, n)
    out = []
    for r in RULES:
        x = s[r]
        t = np.quantile(x[:w], 1 - C)
        q = x[ev] >= t
        tied = np.sum(x[:w] == t) > 1
        j = x + rng.uniform(0, 1e-9, n)
        tj = np.quantile(j[:w], 1 - C)
        qj = j[ev] >= tj
        g = 0.02 * (np.quantile(x[:w], .999) - np.quantile(x[:w], .5))
        T = n - w
        rng_s = float(x.max() - x.min())
        out.append(dict(key, priority=r, load=q.mean() / C, load_jitter=qj.mean() / C, tie_at_threshold=bool(tied),
                        tie_share_eval=np.mean(x[ev] == t), gamma=g, range=rng_s, T=T,
                        bound_rel=(rng_s + g) / (g * T * C) if g > 0 else np.nan))
    return out


def main():
    rng = np.random.default_rng(2026)
    rows = []
    os.environ["MODELS"] = "LGBM"
    for key, z in load_os():
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        y = z["y"]
        idx = make_stream(y, 0.01, np.random.default_rng(key["seed"] + 11), attack_mask=z["unseen"])
        pr, _, _ = priorities(ctx, z["det"][idx].astype(np.float64), z["nov"][idx].astype(np.float64), extra=True)
        rows += tie_rows(dict(key, regime="Unseen family"), pr, y[idx], rng)
    dec = []
    for key, z in load_xd():
        if key["model"] != "LGBM":
            continue
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        tg = [str(t) for t in z["targets"]]
        for tgt in tg:
            y = z[f"{tgt}__y"]
            idx = make_stream(y, 0.01, np.random.default_rng(key["seed"] + 7))
            det, nov = z[f"{tgt}__det"].astype(np.float64), z[f"{tgt}__nov"].astype(np.float64)
            pr, _, _ = priorities(ctx, det[idx], nov[idx], extra=True)
            reg = "In-domain" if tgt == key["source"] else "Cross-dataset"
            rows += tie_rows(dict(key, target=tgt, regime=reg), pr, y[idx], rng)
            # decoy flooding on in-domain CIC streams
            if tgt != key["source"] or key["source"] not in R.CIC:
                continue
            others = [o for o in tg if o != tgt and o in R.CIC]
            for frac in [0.005, 0.01, 0.02]:
                r2 = np.random.default_rng(key["seed"] + 41)
                n = len(idx)
                w = int(WF * n)
                n_dec = int(round(frac * (n - w)))
                pd_det = np.concatenate([z[f"{o}__det"][z[f"{o}__y"] == 0] for o in others]).astype(np.float64)
                pd_nov = np.concatenate([z[f"{o}__nov"][z[f"{o}__y"] == 0] for o in others]).astype(np.float64)
                pick = r2.choice(len(pd_det), n_dec, replace=False)
                d_det, d_nov = pd_det[pick], pd_nov[pick]
                pos = np.sort(r2.choice(n - w + n_dec, n_dec, replace=False))
                is_dec = np.zeros(n - w + n_dec, bool)
                is_dec[pos] = True
                ev_det, ev_nov = np.empty(len(is_dec)), np.empty(len(is_dec))
                ev_det[is_dec], ev_nov[is_dec] = d_det, d_nov
                ev_det[~is_dec], ev_nov[~is_dec] = det[idx][w:], nov[idx][w:]
                yy = np.zeros(len(is_dec), int)
                yy[~is_dec] = y[idx][w:]
                s_all, _, _ = priorities(ctx, np.r_[det[idx][:w], ev_det], np.r_[nov[idx][:w], ev_nov], extra=True)
                for r in RULES:
                    s = s_all[r]
                    t = np.quantile(s[:w], 1 - C)
                    g = 0.02 * (np.quantile(s[:w], .999) - np.quantile(s[:w], .5))
                    for rule, q in [("warmup", s[w:] >= t), ("online", online_quantile(s, C, t, g)[w:])]:
                        dec.append(dict(key, decoy_frac=frac, priority=r, rule=rule,
                                        recall=q[yy == 1].mean(), load=q.mean() / C,
                                        decoy_share=(q & is_dec).sum() / max(q.sum(), 1),
                                        decoys_flagged=(q & is_dec).sum() / max(is_dec.sum(), 1)))
                # same stream without decoys, for reference
                if frac == 0.005:
                    for r in RULES:
                        s = pr[r]
                        t = np.quantile(s[:w], 1 - C)
                        g = 0.02 * (np.quantile(s[:w], .999) - np.quantile(s[:w], .5))
                        for rule, q in [("warmup", s[w:] >= t), ("online", online_quantile(s, C, t, g)[w:])]:
                            dec.append(dict(key, decoy_frac=0.0, priority=r, rule=rule, recall=q[y[idx][w:] == 1].mean(),
                                            load=q.mean() / C, decoy_share=0.0, decoys_flagged=np.nan))
            print(key, tgt, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "budget_ties.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(OUT, "budget_decoy.csv"), index=False)

    # exact predictive band of the load (Proposition 1(ii)) vs the normal band of Eq. (5)
    W = pd.read_csv(os.path.join(OUT, "os_warmup.csv"))
    W = W[(W.rule == "warmup") & (W.model == "LGBM")]
    band = []
    rs = np.random.default_rng(1)
    z90 = norm.ppf(0.9)
    for _, g in W.iterrows():
        m, n, c = int(g.n_warm), int(g.n_eval), g.budget
        if m < 2:
            continue
        r = min(int(np.floor(c * (m + 1))), m - 1)     # rank of the warm-up threshold from the top
        P = Beta(r + 1, m - r).rvs(size=4000, random_state=rs)
        L = rs.binomial(n, P) / (c * n)
        lo, hi = np.quantile(L, [0.1, 0.9])
        sd = np.sqrt((1 - c) / c * (1 / m + 1 / n))
        band.append(dict(budget=c, warmup=g.warmup, m=m, n=n, load=g.load, in_exact=lo <= g.load <= hi,
                         in_normal=1 - z90 * sd <= g.load <= 1 + z90 * sd))
    band = pd.DataFrame(band).groupby(["budget", "warmup"]).agg(
        runs=("load", "size"), m_median=("m", "median"), cover_exact=("in_exact", "mean"),
        cover_normal=("in_normal", "mean")).reset_index()
    pd.DataFrame(band).to_csv(os.path.join(OUT, "budget_band.csv"), index=False)
    print(pd.DataFrame(band).round(3).to_string())


if __name__ == "__main__":
    main()

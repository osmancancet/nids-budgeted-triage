"""Correlation alignment (CORAL) as a label-free domain-adaptation baseline for Protocol XD-A
(six cross-dataset pairs, LightGBM, five seeds, the target streams of revision_adapt.py).

In the signed-log standardised space of the source training split, the source features are
whitened with the source covariance and re-coloured with the covariance (and mean) of the
unlabeled target warm-up window (first 20% of the stream); both covariances get an identity
ridge, as in the original method. The detector is retrained on the aligned source flows and
scores the target stream in the same space; the kNN novelty reference is the aligned benign
source training flows. The queue follows the warm-up quantile rule of XD-A at each budget.

  python revision_coral.py   -> results/revision/xd_coral.csv
"""
import os
import numpy as np
import pandas as pd

import run_experiments as R
from nidslib import KNNNovelty, LogScaler, make_model
from revision_analysis import pvals, make_stream, OUT
from revision_adapt import BUDGETS, WARM, wfisher, evaluate, warm_rule


def mpow(C, p):
    w, V = np.linalg.eigh(C)
    return (V * np.clip(w, 1e-12, None) ** p) @ V.T


def coral(Zs, Zt_w):
    Cs = np.cov(Zs, rowvar=False) + np.eye(Zs.shape[1])
    Ct = np.cov(Zt_w, rowvar=False) + np.eye(Zs.shape[1])
    A = mpow(Cs, -0.5) @ mpow(Ct, 0.5)
    mu_s, mu_t = Zs.mean(0), Zt_w.mean(0)
    return lambda Z: (Z - mu_s) @ A + mu_t


def main():
    data = {n: R.load(n) for n in R.CIC}
    feats = data[R.CIC[0]][1]
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    rows = []
    for seed in seeds:
        parts = {n: R.split(data[n][0], seed) for n in R.CIC}
        for src in R.CIC:
            df = data[src][0]
            Xa, ya, ps = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
            Xs, ys, Xc, yc = Xa[ps == 0], ya[ps == 0], Xa[ps == 1], ya[ps == 1]
            sc = LogScaler().fit(Xs)
            Zs, Zc = sc.transform(Xs), sc.transform(Xc)
            for tgt in [t for t in R.CIC if t != src]:
                tdf = data[tgt][0]
                tm = parts[tgt] == 2
                Xt0, yt0 = tdf.loc[tm, feats].to_numpy(np.float32), tdf.loc[tm, "label"].to_numpy()
                idx = make_stream(yt0, 0.01, np.random.default_rng(seed + 7))
                Zt, yt = sc.transform(Xt0[idx]), yt0[idx]
                n = len(yt)
                n_w = int(WARM * n)
                f = coral(Zs, Zt[:n_w])
                As, Ac = f(Zs), f(Zc)
                clf = make_model("LGBM", seed).fit(As, ys)
                det, det_c = clf.predict_proba(Zt)[:, 1], clf.predict_proba(Ac)[:, 1]
                nov_m = KNNNovelty(seed=seed).fit(As[ys == 0])
                nov, nov_c = nov_m.score(Zt), nov_m.score(Ac)
                pr = {"CORAL_detector": det,
                      "CORAL_wfisher": wfisher(pvals(det_c[yc == 0], det), pvals(nov_c[yc == 0], nov)) + 1e-9 * det}
                key = dict(source=src, target=tgt, seed=seed, n=n, n_warm=n_w)
                for c in BUDGETS:
                    for name, s in pr.items():
                        q = warm_rule(s, n_w, c)
                        rows.append(dict(key, budget=c, variant=name, load=q.mean() / c, **evaluate(q, yt[n_w:])))
                print(seed, src, tgt, "recall@1%% %.3f" % rows[-6]["recall"], flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "xd_coral.csv"), index=False)


if __name__ == "__main__":
    main()

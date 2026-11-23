"""Alternative novelty scores inside the same conformal wrapper (Protocol OS, LightGBM, five seeds).

The benign training flows do not depend on which family is held out, so each novelty scorer is
fitted once per dataset and seed on the benign training split (signed-log standardised features)
and scores the calibration and test splits. The stored LightGBM scores of each held-out-family run
are reused, and the stream of Section 3.8 (seed + 11) is rebuilt, so only the novelty score changes.

  knn     mean distance to the 5 nearest of 20,000 benign flows (the score used in the paper)
  iforest isolation forest, 200 trees, 256 samples per tree
  lof     local outlier factor (20 neighbours, novelty mode, 20,000 benign flows)
  maha    Mahalanobis distance under a Ledoit-Wolf covariance of the benign flows

  python revision_novalt.py   -> results/revision/os_novalt.csv
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import run_experiments as R
from nidslib import LogScaler
from revision_analysis import Ctx, priorities, topk_mask, make_stream, DUMP, OUT

BUDGETS = [0.01, 0.02, 0.05]


def scorers(Zb, seed):
    rng = np.random.default_rng(seed)
    sub = Zb[rng.choice(len(Zb), min(20000, len(Zb)), replace=False)]
    out = {}
    f = IsolationForest(n_estimators=200, max_samples=256, random_state=seed, n_jobs=2).fit(Zb)
    out["iforest"] = lambda Z, f=f: -f.score_samples(Z)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=2).fit(sub)
    out["lof"] = lambda Z, lof=lof: -lof.score_samples(Z)
    lw = LedoitWolf().fit(Zb)
    out["maha"] = lambda Z, lw=lw: lw.mahalanobis(Z)
    return out


def main():
    rows = []
    names = os.environ.get("DATASETS", ",".join(R.ALL)).split(",")
    for name in names:
        df, feats = R.load(name)
        X, y, fam = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["family"].to_numpy()
        for seed in range(5):
            files = sorted(glob.glob(os.path.join(DUMP, "os", f"{name}__*__LGBM__s{seed}.npz")))
            if not files:
                continue
            pt = R.split(df, seed)
            tr0 = (pt == 0) & (y == 0)
            sc = LogScaler().fit(X[tr0])
            sco = scorers(sc.transform(X[tr0]), seed)
            ca_all, te = pt == 1, pt == 2
            Zca, Zte = sc.transform(X[ca_all]), sc.transform(X[te])
            alt_ca = {k: f(Zca) for k, f in sco.items()}
            alt_te = {k: f(Zte) for k, f in sco.items()}
            fam_ca = fam[ca_all]
            for fpath in files:
                z = np.load(fpath, allow_pickle=False)
                held = str(z["held_out"])
                keep_ca = fam_ca != held
                assert keep_ca.sum() == len(z["cal_y"]) and te.sum() == len(z["y"])
                yt, un = z["y"].astype(int), z["unseen"]
                idx = make_stream(yt, 0.01, np.random.default_rng(seed + 11), attack_mask=un)
                ys = yt[idx]
                for kind in ["knn"] + list(sco):
                    cn = z["cal_nov"] if kind == "knn" else alt_ca[kind][keep_ca]
                    nv = z["nov"] if kind == "knn" else alt_te[kind]
                    ctx = Ctx(z["cal_det"], cn, z["cal_y"])
                    pr, _, _ = priorities(ctx, z["det"][idx], nv[idx], extra=True)
                    auc = roc_auc_score(ys, nv[idx]) if 0 < ys.sum() < len(ys) else np.nan
                    for c in BUDGETS:
                        for rule in ["detector", "novelty", "fisher", "wfisher0.7"]:
                            q = topk_mask(pr[rule], c)
                            rows.append(dict(dataset=name, held_out=held, seed=seed, novelty=kind, budget=c,
                                             rule=rule, auroc_nov=auc,
                                             recall=(q & (ys == 1)).sum() / max(ys.sum(), 1)))
                print(name, seed, held, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "os_novalt.csv"), index=False)


if __name__ == "__main__":
    main()

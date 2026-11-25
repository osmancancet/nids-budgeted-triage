"""Model-ensemble baselines for unseen families (Protocol OS), from the score dumps of revision_dump.py.

For each held-out family and seed, the four detectors (LightGBM, random forest, MLP, logistic
regression) were trained on the same split, so their scores refer to the same test flows. On the
same 1%-prevalence stream as revision_analysis.run_os, three ensemble priorities are compared at
budgets of 1%, 2% and 5% (exact top-b of the stream) with the LightGBM priorities of the paper:

  ens_mean       mean over the four models of -log p_det (each model's own conformal p-value)
  ens_disagree   standard deviation of the four attack probabilities (ensemble disagreement,
                 the usual ensemble uncertainty score for out-of-distribution inputs)
  ens_mean_dis   ens_mean plus the rank of ens_disagree, scaled to the range of ens_mean

  python revision_ensemble.py   -> results/revision/os_ensemble.csv
"""
import os
import glob
import numpy as np
import pandas as pd

from revision_analysis import DUMP, OUT, BUDGETS, Ctx, make_stream, priorities, topk_mask

MODELS = ["LGBM", "RF", "MLP", "LR"]


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(DUMP, "os", "*__LGBM__s*.npz")))
    for f in files:
        base = os.path.basename(f)[:-4]
        z = {m: np.load(f.replace("__LGBM__", f"__{m}__")) for m in MODELS}
        y, unseen = z["LGBM"]["y"], z["LGBM"]["unseen"]
        if not all(np.array_equal(z[m]["y"], y) and np.array_equal(z[m]["unseen"], unseen) for m in MODELS):
            raise ValueError(f"test flows differ across models: {base}")
        seed = int(base.split("__")[-1][1:])
        rng = np.random.default_rng(seed + 11)
        idx = make_stream(y, 0.01, rng, attack_mask=unseen)
        ys = y[idx]
        ctx = {m: Ctx(z[m]["cal_det"], z[m]["cal_nov"], z[m]["cal_y"]) for m in MODELS}
        det = {m: z[m]["det"][idx].astype(np.float64) for m in MODELS}
        pr, _, _ = priorities(ctx["LGBM"], det["LGBM"], z["LGBM"]["nov"][idx], extra=True)
        ld = np.mean([-np.log(ctx[m].p(det[m], det[m])[0]) for m in MODELS], axis=0)
        dis = np.std(np.vstack([det[m] for m in MODELS]), axis=0)
        rk = pd.Series(dis).rank(pct=True).to_numpy()
        tie = 1e-9 * det["LGBM"]
        cand = {"detector": pr["detector"], "wfisher0.7": pr["wfisher0.7"], "uncertainty": pr["uncertainty"],
                "ens_mean": ld + tie, "ens_disagree": dis + tie,
                "ens_mean_dis": ld + rk * (ld.max() - ld.min()) + tie}
        key = dict(dataset=str(z["LGBM"]["dataset"]), held_out=str(z["LGBM"]["held_out"]), seed=seed)
        for c in BUDGETS:
            for name, s in cand.items():
                q = topk_mask(s, c)
                rows.append(dict(key, budget=c, priority=name, recall=q[ys == 1].mean(),
                                 precision=ys[q].mean() if q.any() else np.nan))
        print(base, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "os_ensemble.csv"), index=False)


if __name__ == "__main__":
    main()

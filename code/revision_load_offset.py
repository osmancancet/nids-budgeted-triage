"""Median load of the warm-up rule on held-out-family streams against the exchangeable prediction.

For each model and priority (1% budget, 20% warm-up window, results/revision/os_warmup.csv):
  median     median load over all runs
  ci         95% bootstrap interval of the median, resampling whole (dataset, seed) streams, because the
             held-out families of one dataset and seed share the benign flows of the stream
  expected   mean load of the same rule (interpolated empirical quantile, S >= t) on exchangeable
             streams with the same m and n, by simulation with continuous i.i.d. priorities
  sd_within / sd_between   spread of the load within and between (dataset, seed) streams

  python revision_load_offset.py   -> results/revision/load_offset.csv
"""
import os
import numpy as np
import pandas as pd

from revision_analysis import OUT


def main():
    w = pd.read_csv(os.path.join(OUT, "os_warmup.csv"))
    g = w[(w.rule == "warmup") & (w.budget == 0.01) & (w.warmup == 0.2)]
    rng = np.random.default_rng(7)
    rows = []
    for (mo, pr), x in g.groupby(["model", "priority"]):
        keys = list(x.groupby(["dataset", "seed"]).groups.values())
        loads = [x.loc[k].load.to_numpy() for k in keys]
        meds = []
        for _ in range(4000):
            pick = rng.integers(0, len(loads), len(loads))
            meds.append(np.median(np.concatenate([loads[i] for i in pick])))
        sim = []
        for m, n in zip(x.n_warm.astype(int), x.n_eval.astype(int)):
            u = rng.random(m)
            sim.append(rng.binomial(n, 1 - np.quantile(u, 0.99)) / (0.01 * n))
        rows.append(dict(model=mo, priority=pr, runs=len(x), streams=len(keys), median=x.load.median(),
                         ci_lo=np.quantile(meds, .025), ci_hi=np.quantile(meds, .975),
                         expected=np.mean(sim), sim_median=np.median(sim),
                         sd_within=x.groupby(["dataset", "seed"]).load.std().mean(),
                         sd_between=x.groupby(["dataset", "seed"]).load.mean().std()))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "load_offset.csv"), index=False)
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()

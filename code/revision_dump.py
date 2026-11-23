"""Score dumps for the revision analyses.

Re-runs the exact training pipeline of run_experiments.py (same splits, seeds,
models, calibration and kNN novelty reference) and stores per-flow scores so
that every revision analysis (warm-up sensitivity, prevalence, priority
functions, queue precision, confident-miss thresholds, rejection baselines)
is computed from the same fitted models without retraining.

  python revision_dump.py xd            # in-domain + cross-dataset, 4 models x 5 seeds
  python revision_dump.py os [models]   # leave-one-family-out, 5 seeds

Output: data/dumps/{xd,os}/<config>.npz (float32 scores, int labels).
"""
import os
import sys
import time
import itertools
import numpy as np

import run_experiments as R

DUMP = os.path.join(R.ROOT, "data", "dumps")


def _save(path, **arrs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrs)


def _fit(model, seed, Xtr, ytr, Xcal, ycal):
    t0 = time.time()
    F = R.Fitted(model, seed, Xtr, ytr, Xcal, ycal)
    fit_s = time.time() - t0
    pc, nc = F.scores(Xcal)
    return F, pc, nc, fit_s


def run_xd(models=R.MODEL_NAMES, seeds=R.SEEDS):
    data = {n: R.load(n) for n in R.ALL}
    for seed in seeds:
        parts = {n: R.split(data[n][0], seed) for n in R.ALL}
        for src in R.ALL:
            df, feats = data[src]
            X, y, pt = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
            targets = [src] + ([t for t in R.CIC if t != src] if src in R.CIC else [])
            for model in models:
                out = os.path.join(DUMP, "xd", f"{src}__{model}__s{seed}.npz")
                if os.path.exists(out):
                    continue
                F, pc, nc, fit_s = _fit(model, seed, X[pt == 0], y[pt == 0], X[pt == 1], y[pt == 1])
                arrs = dict(cal_det=pc.astype(np.float32), cal_nov=nc.astype(np.float32),
                            cal_y=y[pt == 1].astype(np.int8), fit_s=np.float32(fit_s),
                            targets=np.array(targets))
                for tgt in targets:
                    tdf, _ = data[tgt]
                    tm = parts[tgt] == 2
                    Xt = tdf.loc[tm, feats].to_numpy(np.float32)
                    t0 = time.time()
                    det = F.det.score(Xt)
                    t1 = time.time()
                    nov = F.nov.score(Xt)
                    t2 = time.time()
                    arrs[f"{tgt}__det"] = det.astype(np.float32)
                    arrs[f"{tgt}__nov"] = nov.astype(np.float32)
                    arrs[f"{tgt}__y"] = tdf.loc[tm, "label"].to_numpy().astype(np.int8)
                    arrs[f"{tgt}__group"] = tdf.loc[tm, "group"].to_numpy().astype(str)
                    arrs[f"{tgt}__family"] = tdf.loc[tm, "family"].to_numpy().astype(str)
                    arrs[f"{tgt}__t_det"] = np.float32(t1 - t0)
                    arrs[f"{tgt}__t_nov"] = np.float32(t2 - t1)
                _save(out, **arrs)
                print("xd", seed, src, model, "%.0fs" % fit_s, flush=True)


def run_os(models=R.MODEL_NAMES, seeds=R.SEEDS, min_n=300):
    names = os.environ.get("DATASETS", ",".join(R.ALL)).split(",")
    for name in names:
        df, feats = R.load(name)
        X, y, fam = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["family"].to_numpy()
        fams = [f for f, n in df.loc[df.label == 1, "family"].value_counts().items() if n >= min_n]
        for seed in seeds:
            pt = R.split(df, seed)
            for f, model in itertools.product(fams, models):
                safe = "".join(c if c.isalnum() else "_" for c in f)
                out = os.path.join(DUMP, "os", f"{name}__{safe}__{model}__s{seed}.npz")
                if os.path.exists(out):
                    continue
                keep = fam != f
                tr, ca, te = (pt == 0) & keep, (pt == 1) & keep, pt == 2
                F, pc, nc, fit_s = _fit(model, seed, X[tr], y[tr], X[ca], y[ca])
                det, nov = F.scores(X[te])
                _save(out, cal_det=pc.astype(np.float32), cal_nov=nc.astype(np.float32),
                      cal_y=y[ca].astype(np.int8), det=det.astype(np.float32),
                      nov=nov.astype(np.float32), y=y[te].astype(np.int8),
                      unseen=(fam[te] == f), family=fam[te].astype(str),
                      dataset=np.array(name), held_out=np.array(f), fit_s=np.float32(fit_s))
                print("os", name, seed, f, model, "%.0fs" % fit_s, flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else R.MODEL_NAMES
    if mode == "xd":
        run_xd(models)
    else:
        run_os(models)

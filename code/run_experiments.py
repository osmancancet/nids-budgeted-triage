"""Joint deployment-oriented evaluation of ML-based NIDS.

Experiment XD  (in-domain and cross-dataset):  python run_experiments.py xd
Experiment OS  (leave-one-family-out, unseen attack families):  python run_experiments.py os

All outputs go to results/*.csv (one row per configuration and metric set).
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd

from nidslib import (Detector, Isotonic, Platt, KNNNovelty, ConformalTriage, ece, brier,
                     safe_auc, safe_ap, rates, precision_at_prevalence, queue_by_budget,
                     MODEL_NAMES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
CIC = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023"]
ALL = CIC + ["UNSW-NB15", "ToN-IoT"]
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.01                       # nominal benign false-alarm rate
STREAM_PI = 0.01                   # attack prevalence of simulated target streams
BUDGETS = [0.01, 0.02, 0.05]       # analyst review budget (fraction of the stream)
WARMUP = 0.2                       # unlabeled warm-up fraction for label-free thresholds
PRIORITIES = ["detector", "novelty", "combined", "fisher"]


def load(name):
    df = pd.read_parquet(os.path.join(PROC, f"{name}.parquet"))
    feats = [c for c in df.columns if c not in ("label", "family", "group")]
    return df, feats


def split(df, seed, fr=(0.5, 0.2, 0.3)):
    """Stratified (by family) train/cal/test split."""
    rng = np.random.default_rng(seed)
    part = np.empty(len(df), dtype=np.int8)
    for _, idx in df.groupby("family").indices.items():
        idx = rng.permutation(idx)
        a = int(round(fr[0] * len(idx)))
        b = a + int(round(fr[1] * len(idx)))
        part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
    return part


def make_stream(y, pi, rng, attack_mask=None):
    """Indices of a target stream: all benign flows + attacks at prevalence pi."""
    ben = np.flatnonzero(y == 0)
    att = np.flatnonzero((y == 1) if attack_mask is None else attack_mask)
    n_att = min(len(att), int(round(pi / (1 - pi) * len(ben))))
    idx = np.concatenate([ben, rng.choice(att, n_att, replace=False)])
    return rng.permutation(idx)


def detection_metrics(p_raw, cals, y, tau):
    out = {"auroc": safe_auc(y, p_raw), "ap": safe_ap(y, p_raw)}
    for cname, f in cals.items():
        p = f(p_raw)
        out[f"ece_{cname}"] = ece(p, y)
        out[f"brier_{cname}"] = brier(p, y)
    tpr, fpr = rates(p_raw >= tau, y)
    out.update(tpr_at_alpha=tpr, fpr_at_alpha=fpr, fpr_inflation=fpr / ALPHA)
    for pi in (0.1, 0.01, 0.001):
        out[f"prec_at_pi{pi}"] = precision_at_prevalence(tpr, fpr, pi)
    return out


def triage_metrics(pr, y, rng, src_thresholds, extra_mask=None):
    """Queue metrics on a stream; pr = dict priority_name -> priority array (higher = review)."""
    rows = []
    n = len(y)
    w = int(WARMUP * n)
    for name, s in pr.items():
        for c in BUDGETS:
            # oracle-free ideal: exact top-c of the whole stream
            q = queue_by_budget(s, c)
            rec = q[y == 1].mean() if (y == 1).any() else np.nan
            # source threshold transferred without adaptation
            t_src = src_thresholds[name][c]
            q_src = s[w:] >= t_src
            # label-free warm-up quantile on the unlabeled target stream
            t_tgt = np.quantile(s[:w], 1 - c)
            q_tgt = s[w:] >= t_tgt
            yy = y[w:]
            row = dict(priority=name, budget=c, recall_topc=rec,
                       prec_topc=y[q].mean() if q.any() else np.nan,
                       load_src=q_src.mean() / c, recall_src=q_src[yy == 1].mean(),
                       load_tgt=q_tgt.mean() / c, recall_tgt=q_tgt[yy == 1].mean())
            if extra_mask is not None:
                em = extra_mask
                row["recall_topc_unseen"] = q[em].mean() if em.any() else np.nan
            rows.append(row)
    return rows


class Fitted:
    """Detector + calibrators + novelty + conformal triage trained on one source split."""

    def __init__(self, model, seed, Xtr, ytr, Xcal, ycal):
        t0 = time.time()
        self.det = Detector(model, seed).fit(Xtr, ytr)
        pc = self.det.score(Xcal)
        self.cals = {"raw": lambda p: p, "platt": Platt().fit(pc, ycal),
                     "iso": Isotonic().fit(pc, ycal)}
        self.tau = np.quantile(pc[ycal == 0], 1 - ALPHA)
        self.nov = KNNNovelty(seed=seed).fit(Xtr[ytr == 0])  # reference = benign training flows
        nc = self.nov.score(Xcal)
        self.tri = ConformalTriage().fit(pc[ycal == 0], nc[ycal == 0])
        self.fit_seconds = time.time() - t0
        # source thresholds for each priority/budget on a source stream (cal split)
        rng = np.random.default_rng(seed + 1000)
        idx = make_stream(ycal, STREAM_PI, rng)
        pr = self.priorities(pc[idx], nc[idx])
        self.src_thr = {k: {c: np.quantile(v, 1 - c) for c in BUDGETS} for k, v in pr.items()}

    def priorities(self, det, nov):
        tie = {"detector": 1e-9 * det, "novelty": 1e-12 * nov, "combined": 1e-9 * det + 1e-12 * nov,
               "fisher": 1e-9 * det + 1e-12 * nov}
        return {m: -np.log(self.tri.pvalues(det, nov, mode=m)) + tie[m] for m in PRIORITIES}

    def scores(self, X):
        return self.det.score(X), self.nov.score(X)


def run_xd():
    data = {n: load(n) for n in ALL}
    rows, trows = [], []
    for seed in SEEDS:
        parts = {n: split(data[n][0], seed) for n in ALL}
        for src in ALL:
            df, feats = data[src]
            X, y, pt = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
            targets = [src] + ([t for t in CIC if t != src] if src in CIC else [])
            for model in MODEL_NAMES:
                F = Fitted(model, seed, X[pt == 0], y[pt == 0], X[pt == 1], y[pt == 1])
                for tgt in targets:
                    tdf, _ = data[tgt]
                    tm = parts[tgt] == 2
                    Xt = tdf.loc[tm, feats].to_numpy(np.float32)
                    yt = tdf.loc[tm, "label"].to_numpy()
                    gt = tdf.loc[tm, "group"].to_numpy()
                    t0 = time.time()
                    det, nov = F.scores(Xt)
                    score_s = time.time() - t0
                    base = dict(seed=seed, source=src, target=tgt, model=model,
                                setting="in-domain" if tgt == src else "cross-dataset")
                    m = detection_metrics(det, F.cals, yt, F.tau)
                    m.update(fit_s=F.fit_seconds, score_us_per_flow=1e6 * score_s / len(yt))
                    rows.append({**base, **m})
                    # per-group recall at the source alpha threshold
                    for g in np.unique(gt[yt == 1]):
                        gm = gt == g
                        rows.append({**base, "group": g, "group_recall": (det[gm] >= F.tau).mean(),
                                     "group_n": int(gm.sum())})
                    rng = np.random.default_rng(seed + 7)
                    idx = make_stream(yt, STREAM_PI, rng)
                    pr = F.priorities(det[idx], nov[idx])
                    for r in triage_metrics(pr, yt[idx], rng, F.src_thr):
                        trows.append({**base, **r})
                print(seed, src, model, "done", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, "xd_detection.csv"), index=False)
                pd.DataFrame(trows).to_csv(os.path.join(RES, "xd_triage.csv"), index=False)


def run_os(names=None, models=("LGBM",), seeds=(0, 1, 2), min_n=300):
    names = names or ALL
    tag = "" if names == ALL else "_" + "+".join(names)
    rows, trows = [], []
    for name in names:
        df, feats = load(name)
        X, y, fam = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["family"].to_numpy()
        fams = [f for f, n in df.loc[df.label == 1, "family"].value_counts().items() if n >= min_n]
        for seed in seeds:
            pt = split(df, seed)
            for f, model in itertools.product(fams, models):
                keep = fam != f
                tr, ca, te = (pt == 0) & keep, (pt == 1) & keep, pt == 2
                F = Fitted(model, seed, X[tr], y[tr], X[ca], y[ca])
                det, nov = F.scores(X[te])
                yt, ft = y[te], fam[te]
                unseen = ft == f
                base = dict(dataset=name, seed=seed, model=model, held_out=f,
                            n_unseen=int(unseen.sum()))
                # stream where the only attacks are the unseen family (zero-day campaign)
                rng = np.random.default_rng(seed + 11)
                idx = make_stream(yt, STREAM_PI, rng, attack_mask=unseen)
                ys = yt[idx]
                p_iso = F.cals["iso"](det[unseen])
                rows.append({**base,
                             "recall_unseen_at_alpha": (det[unseen] >= F.tau).mean(),
                             "recall_seen_at_alpha": (det[(yt == 1) & ~unseen] >= F.tau).mean(),
                             "fpr_at_alpha": (det[yt == 0] >= F.tau).mean(),
                             "confident_miss_iso": (p_iso < 0.05).mean(),
                             "mean_p_iso_unseen": p_iso.mean(),
                             "ece_iso_stream": ece(F.cals["iso"](det[idx]), ys),
                             "auroc_unseen_vs_benign": safe_auc(ys, det[idx]),
                             "auroc_nov_unseen_vs_benign": safe_auc(ys, nov[idx])})
                pr = F.priorities(det[idx], nov[idx])
                for r in triage_metrics(pr, ys, rng, F.src_thr):
                    trows.append({**base, **r})
                print(name, seed, f, "done", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, f"os_detection{tag}.csv"), index=False)
                pd.DataFrame(trows).to_csv(os.path.join(RES, f"os_triage{tag}.csv"), index=False)


if __name__ == "__main__":
    if sys.argv[1] == "xd":
        run_xd()
    else:
        run_os(sys.argv[2:] or None)

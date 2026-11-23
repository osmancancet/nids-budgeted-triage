"""Additional revision experiments that require retraining (LightGBM, five seeds).

  python revision_extra.py features    # cross-dataset with a tooling-robust feature subset
  python revision_extra.py subsample   # sensitivity to the amount of benign training data
  python revision_extra.py reference   # novelty reference size, staleness proxy and poisoning
  python revision_extra.py temporal    # chronological splits on CSE-CIC-IDS2018
  python revision_extra.py cost        # cost breakdown and scaling

Outputs: results/revision/extra_*.csv
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import run_experiments as R
from nidslib import Detector, KNNNovelty, LogScaler
from revision_analysis import Ctx, priorities, topk_mask, pvals, make_stream, online_quantile, rolling_quantile_flags, BUDGETS, ALPHA
from prepare import CIC_FEATURES

OUT = os.path.join(R.ROOT, "results", "revision")
TAG = os.environ.get("TAG", "")   # suffix for output files of a rerun
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

# Packet-count and packet-length features plus protocol and initial window sizes: quantities that
# CICFlowMeter versions compute from the same packet fields. Timing, rate, flag-count, header-length,
# bulk, subflow and active/idle features, whose implementation differs across versions, are excluded.
ROBUST = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
          'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
          'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
          'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
          'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Down/Up Ratio',
          'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Protocol']


def fit_eval(model, seed, Xtr, ytr, Xcal, ycal):
    F = R.Fitted(model, seed, Xtr, ytr, Xcal, ycal)
    pc, nc = F.scores(Xcal)
    return F, Ctx(pc, nc, ycal)


def triage_row(ctx, det, nov, y, seed, c=0.01, mask=None):
    rng = np.random.default_rng(seed + 7)
    idx = make_stream(y, 0.01, rng, attack_mask=mask)
    ys = y[idx]
    pr, _, _ = priorities(ctx, det[idx], nov[idx], extra=True)
    out = {}
    for name in ["detector", "fisher", "wfisher0.7"]:
        q = topk_mask(pr[name], c)
        out[f"rec_{name.replace('wfisher0.7', 'wfisher')}"] = q[ys == 1].mean()
    w = int(0.2 * len(ys))
    t = np.quantile(pr["fisher"][:w], 1 - c)
    out["load_warmup"] = (pr["fisher"][w:] >= t).mean() / c
    return out


# ------------------------------------------------------------ tooling shift
def run_features():
    data = {n: R.load(n) for n in R.CIC}
    rows = []
    for seed in SEEDS:
        parts = {n: R.split(data[n][0], seed) for n in R.CIC}
        for fs_name, feats in [("all", CIC_FEATURES + ["Protocol"]), ("robust", ROBUST)]:
            for src in R.CIC:
                df = data[src][0]
                X, y, pt = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
                F, ctx = fit_eval("LGBM", seed, X[pt == 0], y[pt == 0], X[pt == 1], y[pt == 1])
                for tgt in R.CIC:
                    tdf = data[tgt][0]
                    tm = parts[tgt] == 2
                    Xt, yt = tdf.loc[tm, feats].to_numpy(np.float32), tdf.loc[tm, "label"].to_numpy()
                    det = F.det.score(Xt)
                    rows.append(dict(seed=seed, features=fs_name, source=src, target=tgt,
                                     auroc=roc_auc_score(yt, det),
                                     fpr_inflation=(det[yt == 0] >= ctx.tau).mean() / ALPHA,
                                     tpr=(det[yt == 1] >= ctx.tau).mean()))
            # benign-only domain classifier: how separable are the benign flows of two datasets?
            for a, b in itertools.combinations(R.CIC, 2):
                A = data[a][0]; B = data[b][0]
                xa = A.loc[(A.label == 0) & (parts[a] == 0), feats].to_numpy(np.float32)
                xb = B.loc[(B.label == 0) & (parts[b] == 0), feats].to_numpy(np.float32)
                ta = A.loc[(A.label == 0) & (parts[a] == 2), feats].to_numpy(np.float32)
                tb = B.loc[(B.label == 0) & (parts[b] == 2), feats].to_numpy(np.float32)
                rng = np.random.default_rng(seed)
                n = min(len(xa), len(xb), 50000)
                xa, xb = xa[rng.choice(len(xa), n, replace=False)], xb[rng.choice(len(xb), n, replace=False)]
                D = Detector("LGBM", seed).fit(np.r_[xa, xb], np.r_[np.zeros(n), np.ones(n)])
                s = D.score(np.r_[ta, tb])
                rows.append(dict(seed=seed, features=fs_name, source=a, target=b, kind="benign_domain_auc",
                                 auroc=roc_auc_score(np.r_[np.zeros(len(ta)), np.ones(len(tb))], s)))
            print("features", seed, fs_name, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_features.csv"), index=False)


# ----------------------------------------------------- benign subsampling
def run_subsample():
    rows = []
    for name in ["CIC-IDS2017", "ToN-IoT"]:
        df, feats = R.load(name)
        X, y, fam = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["family"].to_numpy()
        fams = [f for f, n in df.loc[df.label == 1, "family"].value_counts().items() if n >= 300]
        targets = [t for t in R.CIC if t != name] if name in R.CIC else []
        tdata = {t: R.load(t)[0] for t in targets}
        for seed in SEEDS:
            pt = R.split(df, seed)
            tparts = {t: R.split(tdata[t], seed) for t in targets}
            for frac in [0.25, 0.5, 1.0]:
                rng = np.random.default_rng(100 + seed)
                drop = (y == 0) & (pt == 0) & (rng.random(len(y)) > frac)  # thin benign training flows only
                trm = (pt == 0) & ~drop
                F, ctx = fit_eval("LGBM", seed, X[trm], y[trm], X[pt == 1], y[pt == 1])
                te = pt == 2
                det, nov = F.scores(X[te])
                r = dict(dataset=name, seed=seed, benign_frac=frac, protocol="in-domain",
                         auroc=roc_auc_score(y[te], det),
                         fpr_inflation=(det[y[te] == 0] >= ctx.tau).mean() / ALPHA)
                r.update(triage_row(ctx, det, nov, y[te], seed))
                rows.append(r)
                for t in targets:
                    tm = tparts[t] == 2
                    Xt, yt = tdata[t].loc[tm, feats].to_numpy(np.float32), tdata[t].loc[tm, "label"].to_numpy()
                    d2, n2 = F.scores(Xt)
                    r = dict(dataset=name, seed=seed, benign_frac=frac, protocol="cross-dataset", target=t,
                             auroc=roc_auc_score(yt, d2), fpr_inflation=(d2[yt == 0] >= ctx.tau).mean() / ALPHA)
                    r.update(triage_row(ctx, d2, n2, yt, seed))
                    rows.append(r)
                for f in fams:
                    keep = fam != f
                    tr, ca = trm & keep, (pt == 1) & keep
                    F, ctx = fit_eval("LGBM", seed, X[tr], y[tr], X[ca], y[ca])
                    det, nov = F.scores(X[te])
                    un = fam[te] == f
                    r = dict(dataset=name, seed=seed, benign_frac=frac, protocol="unseen", held_out=f,
                             recall_unseen=(det[un] >= ctx.tau).mean(),
                             auroc_nov=roc_auc_score(np.r_[np.zeros((y[te] == 0).sum()), np.ones(un.sum())],
                                                     np.r_[nov[y[te] == 0], nov[un]]))
                    r.update(triage_row(ctx, det, nov, y[te], seed, mask=un))
                    rows.append(r)
                print("subsample", name, seed, frac, flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_subsample.csv"), index=False)


# ------------------------------------------- novelty reference: size, poisoning
def run_reference():
    """Uses the LightGBM leave-one-family-out dumps for the detector score and refits only the kNN."""
    rows = []
    for name in R.ALL:
        df, feats = R.load(name)
        X, y, fam = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["family"].to_numpy()
        fams = [f for f, n in df.loc[df.label == 1, "family"].value_counts().items() if n >= 300]
        for seed in [int(v) for v in os.environ.get("REF_SEEDS", "0,1,2").split(",")]:
            pt = R.split(df, seed)
            for f in fams:
                safe = "".join(c if c.isalnum() else "_" for c in f)
                z = np.load(os.path.join(R.ROOT, "data", "dumps", "os", f"{name}__{safe}__LGBM__s{seed}.npz"))
                keep = fam != f
                tr, ca, te = (pt == 0) & keep, (pt == 1) & keep, pt == 2
                ben_tr = np.flatnonzero(tr & (y == 0))
                pool_f = np.flatnonzero((pt == 0) & (fam == f))  # held-out family flows from the training period
                det = z["det"].astype(np.float64)
                yt, un = y[te], fam[te] == f
                configs = [(1000, 0), (5000, 0), (20000, 0), (50000, 0), (20000, 0.001), (20000, 0.01), (20000, 0.05)]
                if os.environ.get("REF_CONTAM_ONLY"):
                    configs = [c_ for c_ in configs if c_[0] == 20000]
                for n_ref, contam in configs:
                    rng = np.random.default_rng(seed)
                    ref = rng.choice(ben_tr, size=min(n_ref, len(ben_tr)), replace=False)
                    n_bad = int(round(contam * len(ref)))
                    if n_bad > len(pool_f):
                        continue
                    if n_bad:
                        ref = np.r_[ref[:len(ref) - n_bad], rng.choice(pool_f, n_bad, replace=False)]
                    sc = LogScaler().fit(X[tr][y[tr] == 0])
                    if os.environ.get("REF_FAISS"):     # exact search with Faiss (same neighbours, faster)
                        import faiss
                        ix = faiss.IndexFlatL2(X.shape[1])
                        ix.add(np.ascontiguousarray(sc.transform(X[ref]), dtype=np.float32))
                        knn = lambda Z: np.sqrt(np.maximum(ix.search(np.ascontiguousarray(Z, dtype=np.float32), 5)[0], 0)).mean(1)
                    else:
                        nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(X[ref]))
                        knn = lambda Z: nn.kneighbors(Z)[0].mean(1)
                    t0 = time.time()
                    nov = knn(sc.transform(X[te]))
                    t_knn = time.time() - t0
                    nc = knn(sc.transform(X[ca]))
                    ctx = Ctx(z["cal_det"], nc, z["cal_y"])
                    r = dict(dataset=name, held_out=f, seed=seed, n_ref=len(ref), contamination=contam,
                             knn_us=1e6 * t_knn / te.sum(),
                             auroc_nov=roc_auc_score(np.r_[np.zeros((yt == 0).sum()), np.ones(un.sum())],
                                                     np.r_[nov[yt == 0], nov[un]]))
                    r.update(triage_row(ctx, det, nov, yt, seed, mask=un))
                    rows.append(r)
                print("reference", name, seed, f, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"extra_reference{TAG}.csv"), index=False)


# ------------------------------------------------------------- temporal
def run_temporal():
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
        for proto in ["T1-benign-time", "T1-random", "T2-day", "T2-random"]:
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
            print("temporal", seed, proto, r, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_temporal.csv"), index=False)
            pd.DataFrame(ns).to_csv(os.path.join(OUT, "extra_temporal_stream.csv"), index=False)


# ------------------------------------------------ leave-one-group-out
def run_group():
    """Hold out a whole coarse attack group (e.g. every DoS family) instead of one family, so
    that no close relative of the unseen attacks remains in training (CICFlowMeter datasets)."""
    rows = []
    for name in R.CIC:
        df, feats = R.load(name)
        X, y, grp = df[feats].to_numpy(np.float32), df["label"].to_numpy(), df["group"].to_numpy()
        groups = [g for g, n in df.loc[df.label == 1, "group"].value_counts().items() if n >= 300]
        for seed in SEEDS:
            pt = R.split(df, seed)
            for g in groups:
                keep = grp != g
                if (y[(pt == 0) & keep] == 1).sum() == 0:
                    continue
                tr, ca, te = (pt == 0) & keep, (pt == 1) & keep, pt == 2
                F, ctx = fit_eval("LGBM", seed, X[tr], y[tr], X[ca], y[ca])
                det, nov = F.scores(X[te])
                un = grp[te] == g
                yt = y[te]
                p_iso = ctx.iso.predict(det[un])
                r = dict(dataset=name, held_out_group=g, seed=seed, n_unseen=int(un.sum()),
                         recall_unseen=(det[un] >= ctx.tau).mean(), cm_005=(p_iso < 0.05).mean(),
                         auroc_det=roc_auc_score(np.r_[np.zeros((yt == 0).sum()), np.ones(un.sum())],
                                                 np.r_[det[yt == 0], det[un]]),
                         auroc_nov=roc_auc_score(np.r_[np.zeros((yt == 0).sum()), np.ones(un.sum())],
                                                 np.r_[nov[yt == 0], nov[un]]))
                r.update(triage_row(ctx, det, nov, yt, seed, mask=un))
                rows.append(r)
                print("group", name, seed, g, flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"extra_group{TAG}.csv"), index=False)


# ------------------------------------------------------------------- cost
def run_cost():
    """Per-flow cost of detector inference and kNN novelty scoring, by model, reference size,
    thread count and kNN index; measured on CSE-CIC-IDS2018 test flows (seed 0)."""
    import threadpoolctl
    df, feats = R.load("CSE-CIC-IDS2018")
    X, y = df[feats].to_numpy(np.float32), df["label"].to_numpy()
    pt = R.split(df, 0)
    Xtr, ytr, Xte = X[pt == 0], y[pt == 0], X[pt == 2][:50000]
    rows = []
    for threads in [1, 4]:
        with threadpoolctl.threadpool_limits(threads):
            for m in ["LGBM", "RF", "MLP", "LR"]:
                D = Detector(m, 0).fit(Xtr, ytr)
                if m == "LGBM":
                    D.clf.set_params(n_jobs=threads)
                if m == "RF":
                    D.clf.set_params(n_jobs=threads)
                D.score(Xte[:1000])
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter(); D.score(Xte); ts.append(time.perf_counter() - t0)
                rows.append(dict(component="detector", model=m, threads=threads, n_ref=np.nan,
                                 us_per_flow=1e6 * np.median(ts) / len(Xte)))
            sc = LogScaler().fit(Xtr[ytr == 0])
            Zte = sc.transform(Xte)
            ben = Xtr[ytr == 0]
            for n_ref in [1000, 5000, 20000, 50000, 100000]:
                rng = np.random.default_rng(0)
                Zr = sc.transform(ben[rng.choice(len(ben), min(n_ref, len(ben)), replace=False)])
                for algo in ["brute", "kd_tree", "ball_tree"]:
                    nn = NearestNeighbors(n_neighbors=5, algorithm=algo, n_jobs=threads).fit(Zr)
                    n_q = 10000 if algo != "brute" or n_ref <= 20000 else 5000
                    t0 = time.perf_counter(); nn.kneighbors(Zte[:n_q]); dt = time.perf_counter() - t0
                    rows.append(dict(component="knn", model=algo, threads=threads, n_ref=n_ref,
                                     us_per_flow=1e6 * dt / n_q))
                try:
                    import faiss
                    faiss.omp_set_num_threads(threads)
                    idx = faiss.IndexHNSWFlat(Zr.shape[1], 32)
                    idx.add(Zr.astype(np.float32))
                    idx.hnsw.efSearch = 64
                    t0 = time.perf_counter(); _, I = idx.search(Zte[:10000].astype(np.float32), 5)
                    dt = time.perf_counter() - t0
                    exact = NearestNeighbors(n_neighbors=5).fit(Zr).kneighbors(Zte[:2000], return_distance=False)
                    rec = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(I[:2000], exact)])
                    rows.append(dict(component="knn", model="hnsw", threads=threads, n_ref=n_ref,
                                     us_per_flow=1e6 * dt / 10000, recall_at5=rec))
                except ImportError:
                    pass
                print("cost", threads, n_ref, flush=True)
            # conformal p-values and Fisher combination
            cal = np.sort(np.random.default_rng(0).random(60000))
            s = np.random.default_rng(1).random(len(Xte))
            t0 = time.perf_counter(); p1 = pvals(cal, s); p2 = pvals(cal, s[::-1])
            from scipy.stats import chi2
            chi2.sf(-2 * (np.log(p1) + np.log(p2)), 4); dt = time.perf_counter() - t0
            rows.append(dict(component="conformal+fisher", model="", threads=threads, us_per_flow=1e6 * dt / len(Xte)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "extra_cost.csv"), index=False)


if __name__ == "__main__":
    {"features": run_features, "subsample": run_subsample, "reference": run_reference,
     "temporal": run_temporal, "cost": run_cost, "group": run_group}[sys.argv[1]]()

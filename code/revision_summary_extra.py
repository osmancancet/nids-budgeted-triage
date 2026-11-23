"""Summaries of the additional experiments (called from revision_summary.py extra)."""
import os
import numpy as np
import pandas as pd
from revision_summary import RES, md_table, paired, rd


def _read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print("missing", name)
        return None
    return pd.read_csv(p)


def temporal(N):
    t = _read("extra_temporal.csv")
    if t is None:
        return
    T = t.groupby("protocol")[["n_test", "attack_share", "auroc", "ece_iso", "fpr_inflation", "tpr", "auroc_nov"]] \
        .agg(["mean", "std"]).round(4)
    T.columns = ["_".join(c) for c in T.columns]
    T = T.reset_index()
    md_table(T, "temporal_detection.md", 3)
    N["temporal_detection"] = T.to_dict("records")
    s = _read("extra_temporal_stream.csv")
    S = s.groupby(["protocol", "budget", "priority", "rule"]).agg(
        load=("load", "mean"), load_sd=("load", "std"), load_min10=("load_min10", "mean"),
        load_max10=("load_max10", "mean"), recall=("recall", "mean"), precision=("precision", "mean")).reset_index()
    md_table(S, "temporal_stream.md", 3)
    N["temporal_stream"] = S.round(4).to_dict("records")


def features(N):
    f = _read("extra_features.csv")
    if f is None:
        return
    if "kind" not in f:
        f["kind"] = np.nan
    det = f[f.kind.isna() & (f.source != f.target)]
    P = det.groupby(["features", "source", "target"])[["auroc", "fpr_inflation", "tpr"]].mean().reset_index()
    md_table(P, "features_xd_pairs.md")
    M = P.groupby("features")[["auroc", "fpr_inflation", "tpr"]].agg(["mean", "min", "max"]).round(3)
    M.columns = ["_".join(c) for c in M.columns]
    M = M.reset_index()
    md_table(M, "features_xd_summary.md")
    N["features_xd"] = M.to_dict("records")
    a = P[P.features == "robust"].set_index(["source", "target"])
    b = P[P.features == "all"].set_index(["source", "target"])
    N["features_paired_auroc"] = paired(a.auroc, b.auroc)
    N["features_paired_logfprinf"] = paired(np.log10(a.fpr_inflation.clip(lower=1e-3)),
                                            np.log10(b.fpr_inflation.clip(lower=1e-3)))
    dom = f[f.kind == "benign_domain_auc"].groupby(["features", "source", "target"]).auroc.mean().reset_index()
    md_table(dom, "features_domain_auc.md")
    N["features_domain_auc"] = dom.round(4).to_dict("records")


def group(N):
    g = _read("extra_group.csv")
    if g is None:
        return
    G = g.groupby(["dataset", "held_out_group"]).agg(n_unseen=("n_unseen", "mean"), recall=("recall_unseen", "mean"),
                                                     cm_005=("cm_005", "mean"), auroc_det=("auroc_det", "mean"),
                                                     auroc_nov=("auroc_nov", "mean"), rec_det=("rec_detector", "mean"),
                                                     rec_fisher=("rec_fisher", "mean"),
                                                     load_warmup=("load_warmup", "median")).reset_index()
    md_table(G, "group_out.md")
    N["group_out"] = G.round(4).to_dict("records")
    N["group_out_summary"] = dict(n_groups=int(len(G)), n_below_half=int((G.recall < 0.5).sum()),
                                  median_recall=rd(G.recall.median()),
                                  fisher_vs_det=paired(G.set_index(["dataset", "held_out_group"]).rec_fisher,
                                                       G.set_index(["dataset", "held_out_group"]).rec_det),
                                  load_warmup_range=[rd(G.load_warmup.min()), rd(G.load_warmup.max())])


def reference(N):
    r = _read("extra_reference.csv")
    if r is None:
        return
    fam = r.groupby(["dataset", "held_out", "n_ref", "contamination"])[["auroc_nov", "rec_detector", "rec_fisher",
                                                                        "load_warmup", "knn_us"]].mean().reset_index()
    R = fam.groupby(["n_ref", "contamination"]).agg(n_fam=("held_out", "size"), auroc_nov=("auroc_nov", "mean"),
                                                    rec_det=("rec_detector", "mean"), rec_fisher=("rec_fisher", "mean"),
                                                    load_warmup=("load_warmup", "median"),
                                                    knn_us=("knn_us", "median")).reset_index()
    md_table(R, "reference_size_poison.md")
    N["reference"] = R.round(4).to_dict("records")
    base = fam[(fam.n_ref == 20000) & (fam.contamination == 0)].set_index(["dataset", "held_out"])
    for (n, c), g in fam.groupby(["n_ref", "contamination"]):
        if (n, c) == (20000, 0):
            continue
        g = g.set_index(["dataset", "held_out"])
        N[f"reference_paired_n{n}_c{c}"] = paired(g.rec_fisher, base.rec_fisher)


def subsample(N):
    s = _read("extra_subsample.csv")
    if s is None:
        return
    rows = []
    for (d, pr, fr), g in s.groupby(["dataset", "protocol", "benign_frac"]):
        rows.append(dict(dataset=d, protocol=pr, benign_frac=fr,
                         auroc=g.auroc.mean() if "auroc" in g and g.auroc.notna().any() else np.nan,
                         fpr_inflation=g.fpr_inflation.mean() if g.fpr_inflation.notna().any() else np.nan,
                         recall_unseen=g.recall_unseen.mean() if "recall_unseen" in g and g.recall_unseen.notna().any() else np.nan,
                         rec_det=g.rec_detector.mean(), rec_fisher=g.rec_fisher.mean(),
                         load_warmup=g.load_warmup.median()))
    S = pd.DataFrame(rows)
    md_table(S, "subsample.md")
    N["subsample"] = S.round(4).to_dict("records")


def cost(N):
    c = _read("extra_cost.csv")
    if c is None:
        return
    md_table(c, "cost.md", 2)
    N["cost"] = c.round(3).to_dict("records")


def extra_all(N):
    for f in (temporal, features, group, reference, subsample, cost):
        f(N)

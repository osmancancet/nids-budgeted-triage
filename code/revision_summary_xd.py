"""In-domain / cross-dataset summaries for the revision (called from revision_summary.py xd)."""
import os
import numpy as np
import pandas as pd
from revision_summary import RES, ROOT, md_table, paired, rd

ORIG = os.path.join(ROOT, "results")
CIC = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023"]


def _rng(s):
    return [rd(s.min(), 3), rd(s.max(), 3)]


def xd_reconcile(N):
    """Ranges quoted in the text, recomputed per directed pair (mean over seeds) for every model."""
    d = pd.read_csv(os.path.join(ORIG, "xd_detection.csv"))
    d = d[d.group.isna()]
    x = d[d.setting == "cross-dataset"]
    pp = x.groupby(["model", "source", "target"])[["auroc", "ece_raw", "ece_platt", "ece_iso", "fpr_inflation",
                                                   "tpr_at_alpha", "prec_at_pi0.01"]].mean().reset_index()
    md_table(pp, "xd_per_pair_all_models.md")
    out = {}
    for m, g in pp.groupby("model"):
        out[m] = {c: _rng(g[c]) for c in ["auroc", "ece_raw", "ece_platt", "ece_iso", "fpr_inflation",
                                          "tpr_at_alpha", "prec_at_pi0.01"]}
    out["all"] = {c: _rng(pp[c]) for c in ["auroc", "ece_raw", "ece_platt", "ece_iso", "fpr_inflation",
                                           "tpr_at_alpha", "prec_at_pi0.01"]}
    # model means over pairs (Table 3 form)
    mm = pp.groupby("model")[["auroc", "ece_raw", "ece_iso", "fpr_inflation", "prec_at_pi0.01"]].mean()
    out["model_mean_over_pairs"] = {c: _rng(mm[c]) for c in mm.columns}
    # per-run range of AUROC
    out["auroc_run_range"] = _rng(x.auroc)
    # flooding vs silent sources (LightGBM)
    lg = pp[pp.model == "LGBM"]
    out["lgbm_by_source"] = {src: {c: _rng(g[c]) for c in ["fpr_inflation", "tpr_at_alpha", "auroc"]}
                             for src, g in lg.groupby("source")}
    out["lgbm_pairs"] = lg[["source", "target", "auroc", "fpr_inflation", "tpr_at_alpha"]].round(3).to_dict("records")
    ind = d[d.setting == "in-domain"]
    ip = ind.groupby(["model", "source"])[["auroc", "ece_raw", "ece_iso", "fpr_inflation", "prec_at_pi0.01"]].mean()
    out["in_domain"] = {c: _rng(ip[c]) for c in ip.columns}
    N["xd_reconcile"] = out
    # budget overflow / underfill from the source threshold (original triage table)
    t = pd.read_csv(os.path.join(ORIG, "xd_triage.csv"))
    t = t[t.model == "LGBM"]
    L = t.groupby(["setting", "priority", "budget"]).load_src.quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    L.columns = [str(c) for c in L.columns]
    md_table(L, "xd_source_threshold_load.md")
    N["xd_source_load"] = L.round(3).to_dict("records")


def xd_check(N):
    """Re-scored dumps against the original run (same seeds and splits)."""
    d = pd.read_csv(os.path.join(ORIG, "xd_detection.csv"))
    d = d[d.group.isna()][["seed", "source", "target", "model", "auroc", "fpr_inflation"]]
    c = pd.read_csv(os.path.join(RES, "xd_detection_check.csv"))
    m = c.merge(d, on=["seed", "source", "target", "model"], suffixes=("", "_orig"))
    m["da"] = (m.auroc - m.auroc_orig).abs()
    tree = m[m.model.isin(["LGBM", "RF"])]
    off = m[m.da > 1e-3]
    N["xd_repro"] = dict(n=int(len(m)), max_abs_auroc=rd(m.da.max(), 6),
                         max_abs_fprinf=rd((m.fpr_inflation - m.fpr_inflation_orig).abs().max(), 6),
                         max_abs_auroc_trees=float(tree.da.max()), n_off=int(len(off)),
                         off_sources=sorted(off.source.unique().tolist()), off_models=sorted(off.model.unique().tolist()))


def xd_budget(N):
    w = pd.read_csv(os.path.join(RES, "xd_warmup.csv"))
    w = w[w.model == "LGBM"]
    W = w.groupby(["regime", "priority", "budget", "warmup"]).load.quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    W.columns = [str(c) for c in W.columns]
    md_table(W, "xd_warmup.md")
    N["xd_warmup"] = W.round(4).to_dict("records")

    s = pd.read_csv(os.path.join(RES, "xd_nonstationary.csv"))
    s = s[s.model == "LGBM"]
    S = s.groupby(["regime", "scenario", "budget", "rule"]).agg(load_med=("load", "median"),
                                                                 load_q10=("load", lambda v: v.quantile(.1)),
                                                                 load_q90=("load", lambda v: v.quantile(.9)),
                                                                 recall=("recall", "mean")).reset_index()
    md_table(S, "xd_nonstationary.md")
    N["xd_nonstationary"] = S.round(4).to_dict("records")

    p = pd.read_csv(os.path.join(RES, "xd_prevalence.csv"))
    p = p[p.model == "LGBM"]
    P = p.groupby(["regime", "target_pi", "budget", "priority"]).agg(real_pi=("real_pi", "mean"),
                                                                     load_med=("load", "median"),
                                                                     recall=("recall", "mean"),
                                                                     precision=("precision", "mean")).reset_index()
    md_table(P, "xd_prevalence.md")
    N["xd_prevalence"] = P.round(4).to_dict("records")

    q = pd.read_csv(os.path.join(RES, "xd_queue_composition.csv"))
    q = q[q.model == "LGBM"]
    Q = q.groupby(["regime", "budget", "priority"]).agg(load_med=("load", "median"), recall=("recall", "mean"),
                                                        precision=("precision", "mean"),
                                                        fa_per_tp=("fa_per_tp", "median"),
                                                        fa_novelty_share=("fa_novelty_share", "mean")).reset_index()
    md_table(Q, "xd_queue.md")
    N["xd_queue"] = Q.round(4).to_dict("records")

    v = pd.read_csv(os.path.join(RES, "xd_validity.csv"))
    V = v.groupby(["regime", "model"])[["fpr_pdet", "fpr_pB"]].agg(["mean", "min", "max"]).round(4)
    V.columns = ["_".join(c) for c in V.columns]
    V = V.reset_index()
    md_table(V, "xd_validity.md", 4)
    N["xd_validity"] = V.to_dict("records")


def xd_priorities(N):
    p = pd.read_csv(os.path.join(RES, "xd_priorities.csv"))
    p = p[p.model == "LGBM"]
    pp = p.groupby(["regime", "source", "target", "budget", "priority"]).recall.mean().reset_index()
    T = pp.groupby(["regime", "budget", "priority"]).recall.mean().unstack("priority").reset_index()
    md_table(T, "xd_priorities.md")
    N["xd_priorities"] = T.round(4).to_dict("records")
    x = pp[pp.regime == "Cross-dataset"]
    for c in [0.01, 0.02, 0.05]:
        xb = x[x.budget == c]
        det = xb[xb.priority == "detector"].set_index(["source", "target"]).recall
        for pr in ["fisher", "wfisher0.7", "novelty", "combined"]:
            a = xb[xb.priority == pr].set_index(["source", "target"]).recall
            if len(a):
                N[f"xd_paired_{pr}_vs_det_b{c}"] = paired(a, det)


def xd_cost(N):
    c = pd.read_csv(os.path.join(RES, "xd_cost.csv"))
    C = c.groupby("model")[["det_us", "nov_us", "fit_s"]].median().reset_index()
    md_table(C, "xd_cost.md", 2)
    N["xd_cost"] = C.round(3).to_dict("records")


def xd_adapt(N):
    a = pd.read_csv(os.path.join(RES, "xd_adapt.csv"))
    pp = a.groupby(["source", "target", "budget", "variant"])[["recall", "precision", "load", "labels",
                                                                "n_attack_labels"]].mean().reset_index()
    T = pp.groupby(["budget", "variant"])[["recall", "precision", "load", "labels"]].mean().reset_index()
    md_table(T, "xd_adapt.md")
    N["xd_adapt"] = T.round(4).to_dict("records")
    N["xd_adapt_pairs"] = pp.round(4).to_dict("records")
    runs = a.set_index(["source", "target", "seed", "budget", "variant"]).recall.unstack("variant")
    for c in [0.01, 0.02, 0.05]:
        b = pp[pp.budget == c]
        base = b[b.variant == "base_fisher"].set_index(["source", "target"])
        for v in ["FB", "FB_noexplore", "TR_wfisher", "TR_novelty", "TN_wfisher", "TNTR_wfisher"]:
            x = b[b.variant == v].set_index(["source", "target"])
            if len(x):
                N[f"xd_adapt_{v}_vs_base_b{c}"] = paired(x.recall, base.recall)
                N[f"xd_adapt_{v}_vs_base_prec_b{c}"] = paired(x.precision, base.precision)
        r = runs.xs(c, level="budget")
        N[f"xd_adapt_fb_runs_better_b{c}"] = [int((r.FB > r.base_fisher).sum()), int(r.FB.notna().sum())]
        N[f"xd_adapt_fb_minmax_pair_b{c}"] = _rng(b[b.variant == "FB"].recall)
    cv = pd.read_csv(os.path.join(RES, "xd_adapt_curve.csv"))
    C = cv.groupby(["budget", "variant", "segment"])[["recall", "precision", "n_labels", "n_attack_labels"]].mean()
    C = C.reset_index()
    md_table(C, "xd_adapt_curve.md")
    N["xd_adapt_curve"] = C.round(4).to_dict("records")


def xd_all(N):
    xd_reconcile(N)
    for f in (xd_check, xd_budget, xd_priorities, xd_cost, xd_adapt):
        try:
            f(N)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)

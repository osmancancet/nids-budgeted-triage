"""Aggregate the revision result CSVs into the numbers and tables used in the revised manuscript.

  python revision_summary.py

Reads results/revision/*.csv, writes results/revision/revision_numbers.json and
results/revision/tables/*.md. Every number quoted in the revision comes from here.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "revision")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)
N = {}
DS = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023", "UNSW-NB15", "ToN-IoT"]
MODELS = ["LGBM", "RF", "MLP", "LR"]
rng = np.random.default_rng(0)


def rd(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def boot_ci(v, B=10000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return tuple(np.quantile(m, [0.025, 0.975]))


def paired(a, b):
    """a, b: Series indexed by cluster (family or pair). Returns mean diff, CI, Wilcoxon p, n."""
    a, b = a.align(b, join="inner")
    d = (a - b).dropna()
    lo, hi = boot_ci(d)
    try:
        p = wilcoxon(d[d != 0]).pvalue
    except ValueError:
        p = np.nan
    return dict(n=int(len(d)), mean=rd(d.mean(), 4), lo=rd(lo, 4), hi=rd(hi, 4), p=rd(p, 6),
                win=rd((d > 0).mean()), loss=rd((d < 0).mean()))


def md_table(df, path, floatfmt=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(v) for v in r.values) + " |")
    open(os.path.join(TAB, path), "w").write("\n".join(lines) + "\n")


def q(x, p):
    return float(np.nanquantile(x, p))


# ------------------------------------------------------------------- OS
def os_summary():
    d = pd.read_csv(os.path.join(RES, "os_detection_all.csv"))
    N["os_models_available"] = sorted(d.model.unique().tolist())
    N["os_seeds"] = sorted(d.seed.unique().tolist())
    fam = d.groupby(["model", "dataset", "held_out"]).mean(numeric_only=True).reset_index()
    fam.to_csv(os.path.join(RES, "os_per_family_all_models.csv"), index=False)
    # per-model summary
    rows = []
    for m in MODELS:
        x = fam[fam.model == m]
        if x.empty:
            continue
        r = dict(model=m, families=len(x), recall_seen_median=x.recall_seen_at_alpha.median(),
                 recall_unseen_median=x.recall_unseen_at_alpha.median(),
                 below_half=int((x.recall_unseen_at_alpha < .5).sum()),
                 confmiss_over_half=int((x["cm_0.05"] > .5).sum()),
                 auroc_det_median=x.auroc_det.median(), auroc_nov_median=x.auroc_nov.median(),
                 fpr_median=x.fpr_at_alpha.median())
        rows.append(r)
        N[f"os_{m}"] = {k: rd(v) for k, v in r.items() if k != "model"}
    T = pd.DataFrame(rows)
    md_table(T, "os_per_model.md")
    # families missed by all models / by LGBM only
    pv = fam.pivot_table(index=["dataset", "held_out"], columns="model", values="recall_unseen_at_alpha")
    avail = [m for m in MODELS if m in pv.columns]
    N["os_missed_by_all_models"] = int((pv[avail] < .5).all(axis=1).sum())
    N["os_missed_by_any_model"] = int((pv[avail] < .5).any(axis=1).sum())
    N["os_missed_by_lgbm"] = int((pv["LGBM"] < .5).sum())
    # agreement between models on which families are missed
    if len(avail) > 1:
        miss = (pv[avail] < .5)
        N["os_missed_lgbm_also_missed_by_k_others"] = miss[miss["LGBM"]].drop(columns="LGBM").sum(1).value_counts().sort_index().to_dict()
    # per-dataset table (LGBM, five seeds), as Table 4 of the paper
    L = fam[fam.model == "LGBM"]
    rows = []
    for ds in DS:
        x = L[L.dataset == ds]
        rows.append(dict(dataset=ds, families=len(x), recall_seen=x.recall_seen_at_alpha.median(),
                         recall_unseen=x.recall_unseen_at_alpha.median(),
                         below_half=int((x.recall_unseen_at_alpha < .5).sum()),
                         cm_median=x["cm_0.05"].median(), cm_max=x["cm_0.05"].max(),
                         auroc_det=x.auroc_det.median(), auroc_nov=x.auroc_nov.median()))
    md_table(pd.DataFrame(rows), "os_lgbm_by_dataset.md")
    N["os_lgbm_by_dataset"] = [{k: (rd(v) if not isinstance(v, str) else v) for k, v in r.items()} for r in rows]
    N["os_lgbm_overall"] = dict(families=len(L), below_half=int((L.recall_unseen_at_alpha < .5).sum()),
                                confmiss_over_half=int((L["cm_0.05"] > .5).sum()),
                                median_recall=rd(L.recall_unseen_at_alpha.median()),
                                median_recall_seen=rd(L.recall_seen_at_alpha.median()),
                                stream_pi_min=rd(L.stream_pi.min(), 4),
                                n_stream_below_1pct=int((L.stream_pi < 0.0095).sum()))
    # per-family LGBM table for the appendix
    pf = L[["dataset", "held_out", "n_unseen", "recall_unseen_at_alpha", "cm_0.05", "auroc_det", "auroc_nov", "stream_pi"]]
    md_table(pf.sort_values(["dataset", "held_out"]), "os_lgbm_per_family.md")
    # confident-miss cutoff sensitivity (all models)
    rows = []
    for m in avail:
        x = fam[fam.model == m]
        for c in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            rows.append(dict(model=m, cutoff=c, over_half=int((x[f"cm_{c}"] > .5).sum()),
                             over_quarter=int((x[f"cm_{c}"] > .25).sum()),
                             median=x[f"cm_{c}"].median()))
    C = pd.DataFrame(rows)
    md_table(C, "os_confmiss_cutoffs.md")
    N["os_cm_cutoffs"] = C.round(3).to_dict("records")
    # seed variability of the headline count
    per_seed = d[d.model == "LGBM"].groupby("seed").apply(lambda g: int((g.recall_unseen_at_alpha < .5).sum()))
    N["os_lgbm_below_half_per_seed"] = per_seed.to_dict()
    # validity of the conformal bound with benign flows of the same network
    v = pd.read_csv(os.path.join(RES, "os_validity.csv"))
    N["os_validity"] = dict(fpr_pB_mean=rd(v.fpr_pB.mean(), 4), fpr_pB_q90=rd(q(v.fpr_pB, .9), 4),
                            fpr_pB_max=rd(v.fpr_pB.max(), 4), share_above_alpha=rd((v.fpr_pB > 0.01).mean()),
                            fpr_pdet_mean=rd(v.fpr_pdet.mean(), 4))


def os_priorities():
    p = pd.read_csv(os.path.join(RES, "os_priorities.csv"))
    det = pd.read_csv(os.path.join(RES, "os_detection_all.csv"))
    p = p[p.model == "LGBM"]
    det = det[det.model == "LGBM"]
    # grouping defined on seeds 0-2, gains evaluated on seeds 3-4 (and vice versa), to avoid
    # selecting families on the same runs used to measure the gain (regression to the mean)
    g_a = det[det.seed <= 2].groupby(["dataset", "held_out"]).recall_unseen_at_alpha.mean() < .5
    g_b = det[det.seed >= 3].groupby(["dataset", "held_out"]).recall_unseen_at_alpha.mean() < .5
    g_all = det.groupby(["dataset", "held_out"]).recall_unseen_at_alpha.mean() < .5
    N["os_group_agreement"] = dict(weak_a=int(g_a.sum()), weak_b=int(g_b.sum()),
                                   agree=int((g_a == g_b).sum()), total=int(len(g_a)))
    p = p.set_index(["dataset", "held_out"])
    p["weak_cf"] = np.where(p.seed >= 3, g_a.reindex(p.index).values, g_b.reindex(p.index).values)
    p["weak_all"] = g_all.reindex(p.index).values
    p = p.reset_index()
    prios = ["detector", "novelty", "combined", "fisher", "wfisher0.6", "wfisher0.7", "wfisher0.8", "wfisher0.9",
             "stouffer", "stacked", "borda", "uncertainty", "split0.1", "split0.25", "split0.5", "proxy_switch"]
    rows = []
    for c in [0.01, 0.02, 0.05]:
        x = p[p.budget == c]
        famv = x.groupby(["priority", "dataset", "held_out"]).recall.mean()
        famw = x.groupby(["priority", "dataset", "held_out", "weak_cf"]).recall.mean()
        for pr in prios:
            if pr not in famv.index.get_level_values(0):
                continue
            r = dict(budget=c, priority=pr, all=famv[pr].mean())
            for wk, lab in [(False, "caught"), (True, "missed")]:
                sub = x[(x.priority == pr) & (x.weak_cf == wk)].groupby(["dataset", "held_out"]).recall.mean()
                r[lab] = sub.mean()
            rows.append(r)
            # paired comparisons with the detector, family level, overall and per group
            if pr != "detector":
                for wk, lab in [(None, "all"), (False, "caught"), (True, "missed")]:
                    xx = x if wk is None else x[x.weak_cf == wk]
                    a = xx[xx.priority == pr].groupby(["dataset", "held_out"]).recall.mean()
                    b = xx[xx.priority == "detector"].groupby(["dataset", "held_out"]).recall.mean()
                    N[f"os_paired_{pr}_vs_det_{lab}_b{c}"] = paired(a, b)
    T = pd.DataFrame(rows)
    md_table(T, "os_priorities.md")
    N["os_priorities"] = T.round(4).to_dict("records")
    N["os_n_weak_cf"] = dict(missed=int(p[(p.budget == .01) & (p.priority == "detector") & p.weak_cf].groupby(["dataset", "held_out"]).ngroups),
                             caught=int(p[(p.budget == .01) & (p.priority == "detector") & ~p.weak_cf.astype(bool)].groupby(["dataset", "held_out"]).ngroups))
    # same grouping as the original paper (all seeds) for comparison
    x = p[(p.budget == .01)]
    for pr in ["detector", "fisher"]:
        for wk, lab in [(False, "caught"), (True, "missed")]:
            N[f"os_insample_{pr}_{lab}_b0.01"] = rd(x[(x.priority == pr) & (x.weak_all == wk)].groupby(["dataset", "held_out"]).recall.mean().mean(), 4)
    # break-even probability that a new campaign is one the detector misses
    be = []
    for c in [0.01, 0.02, 0.05]:
        t = T[T.budget == c].set_index("priority")
        for pr in ["fisher", "wfisher0.7", "wfisher0.8", "stouffer", "borda", "combined", "split0.1", "split0.25"]:
            if pr not in t.index:
                continue
            gm = t.loc[pr, "missed"] - t.loc["detector", "missed"]
            gc = t.loc[pr, "caught"] - t.loc["detector", "caught"]
            qstar = (-gc) / (gm - gc) if (gm - gc) != 0 and gc < 0 < gm else (0.0 if gc >= 0 and gm >= 0 else np.nan)
            be.append(dict(budget=c, priority=pr, gain_missed=gm, gain_caught=gc, breakeven_q=qstar))
    B = pd.DataFrame(be)
    md_table(B, "os_breakeven.md")
    N["os_breakeven"] = B.round(4).to_dict("records")
    # weight of the weighted Fisher score chosen without looking at the evaluated dataset:
    # for each dataset, pick w maximising mean family recall on the other four datasets
    wmap = {"fisher": 0.5, "wfisher0.6": 0.6, "wfisher0.7": 0.7, "wfisher0.8": 0.8, "wfisher0.9": 0.9, "detector": 1.0}
    nest = []
    for c in [0.01, 0.02, 0.05]:
        x = p[(p.budget == c) & p.priority.isin(wmap)]
        fr = x.groupby(["dataset", "held_out", "priority"]).recall.mean().reset_index()
        chosen = {}
        picked = []
        for ds in DS:
            other = fr[fr.dataset != ds].groupby("priority").recall.mean()
            best = other.idxmax()
            chosen[ds] = wmap[best]
            picked.append(fr[(fr.dataset == ds) & (fr.priority == best)].set_index(["dataset", "held_out"]).recall)
        sel = pd.concat(picked)
        detr = fr[fr.priority == "detector"].set_index(["dataset", "held_out"]).recall
        r = paired(sel, detr)
        nest.append(dict(budget=c, chosen_w=json.dumps(chosen), recall_nested=sel.mean(), recall_det=detr.mean(),
                         diff=r["mean"], lo=r["lo"], hi=r["hi"], p=r["p"]))
        if c == 0.01:
            # the same comparisons without the families whose stream could not reach 1% prevalence
            low = det.groupby(["dataset", "held_out"]).stream_pi.mean() < 0.0095
            keep = lambda v: v[~low.reindex(v.index).fillna(False).values]
            N["os_low_pi_families"] = int(low.sum())
            N["os_nested_excl_lowpi_b0.01"] = paired(keep(sel), keep(detr))
            fi = fr[fr.priority == "fisher"].set_index(["dataset", "held_out"]).recall
            N["os_fisher_excl_lowpi_b0.01"] = paired(keep(fi), keep(detr))
            N["os_fisher_all_b0.01_check"] = paired(fi, detr)
    NT = pd.DataFrame(nest)
    md_table(NT, "os_nested_weight.md")
    N["os_nested_weight"] = NT.to_dict("records")
    # label-free switching proxy: how often does it fire on missed vs caught families?
    s = p[(p.priority == "proxy_switch") & (p.budget == .01)]
    N["os_proxy_fire_rate"] = dict(missed=rd(s[s.weak_cf.astype(bool)].proxy_fired.mean()),
                                   caught=rd(s[~s.weak_cf.astype(bool)].proxy_fired.mean()))


def os_budget():
    w = pd.read_csv(os.path.join(RES, "os_warmup.csv"))
    w = w[w.model == "LGBM"]
    if "rule" in w:
        src = w[w.rule == "source"].groupby(["priority", "budget"]).load.quantile([.1, .5, .9]).unstack()
        src.columns = [str(c) for c in src.columns]
        src = src.reset_index()
        md_table(src, "os_source_threshold_load.md")
        N["os_source_load"] = src.round(4).to_dict("records")
        w = w[w.rule == "warmup"]
    rows = []
    for (pr, c, wf), x in w.groupby(["priority", "budget", "warmup"]):
        sd_pred = np.sqrt((1 - c) / c * (1 / x.n_warm + 1 / x.n_eval))
        inside = ((x.load >= 1 - 1.2816 * sd_pred) & (x.load <= 1 + 1.2816 * sd_pred)).mean()
        rows.append(dict(priority=pr, budget=c, warmup=wf, n_warm_median=x.n_warm.median(),
                         load_median=x.load.median(), load_q10=q(x.load, .1), load_q90=q(x.load, .9),
                         pred_sd_median=np.median(sd_pred), share_in_pred_80band=inside,
                         recall=x.recall.mean()))
    W = pd.DataFrame(rows)
    md_table(W, "os_warmup.md")
    N["os_warmup"] = W.round(4).to_dict("records")
    ns = pd.read_csv(os.path.join(RES, "os_nonstationary.csv"))
    ns = ns[ns.model == "LGBM"]
    S = ns.groupby(["scenario", "budget", "rule"]).agg(load_median=("load", "median"),
                                                     load_q10=("load", lambda v: q(v, .1)),
                                                     load_q90=("load", lambda v: q(v, .9)),
                                                     recall=("recall", "mean")).reset_index()
    md_table(S, "os_nonstationary.md")
    N["os_nonstationary"] = S.round(4).to_dict("records")
    pv = pd.read_csv(os.path.join(RES, "os_prevalence.csv"))
    pv = pv[pv.model == "LGBM"]
    P = pv.groupby(["target_pi", "budget", "priority"]).agg(real_pi=("real_pi", "mean"), load_median=("load", "median"),
                                                          load_q90=("load", lambda v: q(v, .9)),
                                                          recall=("recall", "mean"),
                                                          precision=("precision", "mean")).reset_index()
    md_table(P, "os_prevalence.md")
    N["os_prevalence"] = P.round(4).to_dict("records")
    cq = pd.read_csv(os.path.join(RES, "os_queue_composition.csv"))
    cq = cq[cq.model == "LGBM"]
    Q = cq.groupby(["budget", "priority"]).agg(load_median=("load", "median"), recall=("recall", "mean"),
                                               precision=("precision", "mean"), fa_per_tp=("fa_per_tp", "median"),
                                               fa_novelty_share=("fa_novelty_share", "mean")).reset_index()
    md_table(Q, "os_queue.md")
    N["os_queue"] = Q.round(4).to_dict("records")


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["os"]
    if "os" in parts:
        os_summary(); os_priorities(); os_budget()
    if "xd" in parts:
        from revision_summary_xd import xd_all
        xd_all(N)
    if "extra" in parts:
        from revision_summary_extra import extra_all
        extra_all(N)
    path = os.path.join(RES, "revision_numbers.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(N)
    json.dump(old, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("ok", len(N))

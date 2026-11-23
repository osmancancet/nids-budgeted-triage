"""Named values for the manuscript text, computed from results/revision/*.json|csv.

  python revision_macros.py            # writes results/revision/macros.json
  python revision_macros.py fill a.md  # prints a.md with {{name}} replaced; fails on unknown names
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "revision")
ORIG = os.path.join(ROOT, "results")
N = json.load(open(os.path.join(RES, "revision_numbers.json")))
M = {}


def f(x, k=2):
    return f"{x:.{k}f}"


def pct(x, k=0):
    return f"{100 * x:.{k}f}%"


def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def ci(d, k=3):
    return f"{d['mean']:+.{k}f} (95% CI {d['lo']:+.{k}f} to {d['hi']:+.{k}f}; {pv(d['p'])}, {d['n']} families)"


def ci_pairs(d, k=3):
    return f"{d['mean']:+.{k}f} (95% CI {d['lo']:+.{k}f} to {d['hi']:+.{k}f}; {pv(d['p'])}, {d['n']} pairs)"


def rec(lst, **kw):
    for r in lst:
        if all((abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v) for k, v in kw.items()):
            return r
    raise KeyError(kw)


def words(n):
    n = int(round(n))
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    return w[n] if n < len(w) else str(n)


# ------------------------------------------------------------------ OS
def os_macros():
    o = N["os_lgbm_overall"]
    M["os_n_fam"] = str(o["families"])
    M["os_rec_unseen_med"] = f(o["median_recall"], 3)
    M["os_rec_seen_med"] = f(o["median_recall_seen"], 3)
    M["os_n_below_half"] = str(o["below_half"])
    M["os_n_below_half_all"] = str(o["below_half"])
    M["os_pct_below_half"] = pct(o["below_half"] / o["families"])
    M["os_n_cm"] = str(o["confmiss_over_half"])
    _d = pd.read_csv(os.path.join(RES, "os_detection_all.csv"))
    _g = _d[_d.model == "LGBM"].groupby(["dataset", "held_out"]).mean(numeric_only=True)
    M["os_n_cm_and_missed"] = words(int(((_g["cm_0.05"] > .5) & (_g.recall_unseen_at_alpha < .5)).sum()))
    M["os_n_cm_and_missed_cap"] = M["os_n_cm_and_missed"].capitalize()
    _m = _g[_g.recall_unseen_at_alpha < .5]
    M["os_missed_auc_rng"] = f"{_m.auroc_det.min():.2f} to {_m.auroc_det.max():.2f}"
    M["os_missed_auc_hi"] = words(int((_m.auroc_det >= .9).sum()))
    M["os_missed_auc_lo"] = words(int((_m.auroc_det < .7).sum()))
    M["os_missed_auc_med"] = f(_m.auroc_det.median(), 2)
    _odd = _g[(_g["cm_0.05"] > .5) & (_g.recall_unseen_at_alpha >= .5)]
    assert list(_odd.index) == [("CIC-IDS2017", "FTP-Patator")], list(_odd.index)  # named in Section 4.4
    ps = list(N["os_lgbm_below_half_per_seed"].values())
    M["os_below_half_seed_rng"] = f"{min(ps)} to {max(ps)}"
    M["os_n_stream_below_1pct"] = words(o["n_stream_below_1pct"])
    M["os_stream_pi_min"] = pct(o["stream_pi_min"], 2)
    cm = {r["cutoff"]: r for r in N["os_cm_cutoffs"] if r["model"] == "LGBM"}
    M["os_cm_001"], M["os_cm_05"] = str(cm[0.01]["over_half"]), str(cm[0.5]["over_half"])
    M["os_cm_quarter_005"] = str(cm[0.05]["over_quarter"])
    bd = {r["dataset"]: r for r in N["os_lgbm_by_dataset"]}
    M["os_aci_rec"] = f(bd["ACI-IoT-2023"]["recall_unseen"], 3)
    M["os_aci_below"] = words(bd["ACI-IoT-2023"]["below_half"])
    M["os_aci_cm"] = f(bd["ACI-IoT-2023"]["cm_median"], 2)
    au = [bd[d]["auroc_det"] for d in bd]
    M["os_auroc_det_rng"] = f"{min(au):.2f} to {max(au):.3f}"
    v = N["os_validity"]
    M["os_val_pb_mean"], M["os_val_pb_q90"], M["os_val_pb_max"] = f(v["fpr_pB_mean"], 4), f(v["fpr_pB_q90"], 4), f(v["fpr_pB_max"], 4)
    M["os_val_share_above"] = pct(v["share_above_alpha"])
    # per-model rows (all models available)
    for m in N.get("os_models_available", []):
        r = N.get(f"os_{m}")
        if r:
            M[f"os_{m}_below_half"] = str(int(r["below_half"]))
            M[f"os_{m}_cm"] = str(int(r["confmiss_over_half"]))
            M[f"os_{m}_rec_med"] = f(r["recall_unseen_median"], 3)
    M["os_missed_by_all"] = str(N["os_missed_by_all_models"])
    M["os_missed_by_any"] = str(N["os_missed_by_any_model"])
    M["os_n_models"] = words(len(N["os_models_available"]))
    # priorities
    P = N["os_priorities"]
    for c in [0.01, 0.02, 0.05]:
        cc = int(round(100 * c))
        for pr in ["detector", "novelty", "combined", "fisher", "wfisher0.7", "stouffer", "stacked", "borda",
                   "uncertainty", "split0.1", "split0.25", "split0.5", "proxy_switch"]:
            r = rec(P, budget=c, priority=pr)
            key = pr.replace(".", "")
            for g in ["all", "caught", "missed"]:
                M[f"os_r_{key}_{g}_b{cc}"] = f(r[g], 3)
            for g in ["all", "caught", "missed"]:
                k = f"os_paired_{pr}_vs_det_{g}_b{c}"
                if k in N:
                    M[f"os_d_{key}_{g}_b{cc}"] = ci(N[k])
                    M[f"os_dm_{key}_{g}_b{cc}"] = f"{N[k]['mean']:+.3f}"
                    M[f"os_dp_{key}_{g}_b{cc}"] = pv(N[k]["p"])
    M["os_n_missed_cf"] = str(N["os_n_weak_cf"]["missed"])
    M["os_n_caught_cf"] = str(N["os_n_weak_cf"]["caught"])
    ga = N["os_group_agreement"]
    M["os_cf_agree"] = f"{ga['agree']} of {ga['total']}"
    M["os_ins_det_missed"] = f(N["os_insample_detector_missed_b0.01"], 3)
    M["os_ins_fis_missed"] = f(N["os_insample_fisher_missed_b0.01"], 3)
    pf = N["os_proxy_fire_rate"]
    M["os_proxy_missed"], M["os_proxy_caught"] = pct(pf["missed"], 1), pct(pf["caught"], 1)
    B = N["os_breakeven"]
    for c in [0.01, 0.02]:
        cc = int(round(100 * c))
        for pr in ["fisher", "stouffer", "borda"]:
            M[f"os_be_{pr}_b{cc}"] = f(rec(B, budget=c, priority=pr)["breakeven_q"], 2)
    for c in [0.01, 0.02, 0.05]:
        cc = int(round(100 * c))
        r = rec(N["os_nested_weight"], budget=c)
        ch = sorted(set(json.loads(r["chosen_w"]).values()))
        M[f"os_nest_w_b{cc}"] = " or ".join(f(x, 1) for x in ch)
        M[f"os_nest_d_b{cc}"] = ci(dict(mean=r["diff"], lo=r["lo"], hi=r["hi"], p=r["p"], n=int(M["os_n_fam"])))
        M[f"os_nest_r_b{cc}"] = f(r["recall_nested"], 3)
    # warm-up sensitivity
    W = N["os_warmup"]
    for wf in [0.01, 0.2, 0.5]:
        r = rec(W, priority="fisher", budget=0.01, warmup=wf)
        k = int(round(100 * wf))
        M[f"os_wu_q10_{k}"], M[f"os_wu_q90_{k}"] = f(r["load_q10"]), f(r["load_q90"])
        M[f"os_wu_n_{k}"] = f"{int(r['n_warm_median']):,}"
        M[f"os_wu_band_{k}"] = pct(r["share_in_pred_80band"])
        M[f"os_wu_sd_{k}"] = f(r["pred_sd_median"])
    bands = [r["share_in_pred_80band"] for r in W]
    M["os_wu_band_rng"] = f"{pct(min(bands))} to {pct(max(bands))}"
    # non-stationary
    S = N["os_nonstationary"]
    for sc, sk in [("burst_in_warmup", "burst"), ("late_campaign", "late")]:
        for rule in ["warmup", "online", "rolling"]:
            r = rec(S, scenario=sc, budget=0.01, rule=rule)
            M[f"os_ns_{sk}_{rule}_load"] = f(r["load_median"])
            M[f"os_ns_{sk}_{rule}_rec"] = f(r["recall"], 2)
    # prevalence
    Pv = N["os_prevalence"]
    for pi in [0.001, 0.01, 0.05]:
        r = rec(Pv, target_pi=pi, budget=0.01, priority="fisher")
        k = str(pi).replace(".", "")
        M[f"os_prev_prec_{k}"] = f(r["precision"], 3)
        M[f"os_prev_load_{k}"] = f(r["load_median"])
    loads = [r["load_median"] for r in Pv]
    M["os_prev_load_rng"] = f"{min(loads):.2f} to {max(loads):.2f}"
    # queue composition
    Q = N["os_queue"]
    for pr in ["detector", "fisher"]:
        r = rec(Q, budget=0.01, priority=pr)
        M[f"os_q_{pr}_prec"] = f(r["precision"], 2)
        M[f"os_q_{pr}_fatp"] = f((1 - r["precision"]) / r["precision"], 2)  # implied by the mean precision
        M[f"os_q_{pr}_novshare"] = pct(r["fa_novelty_share"])
    for c in [0.02, 0.05]:
        r = rec(Q, budget=c, priority="fisher")
        M[f"os_q_fisher_prec_b{int(round(100 * c))}"] = f(r["precision"], 2)


# ------------------------------------------------------------------ XD
def xd_macros():
    R = N["xd_reconcile"]
    rg = lambda v, k=2: f"{v[0]:.{k}f} to {v[1]:.{k}f}"
    M["xd_mm_auroc"] = rg(R["model_mean_over_pairs"]["auroc"])
    M["xd_mm_ece_raw"] = rg(R["model_mean_over_pairs"]["ece_raw"])
    M["xd_mm_ece_iso"] = rg(R["model_mean_over_pairs"]["ece_iso"])
    M["xd_mm_fprinf"] = rg(R["model_mean_over_pairs"]["fpr_inflation"], 1)
    M["xd_mm_prec"] = rg(R["model_mean_over_pairs"]["prec_at_pi0.01"], 3)
    M["xd_all_ece_raw"] = rg(R["all"]["ece_raw"])
    M["xd_all_ece_iso"] = rg(R["all"]["ece_iso"])
    M["xd_all_fprinf_max"] = f(R["all"]["fpr_inflation"][1], 0)
    M["xd_run_auroc"] = rg(R["auroc_run_range"])
    M["id_auroc"] = rg(R["in_domain"]["auroc"], 3)
    M["id_ece_raw_max"] = f(R["in_domain"]["ece_raw"][1], 3)
    M["id_ece_iso_max"] = f(R["in_domain"]["ece_iso"][1], 3)
    M["id_fprinf"] = rg(R["in_domain"]["fpr_inflation"])
    M["id_prec"] = rg(R["in_domain"]["prec_at_pi0.01"])
    for m in ["LGBM", "RF", "MLP", "LR"]:
        M[f"xd_{m}_fprinf"] = rg(R[m]["fpr_inflation"], 1)
        M[f"xd_{m}_ece_iso"] = rg(R[m]["ece_iso"])
    bs = R["lgbm_by_source"]
    M["xd_flood_fprinf"] = f"{min(bs['CIC-IDS2017']['fpr_inflation'][0], bs['CSE-CIC-IDS2018']['fpr_inflation'][0]):.1f} to " \
                           f"{max(bs['CIC-IDS2017']['fpr_inflation'][1], bs['CSE-CIC-IDS2018']['fpr_inflation'][1]):.1f}"
    M["xd_silent_fprinf"] = rg(bs["ACI-IoT-2023"]["fpr_inflation"])
    M["xd_silent_tpr"] = rg(bs["ACI-IoT-2023"]["tpr_at_alpha"], 3)
    for p in R["lgbm_pairs"]:
        k = (p["source"][:5] + "_" + p["target"][:5]).replace("-", "")
        M[f"xd_auroc_{k}"] = f(p["auroc"], 2)
        M[f"xd_fprinf_{k}"] = f(p["fpr_inflation"], 1)
    L = N["xd_source_load"]
    for pr in ["detector", "fisher", "novelty"]:
        for c in [0.01, 0.02, 0.05]:
            r = rec(L, setting="cross-dataset", priority=pr, budget=c)
            cc = int(round(100 * c))
            M[f"xd_srcload_{pr}_b{cc}"] = f"{r['0.5']:.2f} [{r['0.1']:.2f}, {r['0.9']:.2f}]"
            M[f"xd_srcload_{pr}_med_b{cc}"] = f(r["0.5"], 2)
    M["xd_srcload_cap_b1"] = f(1 / 0.01, 0)
    if "xd_nonstationary" in N:
        S = N["xd_nonstationary"]
        for reg, rk in [("In-domain", "id"), ("Cross-dataset", "xd")]:
            for sc, sk in [("burst_in_warmup", "burst"), ("late_campaign", "late")]:
                for rule in ["warmup", "online", "rolling"]:
                    M[f"{rk}_ns_{sk}_{rule}_load"] = f(rec(S, regime=reg, scenario=sc, budget=0.01, rule=rule)["load_med"])
    if "xd_warmup" in N:
        W = N["xd_warmup"]
        for reg, rk in [("In-domain", "id"), ("Cross-dataset", "xd")]:
            r = rec(W, regime=reg, priority="fisher", budget=0.01, warmup=0.2)
            M[f"{rk}_wu_load"] = f"{r['0.5']:.2f} [{r['0.1']:.2f}, {r['0.9']:.2f}]"
    if "xd_queue" in N:
        r = rec(N["xd_queue"], regime="Cross-dataset", budget=0.01, priority="fisher")
        M["xd_q_fisher_prec"] = f(r["precision"], 3)
        M["xd_q_fisher_fatp"] = f((1 - r["precision"]) / r["precision"], 0)
        r = rec(N["xd_queue"], regime="In-domain", budget=0.01, priority="detector")
        M["id_q_detector_prec"] = f(r["precision"], 2)
    if "xd_repro" in N:
        M["xd_repro_n"] = str(N["xd_repro"]["n"])
        M["xd_repro_max"] = f"{N['xd_repro']['max_abs_auroc']:.1e}"
        M["xd_repro_trees"] = f"{N['xd_repro']['max_abs_auroc_trees']:.0e}"
        M["xd_repro_noff"] = str(N["xd_repro"]["n_off"])
        M["xd_repro_fpr"] = f"{N['xd_repro']['max_abs_fprinf']:.4f}"

def misc_macros():
    z = json.load(open(os.path.join(RES, "zero_features.json")))
    M["n_zero_feat_cic17"] = words(len(z["CIC-IDS2017"]))
    M["n_zero_feat_aci"] = words(len(z["ACI-IoT-2023"]))
    qb = rec(N["os_breakeven"], budget=0.01, priority="fisher")["breakeven_q"]
    M["be_one_in"] = words(int(round(1 / qb)))
    bd = {r["dataset"]: r for r in N["os_lgbm_by_dataset"]}
    M["os_aci_auroc_nov"] = f(bd["ACI-IoT-2023"]["auroc_nov"], 2)
    r = rec(N["os_nested_weight"], budget=0.01)
    M["os_nest_dm_b1"] = f"{r['diff']:.3f}"
    R = N["xd_reconcile"]
    M["xd_LGBM_fprinf_max"] = f(R["LGBM"]["fpr_inflation"][1], 0)
    M["xd_silent_tpr_pct"] = pct(np.mean(R["lgbm_by_source"]["ACI-IoT-2023"]["tpr_at_alpha"]))
    if "os_source_load" in N:
        r = rec(N["os_source_load"], priority="fisher", budget=0.01)
        M["os_srcload_fisher_med_b1"] = f(r["0.5"])
        M["os_srcload_fisher_b1"] = f"{r['0.5']:.2f} [{r['0.1']:.2f}, {r['0.9']:.2f}]"
    models = [m for m in N.get("os_models_available", []) if m != "LGBM" and f"os_{m}" in N]
    if len(models) == 3:
        names = {"RF": "random forest", "MLP": "the MLP", "LR": "logistic regression"}
        M["os_other_models_below_half"] = ", ".join(f"{int(N['os_' + m]['below_half'])} ({names[m]})" for m in models)
    if "xd_validity" in N:
        v = [r for r in N["xd_validity"] if r["regime"] == "Cross-dataset"]
        M["xd_val_pb_max"] = f(max(r["fpr_pB_max"] for r in v), 3)
    if "xd_priorities" in N:
        P = N["xd_priorities"]
        M["xd_r_detector_in_b1"] = f(rec(P, regime="In-domain", budget=0.01)["detector"], 3)
        r = rec(P, regime="Cross-dataset", budget=0.05)
        best = max((v, k) for k, v in r.items() if k not in ("regime", "budget") and v == v)
        M["xd_best_cross_b5"] = f(best[0], 3)

def extra_macros():
    def get(name):
        p = os.path.join(RES, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    t, s = get("extra_temporal.csv"), get("extra_temporal_stream.csv")
    if t is not None:
        g = t.groupby("protocol").mean(numeric_only=True)
        for pr, k in [("T1-benign-time", "t1"), ("T1-random", "t1r"), ("T2-day", "t2"), ("T2-random", "t2r")]:
            M[f"{k}_auroc"], M[f"{k}_ece"] = f(g.loc[pr, "auroc"], 2), f(g.loc[pr, "ece_iso"], 3)
            M[f"{k}_fprinf"], M[f"{k}_tpr"] = f(g.loc[pr, "fpr_inflation"], 1), f(g.loc[pr, "tpr"], 2)
        M["t1_fprinf_round"] = str(int(round(g.loc["T1-benign-time", "fpr_inflation"])))
        z = s[(s.priority == "fisher") & (s.budget == 0.01)].groupby(["protocol", "rule"]).mean(numeric_only=True)
        for rule in ["source", "warmup", "online", "rolling"]:
            r = z.loc[("T1-benign-time", rule)]
            M[f"t1_{rule}_load"] = f(r.load)
            M[f"t1_{rule}_min10"], M[f"t1_{rule}_max10"] = f(r.load_min10), f(r.load_max10)
            for pr, k in [("T1-random", "t1r"), ("T2-day", "t2"), ("T2-random", "t2r")]:
                M[f"{k}_{rule}_load"] = f(z.loc[(pr, rule)].load)
        dev = {rule: max(abs(z.loc[(pr, rule)].load - 1) for pr in ["T1-benign-time", "T1-random", "T2-day", "T2-random"])
               for rule in ["source", "warmup", "online", "rolling"]}
        best = min(dev, key=dev.get)
        others = min(v for k, v in dev.items() if k != best)
        M["tmp_online_maxdev"] = pct(dev["online"])
        if best == "online":
            M["fig6_takeaway"] = (f"The online update is the only rule whose overall load stays within "
                                  f"{pct(dev['online'])} of the plan on all four streams; every other rule misses it "
                                  f"by at least {pct(others)} on one stream.")
        else:
            M["fig6_takeaway"] = f"The rule closest to plan on all four streams is the {best} rule."
    fe = get("extra_features.csv")
    if fe is not None and "kind" in fe:
        dom = fe[fe.kind == "benign_domain_auc"].groupby(["features", "source", "target"]).auroc.mean()
        a, b = dom.loc["all"], dom.loc["robust"]
        d1, d2 = N["features_paired_auroc"], N["features_paired_logfprinf"]
        M["feat_d_auroc"] = f"{d1['mean']:+.3f} on average (95% CI {d1['lo']:+.3f} to {d1['hi']:+.3f}, six pairs)"
        M["feat_d_logfpr"] = f"{d2['mean']:+.2f} (95% CI {d2['lo']:+.2f} to {d2['hi']:+.2f})"
        rng_ = lambda v: f"{v.min():.3f}" if f"{v.min():.3f}" == f"{v.max():.3f}" else f"{v.min():.3f} to {v.max():.3f}"
        M["dom_auc_rng"], M["dom_auc_rob_rng"] = rng_(a), rng_(b)
        fx = {r["features"]: r for r in N["features_xd"]}
        fa, fr = fx["all"]["fpr_inflation_mean"], fx["robust"]["fpr_inflation_mean"]
        ranking_same = d1["lo"] < 0 < d1["hi"]
        fpr_lower = d2["hi"] < 0
        M["feat_fpr_all"], M["feat_fpr_rob"] = f(fa, 1), f(fr, 1)
        if ranking_same and fpr_lower:
            M["feat_sentence"] = (f"The tool-stable features thus lower the mean false-alarm inflation from {f(fa, 1)} to "
                                  f"{f(fr, 1)} times nominal but do not restore the ranking, and benign traffic stays "
                                  "almost perfectly separable, which points to differences between the networks, "
                                  "rather than the extractor, as the main cause of the ranking loss.")
            M["feat_disc"] = ("Restricting the detector to features computed identically by all CICFlowMeter versions "
                              "reduced the false-alarm flood but did not restore the ranking (Section 4.7).")
            M["feat_lim"] = ("Restricting the features to those computed identically across CICFlowMeter versions "
                             "reduced the false-alarm inflation but left AUROC and the separability of benign traffic "
                             "almost unchanged, which bounds the share of tooling shift without removing it")
            M["resp_feat"] = (f"With all features, the benign flows of two datasets are separable with an AUROC of "
                              f"{M['dom_auc_rng']}, and with the tool-stable features {M['dom_auc_rob_rng']}. Restricting "
                              f"the detector to these features changes cross-dataset AUROC by {M['feat_d_auroc']} and "
                              f"lowers the mean false-alarm inflation from {f(fa, 1)} to {f(fr, 1)} times nominal. Tooling "
                              "therefore explains part of the false-alarm flood but not the loss of ranking, which we "
                              "attribute mainly to differences between the networks; Section 5.4 states that public "
                              "data do not allow a complete separation.")
        else:
            raise KeyError("feature outcome needs new wording")
    gr = N.get("group_out_summary")
    if gr:
        M["og_n_groups"], M["og_n_below_half"] = str(gr["n_groups"]), str(gr["n_below_half"])
        M["og_median_recall"] = f(gr["median_recall"], 3)
    if "reference" in N:
        R = N["reference"]
        base = rec(R, n_ref=20000, contamination=0.0)
        _p = N["reference_auc_n1000_c0.0"]
        M["ref_auc_20k"], M["ref_auc_1k"], M["ref_n_common"] = f(_p["base"], 3), f(_p["ref"], 3), str(_p["n"])
        for key, n, c in [("1k", 1000, 0.0), ("50k", 50000, 0.0), ("c1", 20000, 0.01), ("c5", 20000, 0.05), ("c01", 20000, 0.001)]:
            k = f"reference_paired_n{n}_c{c}"
            if k in N:
                M[f"ref_d_{key}"] = ci(N[k])
        M["ref_c1_pct"] = "1%"
        M["ref_d_c01_short"] = f(abs(N["reference_paired_n20000_c0.001"]["mean"]), 3)
    if "subsample" in N:
        S = pd.DataFrame(N["subsample"])
        keep = all((g.rec_fisher > g.rec_det).nunique() == 1 for _, g in S.groupby(["dataset", "protocol"]))
        xd = S[S.protocol == "cross-dataset"]
        ton = S[(S.dataset == "ToN-IoT") & (S.protocol == "unseen")].set_index("benign_frac")
        cic = S[(S.dataset == "CIC-IDS2017") & (S.protocol == "unseen")]
        M["sub_lo_load"], M["sub_hi_load"] = f(S.load_warmup.min(), 2), f(S.load_warmup.max(), 2)
        if not keep:
            raise KeyError("subsample outcome needs new wording")
        M["sub_sentence"] = (
            "In every dataset and protocol, the order of the detector and the Fisher priority at a 1% budget is the "
            "same at all three benign fractions, and the warm-up load stays between "
            f"{M['sub_lo_load']} and {M['sub_hi_load']} times the budget. Cross-dataset AUROC of the CIC-IDS2017 models "
            f"stays between {f(xd.auroc.min(), 3)} and {f(xd.auroc.max(), 3)}, and Fisher recall on held-out "
            f"CIC-IDS2017 families between {f(cic.rec_fisher.min(), 3)} and {f(cic.rec_fisher.max(), 3)}. The one "
            "clear effect is on ToN-IoT, where Fisher recall on held-out families falls from "
            f"{f(ton.loc[1.0, 'rec_fisher'], 3)} with all benign training flows to {f(ton.loc[0.25, 'rec_fisher'], 3)} "
            "with a quarter of them")
        M["resp_sub"] = (
            "The ranking of the triage rules and the budget behaviour of the warm-up quantile were the same at all "
            "three fractions (Section 4.8, Appendix Table A12). Recall on held-out ToN-IoT families fell when we "
            "removed three quarters of the benign training flows, so the absolute recall values for the subsampled "
            "copies are, if anything, conservative; we state this in Section 4.8.")
    if "reference" in N and "reference_paired_n1000_c0.0" in N:
        M["resp_ref"] = (
            "Reducing the benign reference from 20,000 to 1,000 flows changes Fisher recall at a 1% budget by "
            f"{N['reference_paired_n1000_c0.0']['mean']:+.3f}, and increasing it to 50,000 flows by "
            f"{N['reference_paired_n50000_c0.0']['mean']:+.3f}. Contamination matters far more: with 0.1% of the "
            "reference drawn from the held-out family, recall changes by "
            f"{N['reference_paired_n20000_c0.001']['mean']:+.3f}, and with 1% by "
            f"{N['reference_paired_n20000_c0.01']['mean']:+.3f} (Appendix Table A13). Exact kNN cost grows linearly "
            "with the reference, and an approximate index removes most of that growth (Table 8).")
    if gr and "fisher_vs_det" in gr:
        M["og_d_fisher"] = ci(gr["fisher_vs_det"]).replace("families)", "groups)")
    c = get("extra_cost.csv")
    if c is not None:
        z = c[c.threads == 4]
        det = lambda m: z[(z.component == "detector") & (z.model == m)].us_per_flow.iloc[0]
        knn = lambda a, n: z[(z.component == "knn") & (z.model == a) & (z.n_ref == n)]
        M["cost_lgbm_4"] = f(det("LGBM"), 1)
        k20 = knn("brute", 20000).us_per_flow.min() if len(knn("brute", 20000)) else np.nan
        best20 = z[(z.component == "knn") & (z.n_ref == 20000) & (z.model != "hnsw")].us_per_flow.min()
        M["cost_knn20k_4"] = f(best20, 1)
        conf = z[z.component == "conformal+fisher"].us_per_flow.iloc[0]
        M["cost_conf_4"] = f(conf, 2)
        M["cost_knn100k_4"] = f(z[(z.component == "knn") & (z.n_ref == 100000) & (z.model != "hnsw")].us_per_flow.min(), 1)
        h = knn("hnsw", 100000)
        if len(h):
            M["cost_hnsw100k_4"], M["cost_hnsw100k_rec"] = f(h.us_per_flow.iloc[0], 1), f(h.recall_at5.iloc[0], 3)
        tot = det("LGBM") + best20 + conf
        M["cost_total_4"] = f(tot, 1)
        M["cost_flows_per_s"] = f"{int(round(1e6 / tot, -3)):,}"
        h20 = knn("hnsw", 20000)
        M["cost_hnsw20k_4"], M["cost_hnsw20k_rec"] = f(h20.us_per_flow.iloc[0], 1), f(h20.recall_at5.iloc[0], 3)
        M["cost_hnsw_rec_max"] = f(z[(z.component == "knn") & (z.model == "hnsw")].recall_at5.max(), 3)
        M["cost_hnsw_rec_min"] = f(z[(z.component == "knn") & (z.model == "hnsw")].recall_at5.min(), 3)
        M["cost_hnsw_rec_min_n"] = f"{int(z[(z.component == 'knn') & (z.model == 'hnsw')].sort_values('recall_at5').n_ref.iloc[0]):,}"
        toth = det("LGBM") + h20.us_per_flow.iloc[0] + conf
        M["cost_total_hnsw_4"] = f(toth, 1)
        M["cost_flows_per_s_hnsw"] = f"{int(round(1e6 / toth, -3)):,}"
        M["cost_knn_ratio"] = f(best20 / det("LGBM"), 0)
        M["resp_cost"] = (
            f"On four threads, LightGBM inference takes {M['cost_lgbm_4']} µs per flow, exact kNN scoring against the "
            f"20,000-flow reference {M['cost_knn20k_4']} µs and the conformal and Fisher step {M['cost_conf_4']} µs, so "
            "exact kNN search, not the detector, dominates the cost of the layer. It grows linearly with the reference "
            f"({M['cost_knn100k_4']} µs per flow at 100,000 flows). An approximate HNSW index brings it down to "
            f"{M['cost_hnsw20k_4']} µs at 20,000 flows and {M['cost_hnsw100k_4']} µs at 100,000 flows, with top-5 "
            f"neighbour recall between {M['cost_hnsw_rec_min']} and {M['cost_hnsw20k_rec']}; tree indexes are slower "
            "than brute-force search on these 77 features.")


def lowpi_macros():
    M["os_lowpi_n"] = words(N["os_low_pi_families"])
    d = N["os_nested_excl_lowpi_b0.01"]
    M["os_nest_excl_lowpi_b1"] = ci(d)


def adapt_macros():
    A = N["xd_adapt"]
    C = N["xd_adapt_curve"]
    for c, t in [(0.01, "b1"), (0.02, "b2"), (0.05, "b5")]:
        for v, k in [("base_fisher", "base"), ("base_detector", "basedet"), ("FB", "fb"), ("FB_noexplore", "fbnx"),
                     ("TR_wfisher", "tr"), ("TR_novelty", "trnov"), ("TN_wfisher", "tn"), ("TNTR_wfisher", "tntr")]:
            try:
                r = rec(A, budget=c, variant=v)
            except KeyError:
                continue
            M[f"ad_{k}_rec_{t}"] = f(r["recall"], 3)
            M[f"ad_{k}_prec_{t}"] = f(r["precision"], 3)
            M[f"ad_{k}_load_{t}"] = f(r["load"], 2)
            if v.startswith("FB"):
                M[f"ad_{k}_labels_{t}"] = f"{r['labels']:,.0f}"
        last = rec(C, budget=c, variant="FB", segment=4)
        first = rec(C, budget=c, variant="FB", segment=1)
        M[f"ad_fb_last_rec_{t}"] = f(last["recall"], 3)
        M[f"ad_fb_last_prec_{t}"] = f(last["precision"], 3)
        M[f"ad_fb_first_rec_{t}"] = f(first["recall"], 3)
        M[f"ad_fb_last_attlab_{t}"] = f"{last['n_attack_labels']:.0f}"
        M[f"ad_fb_last_lab_{t}"] = f"{last['n_labels']:,.0f}"
        for v, k in [("FB", "fb"), ("FB_noexplore", "fbnx"), ("TR_wfisher", "tr"), ("TR_novelty", "trnov"),
                     ("TN_wfisher", "tn"), ("TNTR_wfisher", "tntr")]:
            key = f"xd_adapt_{v}_vs_base_b{c}"
            if key in N:
                M[f"ad_d_{k}_{t}"] = ci_pairs(N[key])
                M[f"ad_dm_{k}_{t}"] = f"{N[key]['mean']:+.3f}"
                M[f"ad_dprec_{k}_{t}"] = ci_pairs(N[f"xd_adapt_{v}_vs_base_prec_b{c}"])
        w, n = N[f"xd_adapt_fb_runs_better_b{c}"]
        M[f"ad_fb_runs_{t}"] = f"{w} of {n}"
        lo, hi = N[f"xd_adapt_fb_minmax_pair_b{c}"]
        M[f"ad_fb_pairs_rng_{t}"] = f"{lo:.3f} to {hi:.3f}"
    P = pd.DataFrame(N["xd_adapt_pairs"])
    P = P[(P.budget == 0.01)]
    fb = P[P.variant == "FB"].set_index(["source", "target"]).recall
    nx = P[P.variant == "FB_noexplore"].set_index(["source", "target"]).recall
    worst = fb.idxmin()
    M["ad_worst_pair"] = f"{worst[0]} to {worst[1]}"
    M["ad_worst_rec_b1"] = f(fb.min(), 3)
    M["ad_worst_nx_rec_b1"] = f(nx[worst], 3)
    M["ad_nx_better_pairs"] = words((nx > fb).sum())


def ctrl_macros():
    """Controls for the feedback loop (revision_adapt_ctrl.py), paired with FB over the six pairs."""
    import glob
    from revision_summary import paired
    fs = sorted(glob.glob(os.path.join(RES, "xd_adapt_ctrl*.csv")))
    if not fs:
        return
    c = pd.concat([pd.read_csv(x) for x in fs])
    a = pd.read_csv(os.path.join(RES, "xd_adapt.csv"))
    a = a[(a.budget == 0.01) & (a.variant.isin(["FB", "base_fisher"]))]
    d = pd.concat([c, a])
    pair = d.groupby(["variant", "source", "target"])[["recall", "precision", "labels"]].mean()
    n_runs = d.groupby("variant").size()
    M["ctrl_n_runs"] = str(int(n_runs.get("RL", 0)))
    fb = pair.loc["FB"]
    if len(pair.loc["RL"]) < 6 if "RL" in pair.index.get_level_values(0) else True:
        return
    for v, k in [("RL", "rl"), ("US", "us"), ("FB_n10", "n10"), ("FB_n20", "n20")]:
        if v not in pair.index.get_level_values(0):
            continue
        g = pair.loc[v]
        M[f"ctrl_{k}_rec"] = f(g.recall.mean(), 3)
        M[f"ctrl_{k}_prec"] = f(g.precision.mean(), 3)
        M[f"ctrl_{k}_labels"] = f"{g.labels.mean():,.0f}"
        M[f"ctrl_fb_vs_{k}"] = ci_pairs(paired(fb.recall, g.recall))
        M[f"ctrl_fb_better_{k}"] = str(int((fb.recall > g.recall).sum()))
    M["ctrl_fb_labels"] = f"{fb.labels.mean():,.0f}"
    N["xd_adapt_ctrl"] = pair.reset_index().round(4).to_dict("records")


def build():
    os_macros()
    for mod in ("xd_macros", "misc_macros", "extra_macros", "adapt_macros", "lowpi_macros", "ctrl_macros"):
        fn = globals().get(mod)
        if fn:
            try:
                fn()
            except KeyError as e:
                print("skip", mod, e)
    json.dump(M, open(os.path.join(RES, "macros.json"), "w"), indent=1, ensure_ascii=False)
    return M


def fill(text, M):
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(M))
    if missing:
        raise SystemExit("unknown macros: " + ", ".join(missing))
    return re.sub(r"\{\{(\w+)\}\}", lambda m: M[m.group(1)], text)


if __name__ == "__main__":
    M = build()
    if len(sys.argv) > 2 and sys.argv[1] == "fill":
        for p in sys.argv[2:]:
            sys.stdout.write(fill(open(p).read(), M))
    else:
        print(len(M), "macros")

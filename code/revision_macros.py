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
        M[f"os_q_{pr}_prec"] = f(r["precision"], 3)
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
            xd_ = fe[(fe.kind != "benign_domain_auc") & (fe.source != fe.target)] if "kind" in fe else fe
            pp = xd_.groupby(["source", "target", "features"])[["auroc", "fpr_inflation", "tpr"]].mean().unstack()
            dA = pp["auroc"]["robust"] - pp["auroc"]["all"]
            top = dA.idxmax()
            rest = dA.drop(top)
            sil = pp[(pp["tpr"]["robust"] < 0.01)]
            M["feat_top_pair"] = f"{top[0]} to {top[1]}"
            M["feat_top_auc"] = f"{pp['auroc']['all'].loc[top]:.3f} to {pp['auroc']['robust'].loc[top]:.3f}"
            M["feat_rest_rng"] = f"{rest.min():+.3f} to {rest.max():+.3f}"
            M["feat_n_silent"] = words(len(sil))
            M["feat_silent_fpr"] = " and ".join(f"{v:.2f}" for v in sil["fpr_inflation"]["robust"])
            M["feat_silent_tpr"] = " and ".join(f"{v:.3f}" for v in sil["tpr"]["robust"])
            M["feat_sentence"] = (f"The tool-stable features lower the mean false-alarm inflation from {f(fa, 1)} to "
                                  f"{f(fr, 1)} times nominal, but part of this drop is detectors falling silent: in "
                                  f"{M['feat_n_silent']} pairs the inflation falls to {M['feat_silent_fpr']} while the "
                                  f"detection rate falls to {M['feat_silent_tpr']}. The ranking changes differently by "
                                  f"pair. For {M['feat_top_pair']}, AUROC rises from {M['feat_top_auc']}, so the extractor "
                                  f"explains much of the loss for that pair; on the other five pairs AUROC changes by "
                                  f"{M['feat_rest_rng']}, and benign traffic stays almost perfectly separable. The retained "
                                  "packet counts and lengths still depend on where the extractor ends a flow (timeouts and "
                                  "FIN/RST handling), so this residual separability can come from the networks or from "
                                  "remaining tooling differences, and the experiment does not tell the two apart.")
            M["feat_disc"] = ("Restricting the detector to features computed identically by all CICFlowMeter versions "
                              f"raised AUROC markedly for one pair ({M["feat_top_pair"]}, {M["feat_top_auc"]}) but not for the other five, and "
                              "part of its lower false-alarm rate came from detectors that stopped alerting (Section 4.7).")
            M["feat_lim"] = ("Restricting the features to those computed identically across CICFlowMeter versions "
                             f"raised AUROC for {M["feat_top_pair"]} but left AUROC and the separability of benign "
                             "traffic almost unchanged on the other five pairs. Because the retained packet counts and lengths "
                             "still depend on how the extractor ends a flow, this bounds only the tooling shift carried by "
                             "timing, flag and bulk features, not tooling shift as a whole")
            M["resp_feat"] = (f"With all features, the benign flows of two datasets are separable with an AUROC of "
                              f"{M['dom_auc_rng']}, and with the tool-stable features {M['dom_auc_rob_rng']}. Restricting "
                              f"the detector to these features changes cross-dataset AUROC by {M['feat_d_auroc']}; "
                              f"the mean is driven by {M['feat_top_pair']} ({M['feat_top_auc']}), and the other five "
                              f"pairs change by {M['feat_rest_rng']}. The mean false-alarm inflation falls from "
                              f"{f(fa, 1)} to {f(fr, 1)} times nominal, partly because {M['feat_n_silent']} detectors "
                              "stop alerting. Tooling therefore explains the ranking loss for one pair and part of the "
                              "false-alarm flood, and network differences the rest; Section 5.4 states that public "
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
        _raw = pd.read_csv(os.path.join(RES, "extra_subsample.csv"))
        _raw = _raw[(_raw.dataset == "ToN-IoT") & (_raw.protocol == "unseen")].groupby("benign_frac").rec_fisher.mean()
        ton = ton.assign(rec_fisher=_raw.reindex(ton.index).values)  # same aggregation as Table A12
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
            f"neighbour recall between {M['cost_hnsw_rec_min']} and {M['cost_hnsw_rec_max']}; tree indexes are slower "
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
    att = d.groupby("variant").n_attack_labels.mean()
    for v, k in [("RL", "rl"), ("US", "us"), ("FB_n10", "n10"), ("FB_n20", "n20"), ("FB_d1", "d1"), ("PU", "pu")]:
        if v not in pair.index.get_level_values(0):
            continue
        # pair with FB on the seeds that this variant has
        sv = d[d.variant == v][["source", "target", "seed"]]
        fbv = d[d.variant == "FB"].merge(sv, on=["source", "target", "seed"])
        fbp = fbv.groupby(["source", "target"]).recall.mean()
        g = d[d.variant == v].groupby(["source", "target"])[["recall", "precision", "labels"]].mean()
        if len(g) < 6:
            continue
        M[f"ctrl_{k}_rec"] = f(g.recall.mean(), 3)
        M[f"ctrl_{k}_prec"] = f(g.precision.mean(), 3)
        M[f"ctrl_{k}_labels"] = f"{g.labels.mean():,.0f}"
        M[f"ctrl_{k}_att"] = f"{att[v]:.0f}"
        M[f"ctrl_{k}_nseeds"] = str(sv.seed.nunique())
        M[f"ctrl_fb_vs_{k}"] = ci_pairs(paired(fbp, g.recall))
        M[f"ctrl_fb_better_{k}"] = str(int((fbp > g.recall).sum()))
    # label-free PU control, paired run by run with RL and FB on whatever runs exist
    if "PU" in pair.index.get_level_values(0):
        k3 = ["source", "target", "seed"]
        pu = d[d.variant == "PU"].set_index(k3)
        rl = d[d.variant == "RL"].set_index(k3).reindex(pu.index)
        fbr = d[d.variant == "FB"].set_index(k3).reindex(pu.index)
        base = d[d.variant == "base_fisher"].set_index(k3).reindex(pu.index)
        M["pu_nruns"] = str(len(pu))
        M["pu_npairs"] = words(pu.reset_index()[["source", "target"]].drop_duplicates().shape[0])
        M["pu_nseeds"] = words(pu.reset_index().seed.nunique())
        M["pu_rec"] = f(pu.recall.mean(), 3)
        M["pu_rl_rec"] = f(rl.recall.mean(), 3)
        M["pu_fb_rec"] = f(fbr.recall.mean(), 3)
        M["pu_base_rec"] = f(base.recall.mean(), 3)
        M["pu_att"] = f"{pu.n_attack_labels.mean():.0f}"
        M["pu_vs_rl"] = f"{(pu.recall - rl.recall).mean():+.3f}"
        M["pu_vs_base"] = f"{(pu.recall - base.recall).mean():+.3f}"
        M["pu_vs_fb"] = f"{(pu.recall - fbr.recall).mean():+.3f}"
        M["pu_gt_base"] = str(int((pu.recall > base.recall).sum()))
        N["xd_adapt_pu"] = pu.reset_index().round(4).to_dict("records")
    M["ctrl_fb_att"] = f"{att['FB']:.0f}"
    M["ctrl_fb_labels"] = f"{fb.labels.mean():,.0f}"
    N["xd_adapt_ctrl"] = pair.reset_index().round(4).to_dict("records")


def stream_macros():
    """Mixed streams, alternative novelty scores, Holm correction, step size and streaming throughput."""
    from scipy.stats import wilcoxon
    from revision_summary import paired
    # Holm over the rules compared with the detector (OS, LightGBM, 1% budget, family means)
    p = pd.read_csv(os.path.join(RES, "os_priorities.csv"))
    p = p[(p.model == "LGBM") & (p.budget == 0.01)]
    fr = p.groupby(["dataset", "held_out", "priority"]).recall.mean().unstack()
    rules = [r for r in fr.columns if r not in ("detector", "proxy_switch")]
    ps = {}
    for r in rules:
        d = (fr[r] - fr["detector"]).dropna()
        ps[r] = wilcoxon(d[d != 0]).pvalue
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, r in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[r]))
        holm[r] = run
    M["holm_n_rules"] = str(len(rules))
    M["holm_wf07"] = pv(holm["wfisher0.7"])
    N["os_holm_b0.01"] = {r: dict(p=ps[r], p_holm=holm[r]) for r in rules}
    # mixed streams
    fm = os.path.join(RES, "os_mixed.csv")
    if os.path.exists(fm):
        x = pd.read_csv(fm)
        for (pu, mode, tag) in [(0.002, "topk", ""), (0.002, "warmup", "_warm"), (0.005, "topk", "_05")]:
            y = x[(x.pi_u == pu) & (x.budget == 0.01) & (x["mode"] == mode)]
            g = y.groupby(["dataset", "held_out", "rule"])[["recall_unseen", "recall_known", "recall_all"]].mean()
            u, k = g.recall_unseen.unstack(), g.recall_known.unstack()
            r = paired(u["wfisher0.7"], u["detector"])
            M["mix_d" + tag] = ci(r)
            if tag == "":
                M["mix_dm"] = f"{r['mean']:.3f}"
                M["mix_n"] = str(r["n"])
                for rule, kk in [("detector", "det"), ("fisher", "fi"), ("wfisher0.7", "wf")]:
                    M[f"mix_u_{kk}"] = f(u[rule].mean(), 3)
                    M[f"mix_k_{kk}"] = f(k[rule].mean(), 3)
                M["mix_dk_wf"] = ci(paired(k["wfisher0.7"], k["detector"]))
                M["mix_dk_fi"] = ci(paired(k["fisher"], k["detector"]))
                N["os_mixed_b0.01"] = dict(unseen=u.mean().round(4).to_dict(), known=k.mean().round(4).to_dict())
    # alternative novelty scores
    fn = os.path.join(RES, "os_novalt.csv")
    if os.path.exists(fn):
        x = pd.read_csv(fn)
        x = x[x.budget == 0.01]
        g = x.groupby(["dataset", "held_out", "novelty", "rule"]).recall.mean().unstack(["novelty", "rule"])
        a = x.groupby(["dataset", "held_out", "novelty"]).auroc_nov.mean().unstack()
        M["nalt_n"] = str(len(g))
        for kind in ["knn", "iforest", "lof", "maha"]:
            M[f"nalt_{kind}_wf"] = f(g[(kind, "wfisher0.7")].mean(), 3)
            M[f"nalt_{kind}_nov"] = f(g[(kind, "novelty")].mean(), 3)
            M[f"nalt_{kind}_auc"] = f(a[kind].mean(), 2)
            if kind != "knn":
                M[f"nalt_knn_vs_{kind}"] = ci(paired(g[("knn", "wfisher0.7")], g[(kind, "wfisher0.7")]))
                M[f"nalt_{kind}_vs_det"] = ci(paired(g[(kind, "wfisher0.7")], g[("knn", "detector")]))
        M["nalt_det"] = f(g[("knn", "detector")].mean(), 3)
        N["os_novalt_b0.01"] = {k: dict(wf=float(g[(k, "wfisher0.7")].mean()), nov=float(g[(k, "novelty")].mean()),
                                    auroc=float(a[k].mean())) for k in ["knn", "iforest", "lof", "maha"]}
    # step size of the online update
    fg = os.path.join(RES, "extra_gamma.csv")
    if os.path.exists(fg):
        x = pd.read_csv(fg)
        t = x.groupby(["rule", "protocol"])[["load", "load_min10", "load_max10", "recall"]].mean()
        for rule in x.rule.unique():
            k = rule.replace(".", "").replace("_", "")
            M[f"gam_{k}_load"] = f"{t.loc[rule].load.min():.2f} to {t.loc[rule].load.max():.2f}"
            M[f"gam_{k}_rng"] = f"{t.loc[rule].load_min10.min():.2f} to {t.loc[rule].load_max10.max():.2f}"
        N["extra_gamma"] = t.reset_index().round(4).to_dict("records")
    # streaming throughput
    fb = os.path.join(RES, "stream_batch.csv")
    if os.path.exists(fb):
        b = pd.read_csv(fb).set_index("batch")
        for B in b.index:
            M[f"stream_fps_b{B}"] = f"{b.loc[B, 'flows_per_s']:,.0f}".replace(",", ",")
            M[f"stream_us_b{B}"] = f(b.loc[B, "us_per_flow"], 1)
            M[f"stream_p99_b{B}"] = f(b.loc[B, "p99_ms"], 1)
        M["stream_review_ps"] = f"{int(round(0.01 * b.loc[4096, 'flows_per_s'], -1)):,}"
        sh = pd.read_csv(os.path.join(RES, "stream_shards.csv")).set_index("shards")
        for P in sh.index:
            M[f"stream_fps_s{P}"] = f"{sh.loc[P, 'flows_per_s']:,.0f}"
            M[f"stream_load_s{P}"] = f(sh.loc[P, "load_union"], 2)
        M["stream_speedup_s4"] = f(sh.loc[4, "flows_per_s"] / sh.loc[1, "flows_per_s"], 1)
        M["stream_shard_load_rng"] = f"{sh.load_shard_min.min():.2f} to {sh.load_shard_max.max():.2f}"
        sy_ = os.path.join(RES, "stream_sync.csv")
        if os.path.exists(sy_):   # headline shard figures: median of the repeated runs (Table A22)
            g_ = pd.read_csv(sy_)
            g_ = g_[(g_["mode"] == "shards") & (g_.batch == 4096)].groupby("workers").median(numeric_only=True)
            for P in [2, 4]:
                M[f"stream_fps_s{P}"] = f"{g_.loc[P, 'flows_per_s']:,.0f}"
                M[f"stream_load_s{P}"] = f(g_.loc[P, "load_union"], 2)
            M["stream_speedup_s4"] = f(g_.loc[4, "flows_per_s"] / g_.loc[1, "flows_per_s"], 1)
            M["stream_shard_load_rng"] = f"{g_.load_shard_min.min():.2f} to {g_.load_shard_max.max():.2f}"
        mi = pd.read_csv(os.path.join(RES, "stream_misc.csv"))
        v = lambda it, th=None: mi[(mi["item"] == it) & ((mi.threads == th) if th else True)].value.iloc[0]
        M["stream_ntrain"] = f"{int(mi[mi['item'] == 'retrain_lgbm_s'].n_train.iloc[0]):,}"
        M["stream_retrain_1"] = f(v("retrain_lgbm_s", 1), 0)
        M["stream_retrain_4"] = f(v("retrain_lgbm_s", 4), 0)
        M["stream_mb_hnsw"] = f(v("state_mb_hnsw"), 1)
        M["stream_mb_lgbm"] = f(v("state_mb_lgbm"), 1)
        M["stream_mb_cal"] = f(v("state_mb_calibration"), 1)
        h = mi[mi["item"] == "hnsw"].groupby(["n_ref", "ef"])[["recall_at5", "build_s", "us_per_flow"]].agg(["mean", "min", "max"])
        for nr in [20000, 50000, 100000]:
            kk = nr // 1000
            M[f"hnsw_build_{kk}k"] = f(h.loc[(nr, 64), ("build_s", "mean")], 1)
            for ef in [64, 128, 256]:
                M[f"hnsw_rec_{kk}k_ef{ef}"] = f"{h.loc[(nr, ef), ('recall_at5', 'mean')]:.3f}"
                M[f"hnsw_recrng_{kk}k_ef{ef}"] = f"{h.loc[(nr, ef), ('recall_at5', 'min')]:.3f} to {h.loc[(nr, ef), ('recall_at5', 'max')]:.3f}"
                M[f"hnsw_us_{kk}k_ef{ef}"] = f(h.loc[(nr, ef), ("us_per_flow", "mean")], 1)

    # weighted Fisher for all four model families
    fw = os.path.join(RES, "os_models_wf.csv")
    if os.path.exists(fw):
        x = pd.read_csv(fw)
        x = x[x.budget == 0.01]
        out = {}
        for m, g in x.groupby("model"):
            fr = g.groupby(["dataset", "held_out", "rule"]).recall.mean().unstack()
            r = paired(fr["wfisher0.7"], fr["detector"])
            M[f"mwf_{m}"] = ci(r)
            M[f"mwf_{m}_m"] = f"{r['mean']:.3f}"
            M[f"mwf_{m}_det"] = f(fr["detector"].mean(), 3)
            M[f"mwf_{m}_wf"] = f(fr["wfisher0.7"].mean(), 3)
            out[m] = dict(r, det=float(fr["detector"].mean()), wf=float(fr["wfisher0.7"].mean()))
        oth = [out[m]["mean"] for m in out if m != "LGBM"]
        M["mwf_rng"] = f"{min(oth):.3f} to {max(oth):.3f}"
        N["os_models_wf_b0.01"] = out
    # tie-breaking at the p-value floor
    ft = os.path.join(RES, "os_ties.csv")
    if os.path.exists(ft):
        t = pd.read_csv(ft)
        fam = t.groupby(["dataset", "held_out", "rule"])[["floor", "tied", "recall", "recall_random_tie"]].mean()
        dd, ww = fam.xs("detector", level="rule"), fam.xs("wfisher0.7", level="rule")
        M["tie_floor_pct"] = pct(dd.floor.mean(), 0)
        M["tie_det_pct"] = pct(dd.tied.mean(), 1)
        M["tie_wf_pct"] = pct(ww.tied.mean(), 2)
        M["tie_det_max_pct"] = pct(t[t.rule == "detector"].tied.max(), 0)
        M["tie_det_rec"] = f(dd.recall.mean(), 3)
        M["tie_det_rec_rand"] = f(dd.recall_random_tie.mean(), 3)
        M["tie_wf_rec"] = f(ww.recall.mean(), 3)
        M["tie_wf_rec_rand"] = f(ww.recall_random_tie.mean(), 3)
        N["os_ties"] = dict(floor=float(dd.floor.mean()), tied_det=float(dd.tied.mean()), tied_wf=float(ww.tied.mean()))
    # per-flow cost of each stage of the streaming path
    fp = os.path.join(RES, "stream_profile.csv")
    if os.path.exists(fp):
        p = pd.read_csv(fp).set_index("stage")
        for st in p.index:
            M[f"prof_{st}_us"] = f(p.loc[st, "us_per_flow"], 1 if p.loc[st, "us_per_flow"] >= 0.1 else 2)
            M[f"prof_{st}_pct"] = pct(p.loc[st, "share"], 0 if p.loc[st, "share"] >= 0.01 else 1)
        M["prof_triage_pct"] = pct(p.loc[["p_detector", "p_novelty", "fisher", "update"], "share"].sum(), 1)
        M["prof_triage_us"] = f(p.loc[["p_detector", "p_novelty", "fisher", "update"], "us_per_flow"].sum(), 2)
        N["stream_profile"] = p.reset_index().to_dict("records")
    # heterogeneous shards on the chronological streams
    fh = os.path.join(RES, "extra_shards_het.csv")
    if os.path.exists(fh):
        h = pd.read_csv(fh)
        ok = h.groupby(["seed", "protocol"]).scheme.nunique()
        keep = ok[ok == 4].index
        h = h.set_index(["seed", "protocol"]).loc[keep].reset_index()
        M["shet_nseeds"] = words(h.seed.nunique())
        for pr, kp in [("T1-benign-time", "t1"), ("T2-day", "t2")]:
            g = h[h.protocol == pr].groupby("scheme")
            M[f"shet_{kp}_nseeds"] = words(h[h.protocol == pr].seed.nunique())
            for sc in ["global", "protocol", "skew4", "sync4"]:
                x = g.get_group(sc)
                M[f"shet_{kp}_{sc}_rec"] = f(x.recall.mean(), 3)
                M[f"shet_{kp}_{sc}_load"] = f(x.load.mean(), 2)
                M[f"shet_{kp}_{sc}_maxload"] = f(x.load_shard_max.max(), 2)
                M[f"shet_{kp}_{sc}_minload"] = f(x.load_shard_min.min(), 2)
        pr_ = h[h.scheme == "protocol"]
        M["shet_proto_minshare"] = pct(pr_.size_share_min.min(), 1)
        N["extra_shards_het"] = h.round(4).to_dict("records")


def check_macros():
    """Weighted-Fisher load and per-family counts, label conflicts, CORAL baseline."""
    from revision_summary import paired
    w = os.path.join(RES, "wf_load.csv")
    if os.path.exists(w):
        d = pd.read_csv(w)
        d = d[d.budget == 0.01]
        for reg, k in [("Unseen family", "os"), ("Cross-dataset", "xd"), ("In-domain", "id")]:
            for pr_, kp in [("wfisher0.7", "wf"), ("fisher", "fi"), ("detector", "de")]:
                g = d[(d.regime == reg) & (d.priority == pr_)].load
                M[f"wfl_{k}_{kp}_med"] = f(g.median())
                M[f"wfl_{k}_{kp}_q10"], M[f"wfl_{k}_{kp}_q90"] = f(g.quantile(.1)), f(g.quantile(.9))
    m = pd.read_csv(os.path.join(RES, "os_models_wf.csv"))
    m = m[m.budget == 0.01]
    for mod, g in m.groupby("model"):
        q = g.pivot_table(index=["dataset", "held_out"], columns="rule", values="recall", aggfunc="mean")
        y = q["wfisher0.7"] - q["detector"]
        M[f"wfam_{mod}_up"], M[f"wfam_{mod}_down"] = str(int((y > 0).sum())), str(int((y < 0).sum()))
        M[f"wfam_{mod}_same"] = str(int((y == 0).sum()))
        if mod == "LGBM":   # per-dataset mean gain over held-out families
            short = {"ACI-IoT-2023": "ACI-IoT-2023", "CIC-IDS2017": "CIC-IDS2017", "CSE-CIC-IDS2018": "CSE-CIC-IDS2018",
                     "UNSW-NB15": "UNSW-NB15", "ToN-IoT": "ToN-IoT"}
            pdg = y.groupby(level="dataset").agg(["mean", "size"])
            M["wfds_LGBM"] = "; ".join(f"{short.get(ds, ds)} {r_['mean']:+.3f} ({int(r_['size'])} families)"
                                       for ds, r_ in pdg.iterrows())
            M["wfds_LGBM_npos"] = words(int((pdg["mean"] > 0).sum()))
            M["wfds_LGBM_n"] = words(len(pdg))
    gm = pd.read_csv(os.path.join(RES, "extra_gamma.csv"))
    gm = gm[~gm.rule.str.startswith("batched")].groupby(["protocol", "rule"]).load.mean()
    for pr_, kp in [("T1-benign-time", "t1"), ("T2-day", "t2")]:
        M[f"gam_{kp}_load_rng"] = f"{gm.loc[pr_].min():.2f} to {gm.loc[pr_].max():.2f}"
    tb = os.path.join(RES, "budget_ties.csv")
    if os.path.exists(tb):
        tb = pd.read_csv(tb)
        idm = tb[(tb.regime == "In-domain") & (tb.priority == "fisher")]
        M["tie_id_runs_tied"] = words(int(idm.tie_at_threshold.sum()))
        M["tie_id_load"] = f(idm.load.median())
        M["tie_id_load_jit"] = f(idm.load_jitter.median())
        M["tie_id_load_noton"] = f(idm[idm.source != "ToN-IoT"].load.median())
        xw = pd.read_csv(os.path.join(RES, "xd_warmup.csv"))
        xw = xw[(xw.model == "LGBM") & (xw.source == "ToN-IoT") & (xw.target == "ToN-IoT") & (xw.warmup == 0.2)]
        M["tie_id_ton_m"] = f"{int(xw.n_warm.median()):,}"
        allt = tb.groupby("regime").tie_at_threshold.mean()
        M["tie_share_runs_os"] = pct(allt["Unseen family"])
        M["tie_share_runs_xd"] = pct(allt["Cross-dataset"])
        osw = tb[(tb.regime == "Unseen family") & (tb.priority == "wfisher0.7")]
        M["bound_os_wf_med"] = f(osw.bound_rel.median())
        M["bound_os_wf_rng"] = f"{osw.bound_rel.quantile(.1):.2f} to {osw.bound_rel.quantile(.9):.2f}"
        M["bound_os_T"] = f"{int(osw['T'].median()):,}"
        M["bound_os_Rg"] = f"{(osw.range / osw.gamma).median():.0f}"
    bb = os.path.join(RES, "budget_band.csv")
    if os.path.exists(bb):
        bb = pd.read_csv(bb)
        M["band_exact_rng"] = f"{pct(bb.cover_exact.min())} to {pct(bb.cover_exact.max())}"
        M["band_normal_rng"] = f"{pct(bb.cover_normal.min())} to {pct(bb.cover_normal.max())}"
        r1 = bb[(bb.budget == 0.01) & (bb.warmup == 0.01)].iloc[0]
        M["band_small_exact"], M["band_small_normal"] = pct(r1.cover_exact), pct(r1.cover_normal)
    dc = os.path.join(RES, "budget_decoy.csv")
    if os.path.exists(dc):
        dc = pd.read_csv(dc)
        g = dc.groupby(["decoy_frac", "rule", "priority"])[["recall", "load", "decoy_share"]].mean()
        M["dec_nruns"] = str(dc[dc.decoy_frac == 0][["source", "seed"]].drop_duplicates().shape[0])
        for fr_, kf in [(0.0, "0"), (0.01, "1"), (0.02, "2")]:
            for rl, kr in [("online", "on"), ("warmup", "wu")]:
                for pr_, kp in [("detector", "de"), ("fisher", "fi"), ("wfisher0.7", "wf")]:
                    r_ = g.loc[(fr_, rl, pr_)]
                    M[f"dec_{kf}_{kr}_{kp}_rec"] = f(r_.recall, 3)
                    M[f"dec_{kf}_{kr}_{kp}_load"] = f(r_.load)
                    M[f"dec_{kf}_{kr}_{kp}_share"] = pct(r_.decoy_share)
        N["budget_decoy"] = g.reset_index().round(4).to_dict("records")
    if "ctrl_fb_better_rl" in M:
        M["ctrl_rl_better_n"] = words(6 - int(M["ctrl_fb_better_rl"]))
    sy = os.path.join(RES, "stream_sync.csv")
    if os.path.exists(sy):
        sy = pd.read_csv(sy)
        g = sy.groupby(["mode", "workers", "batch"]).median(numeric_only=True)
        b1 = g.loc[("threads", 1, 4096)].flows_per_s
        M["sync_reps"] = words(sy.rep.nunique())
        for w_ in [1, 2, 4]:
            M[f"sync_thr{w_}_fps"] = f"{g.loc[('threads', w_, 4096)].flows_per_s:,.0f}"
            M[f"sync_thr{w_}_su"] = f"{g.loc[('threads', w_, 4096)].flows_per_s / b1:.1f}"
        for w_ in [1, 2, 3, 4]:
            M[f"sync_sh{w_}_fps"] = f"{g.loc[('shards', w_, 4096)].flows_per_s:,.0f}"
            bs = g.loc[("shards", 1, 4096)].flows_per_s
            M[f"sync_sh{w_}_su"] = f"{g.loc[('shards', w_, 4096)].flows_per_s / bs:.1f}"
            M[f"sync_sh{w_}_eff"] = pct(g.loc[('shards', w_, 4096)].flows_per_s / bs / w_)
        M["sync_sh4_wallfps"] = f"{g.loc[('shards', 4, 4096)].flows_per_s_wall:,.0f}"
        M["sync_sh1_wallfps"] = f"{g.loc[('shards', 1, 4096)].flows_per_s_wall:,.0f}"
        for s_ in [256, 1024, 4096]:
            r_ = g.loc[("shared", 4, s_)]
            M[f"sync_shared{s_}_fps"] = f"{r_.flows_per_s:,.0f}"
            M[f"sync_shared{s_}_wait"] = pct(r_.lock_wait_share, 2)
            M[f"sync_shared{s_}_tenth"] = f"{r_.load_tenth_min:.2f} to {r_.load_tenth_max:.2f}"
            M[f"sync_shared{s_}_load"] = f(r_.load_union)
        r_ = g.loc[("shards", 4, 4096)]
        M["sync_sh4_tenth"] = f"{r_.load_tenth_min:.2f} to {r_.load_tenth_max:.2f}"
        mx = sy[sy["mode"] == "shared"].lock_wait_share.max()
        M["sync_wait_max"] = pct(mx, 2)
    c = os.path.join(RES, "label_conflicts.csv")
    if os.path.exists(c):
        c = pd.read_csv(c)
        M["conf_max_share"] = pct(c.share.max(), 1)
        M["conf_max_ds"] = c.loc[c.share.idxmax(), "dataset"]
        M["conf_min_share"] = pct(c.share.min(), 2)
        N["label_conflicts"] = c.round(6).to_dict("records")
    k = os.path.join(RES, "xd_coral.csv")
    if os.path.exists(k):
        k = pd.read_csv(k)
        a = pd.read_csv(os.path.join(RES, "xd_adapt.csv"))
        M["coral_nseeds"] = words(k.seed.nunique())
        for c_ in [0.01, 0.05]:
            cc = int(round(100 * c_))
            for v, kv in [("CORAL_detector", "det"), ("CORAL_wfisher", "wf")]:
                g = k[(k.budget == c_) & (k.variant == v)]
                M[f"coral_{kv}_rec_b{cc}"] = f(g.groupby(["source", "target"]).recall.mean().mean(), 3)
                M[f"coral_{kv}_load_b{cc}"] = f(g.load.median())
            b = a[(a.budget == c_) & (a.variant == "base_wfisher")].merge(
                k[(k.budget == c_) & (k.variant == "CORAL_wfisher")][["source", "target", "seed"]],
                on=["source", "target", "seed"])
            bp = b.groupby(["source", "target"]).recall.mean()
            kp = k[(k.budget == c_) & (k.variant == "CORAL_wfisher")].groupby(["source", "target"]).recall.mean()
            M[f"coral_base_rec_b{cc}"] = f(bp.mean(), 3)
            if len(kp) == 6:
                M[f"coral_vs_base_b{cc}"] = ci_pairs(paired(kp, bp))
        N["xd_coral"] = k.groupby(["source", "target", "variant", "budget"])[["recall", "precision", "load"]].mean() \
            .reset_index().round(4).to_dict("records")


def round7_macros():
    """Load offset, break-even interval, timing spread, budget scaling, weighted Fisher under
    contamination and for held-out groups, staleness of the novelty reference."""
    from scipy.stats import norm as _norm
    from revision_summary import paired
    lo = os.path.join(RES, "load_offset.csv")
    if os.path.exists(lo):
        d = pd.read_csv(lo)
        de, fi = d[d.priority == "detector"], d[d.priority == "fisher"]
        M["lo_det_med_rng"] = f"{de['median'].min():.2f} to {de['median'].max():.2f}"
        M["lo_fis_med_rng"] = f"{fi['median'].min():.2f} to {fi['median'].max():.2f}"
        M["lo_sim_med_rng"] = f"{d.sim_median.min():.2f} to {d.sim_median.max():.2f}"
        ex = fi[fi.ci_lo > fi.sim_median]
        names = {"LGBM": "LightGBM", "RF": "the random forest", "MLP": "the MLP", "LR": "logistic regression"}
        M["lo_fis_excl"] = " and ".join(f"{names[r.model]} ({r.ci_lo:.3f} to {r.ci_hi:.3f})" for r in ex.itertuples())
        M["lo_fis_excl_n"] = words(len(ex))
        M["lo_streams"] = str(int(d.streams.max()))
        M["lo_sd_within"] = f"{d.sd_within.min():.2f} to {d.sd_within.max():.2f}"
        M["lo_sd_between"] = f"{d.sd_between.min():.2f} to {d.sd_between.max():.2f}"
    ow = os.path.join(RES, "os_warmup.csv")
    if os.path.exists(ow):
        w_ = pd.read_csv(ow)
        w_ = w_[(w_.rule == "warmup") & (w_.budget == 0.01) & (w_.warmup == 0.2) & (w_.model == "LGBM")]
        for pr_, k_ in [("fisher", "fis"), ("detector", "det")]:
            x_ = w_[w_.priority == pr_]
            M[f"lo_{k_}_mean"] = f(x_.load.mean())
            M[f"lo_{k_}_med"] = f(x_.load.median())
            bd = x_.groupby("dataset").load.median()
            M[f"lo_{k_}_byds"] = "; ".join(f"{d} {v:.2f}" for d, v in bd.items())
    tb_ = os.path.join(RES, "budget_ties.csv")
    if os.path.exists(tb_):
        o_ = pd.read_csv(tb_)
        o_ = o_[(o_.regime == "Unseen family") & (o_.priority == "wfisher0.7")]
        M["bound_os_ratio_med"] = f(((o_.range / o_.gamma).median() + 1) / (o_["T"].median() * 0.01))
    # hierarchical bootstrap of the weighted Fisher gain (datasets, then families within datasets)
    pp_ = os.path.join(RES, "os_priorities.csv")
    if os.path.exists(pp_):
        q_ = pd.read_csv(pp_)
        q_ = q_[(q_.budget == 0.01) & (q_.model == "LGBM") & q_.priority.isin(["detector", "wfisher0.7"])]
        fam_ = q_.groupby(["dataset", "held_out", "priority"]).recall.mean().unstack()
        dif = (fam_["wfisher0.7"] - fam_["detector"]).dropna()
        groups = [v.to_numpy() for _, v in dif.groupby(level="dataset")]
        rb = np.random.default_rng(11)
        bs = []
        for _ in range(5000):
            pick = rb.integers(0, len(groups), len(groups))
            bs.append(np.concatenate([rb.choice(groups[i], len(groups[i])) for i in pick]).mean())
        M["hb_wf_ci"] = f"{np.quantile(bs, .025):+.3f} to {np.quantile(bs, .975):+.3f}".replace("-0.000", "0.000")
        M["hb_wf_mean"] = f"{dif.mean():+.3f}"
    # queue recall of the threshold rules on the chronological streams (1% budget)
    ts = pd.read_csv(os.path.join(RES, "extra_temporal_stream.csv"))
    ts = ts[ts.budget == 0.01].groupby(["protocol", "priority", "rule"])[["recall", "precision"]].mean()
    for pk, pr in [("t1", "T1-benign-time"), ("t1r", "T1-random"), ("t2", "T2-day"), ("t2r", "T2-random")]:
        for qk, q in [("fi", "fisher"), ("de", "detector")]:
            for rk, rl_ in [("on", "online"), ("so", "source")]:
                M[f"ts_{pk}_{qk}_{rk}"] = f"{ts.loc[(pr, q, rl_), 'recall']:.3f}"
    # Holm results for the other weights (LightGBM, 1% budget)
    from scipy.stats import wilcoxon
    pp = pd.read_csv(os.path.join(RES, "os_priorities.csv"))
    pp = pp[(pp.model == "LGBM") & (pp.budget == 0.01)]
    fr = pp.groupby(["dataset", "held_out", "priority"]).recall.mean().unstack()
    rules = [r_ for r_ in fr.columns if r_ not in ("detector", "proxy_switch")]
    ps = {}
    for r_ in rules:
        d = (fr[r_] - fr["detector"]).dropna()
        ps[r_] = wilcoxon(d[d != 0]).pvalue
    hol, run = {}, 0.0
    for i, r_ in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (len(ps) - i) * ps[r_]))
        hol[r_] = run
    for w in ("0.8", "0.9"):
        M[f"holm_wf{w[2]}"] = f"{hol['wfisher' + w]:.3f}"
        M[f"d_wf{w[2]}"] = f"{(fr['wfisher' + w] - fr['detector']).mean():+.3f}"
    # caught/missed comparison without the family whose group differs between the two seed halves
    from revision_summary import paired as _paired
    po = pd.read_csv(os.path.join(RES, "os_priorities.csv"))
    dt = pd.read_csv(os.path.join(RES, "os_detection_all.csv"))
    po, dt = po[(po.model == "LGBM") & (po.budget == 0.01)], dt[dt.model == "LGBM"]
    ga = dt[dt.seed <= 2].groupby(["dataset", "held_out"]).recall_unseen_at_alpha.mean() < .5
    gb = dt[dt.seed >= 3].groupby(["dataset", "held_out"]).recall_unseen_at_alpha.mean() < .5
    dis = ga[ga != gb].index.tolist()
    M["cm_dis_fam"] = "; ".join(f"{h} ({d_})" for d_, h in dis)
    po = po.set_index(["dataset", "held_out"])
    po["wk"] = np.where(po.seed >= 3, ga.reindex(po.index).values, gb.reindex(po.index).values)
    po = po[~po.index.isin(dis)].reset_index()
    for wk, lab in [(False, "caught"), (True, "missed")]:
        xx = po[po.wk == wk]
        a = xx[xx.priority == "wfisher0.7"].groupby(["dataset", "held_out"]).recall.mean()
        b = xx[xx.priority == "detector"].groupby(["dataset", "held_out"]).recall.mean()
        q = _paired(a, b)
        M[f"cmx_{lab}"] = f"{q['mean']:+.3f} (95% CI {q['lo']:+.3f} to {q['hi']:+.3f}; {pv(q['p'])}, {q['n']} families)"
    # observed load against the exact expectation of the same flows under random order
    fo = os.path.join(RES, "offset_order.csv")
    if os.path.exists(fo):
        od = pd.read_csv(fo)
        rg = np.random.default_rng(7)
        out = {}
        for prn in ["detector", "fisher"]:
            dl, lo_, hi_ = [], [], []
            for m_, x_ in od.groupby("model"):
                st = (x_[f"load_{prn}"] - x_[f"load_{prn}_expected"]).groupby([x_.dataset, x_.seed]).mean().to_numpy()
                bs = [rg.choice(st, len(st)).mean() for _ in range(4000)]
                dl.append(st.mean()); lo_.append(np.quantile(bs, .025)); hi_.append(np.quantile(bs, .975))
            out[prn] = (dl, lo_, hi_)
            M[f"od_{prn}_diff_rng"] = f"{min(dl):+.3f} to {max(dl):+.3f}"
            M[f"od_{prn}_ci_rng"] = f"{min(lo_):+.3f} to {max(hi_):+.3f}"
            M[f"od_{prn}_obs_rng"] = f"{od.groupby('model')[f'load_{prn}'].mean().min():.2f} to {od.groupby('model')[f'load_{prn}'].mean().max():.2f}"
            ex = od.groupby('model')[f'load_{prn}_expected'].mean()
            lo_, hi_ = f"{ex.min():.2f}", f"{ex.max():.2f}"
            M[f"od_{prn}_exp_rng"] = lo_ if lo_ == hi_ else f"{lo_} to {hi_}"
        M["od_nperm"] = "100"
        M["od_all_include0"] = "yes" if all(l <= 0 <= h for prn in out for l, h in zip(out[prn][1], out[prn][2])) else "no"
    # remedies for heterogeneous shards
    fx = os.path.join(RES, "extra_shards_fix.csv")
    if os.path.exists(fx):
        fx_ = pd.read_csv(fx)
        g_ = fx_.groupby(["protocol", "scheme"]).mean(numeric_only=True)
        g_["load_shard_min"] = fx_.groupby(["protocol", "scheme"]).load_shard_min.min()
        g_["load_shard_max"] = fx_.groupby(["protocol", "scheme"]).load_shard_max.max()
        for pk, pr_ in [("t1", "T1-benign-time"), ("t2", "T2-day")]:
            for sc in ["skew4_sync256", "skew4_sync1", "skew4_share", "protocol_sync256", "protocol_share"]:
                v = g_.loc[(pr_, sc)]
                k_ = f"shfix_{pk}_{sc}"
                M[k_ + "_rec"] = f"{v.recall:.3f}"
                M[k_ + "_load"] = f"{v.load:.2f}"
                M[k_ + "_shrng"] = f"{v.load_shard_min:.2f} to {v.load_shard_max:.2f}"
        M["shfix_nseeds"] = words(pd.read_csv(fx).seed.nunique())
    # shuffled-order control of the chronological protocols
    fs_ = os.path.join(RES, "extra_temporal_shuffled.csv")
    if os.path.exists(fs_):
        t_ = pd.read_csv(fs_)
        t_ = t_[t_.budget == 0.01].groupby(["protocol", "priority", "rule"]).mean(numeric_only=True)
        for pk, pr_ in [("t1", "T1-benign-time"), ("t2", "T2-day")]:
            for qk, q_ in [("fi", "fisher"), ("de", "detector")]:
                for rk, rl_ in [("wu", "warmup"), ("on", "online")]:
                    v = t_.loc[(pr_, q_, rl_)]
                    M[f"tsh_{pk}_{qk}_{rk}_load"] = f"{v.load:.2f}"
                    M[f"tsh_{pk}_{qk}_{rk}_rng"] = f"{v.load_min10:.2f} to {v.load_max10:.2f}"
                    M[f"tsh_{pk}_{qk}_{rk}_rec"] = f"{v.recall:.3f}"
    bc = N.get("os_breakeven_ci", {}).get("fisher_b0.01")
    if bc:
        M["be_ci_lo"], M["be_ci_hi"] = f(bc["lo"], 2), f(bc["hi"], 2)
        M["be_share_free"] = pct(bc["share_free"])
    sy = os.path.join(RES, "stream_sync.csv")
    if os.path.exists(sy):
        sy = pd.read_csv(sy)
        def rr(mo, w_, b_=4096):
            v = sy[(sy["mode"] == mo) & (sy.workers == w_) & (sy.batch == b_)].flows_per_s
            return f"{v.min():,.0f} to {v.max():,.0f}"
        M["sync_thr4_rng"], M["sync_sh4_rng"] = rr("threads", 4), rr("shards", 4)
        M["sync_shared4096_rng"] = rr("shared", 4)
        v = sy[(sy["mode"] == "shards") & (sy.workers == 4) & (sy.batch == 4096)].flows_per_s
        M["sync_sh4_spread"] = pct((v.max() - v.min()) / v.median())
    # warm-up size and stream length needed at small budgets (Proposition 1 and bound (7))
    z = _norm.ppf(0.95)
    for b_, k_ in [(0.01, "1"), (0.001, "01"), (0.0001, "001")]:
        m_ = z ** 2 * (1 - b_) / (b_ * 0.1 ** 2)
        M[f"scale_m_b{k_}"] = f"{int(round(m_, -3)):,}" if m_ < 1e6 else f"{m_ / 1e6:.1f} million"
        if "bound_os_Rg" in M:
            T_ = (float(M["bound_os_Rg"]) + 1) / (0.1 * b_)
            M[f"scale_T_b{k_}"] = f"{int(round(T_, -3)):,}" if T_ < 1e6 else f"{T_ / 1e6:.1f} million"
    import glob as _glob
    rfs = sorted(_glob.glob(os.path.join(RES, "extra_reference_wf_s*.csv")))
    if rfs:
        x = pd.concat([pd.read_csv(v) for v in rfs])
        if "rec_wfisher" in x:
            g = x.groupby(["n_ref", "contamination"])[["rec_detector", "rec_fisher", "rec_wfisher"]].mean()
            base = x[(x.n_ref == 20000) & (x.contamination == 0)].set_index(["dataset", "held_out", "seed"])
            for c_, k_ in [(0.001, "c01"), (0.01, "c1"), (0.05, "c5")]:
                if (20000, c_) in g.index:
                    M[f"refwf_{k_}_wf"] = f(g.loc[(20000, c_), "rec_wfisher"], 3)
                    M[f"refwf_{k_}_fi"] = f(g.loc[(20000, c_), "rec_fisher"], 3)
                    M[f"refwf_{k_}_de"] = f(g.loc[(20000, c_), "rec_detector"], 3)
                    cc_ = x[(x.n_ref == 20000) & (x.contamination == c_)].set_index(["dataset", "held_out", "seed"])
                    a = cc_.rec_wfisher.groupby(level=[0, 1]).mean()
                    b = base.rec_wfisher.groupby(level=[0, 1]).mean().reindex(a.index)
                    d_ = paired(a, b)
                    M[f"refwf_{k_}_d"] = f"{d_['mean']:+.3f}"
                    a2 = cc_.rec_detector.groupby(level=[0, 1]).mean()
                    M[f"refwf_{k_}_vsdet"] = f"{(a - a2).mean():+.3f}"
            if (20000, 0.0) in g.index:
                M["refwf_c0_wf"] = f(g.loc[(20000, 0.0), "rec_wfisher"], 3)
                M["refwf_c0_fi"] = f(g.loc[(20000, 0.0), "rec_fisher"], 3)
                M["refwf_c0_de"] = f(g.loc[(20000, 0.0), "rec_detector"], 3)
    gf = os.path.join(RES, "extra_group_wf.csv")
    if os.path.exists(gf):
        x = pd.read_csv(gf)
        if "rec_wfisher" in x:
            fam = x.groupby(["dataset", "held_out_group"])[["rec_detector", "rec_fisher", "rec_wfisher"]].mean()
            M["ogwf_n"] = str(len(fam))
            M["ogwf_det"], M["ogwf_fi"], M["ogwf_wf"] = (f(fam[c].mean(), 3) for c in ["rec_detector", "rec_fisher", "rec_wfisher"])
            d_ = paired(fam.rec_wfisher, fam.rec_detector)
            M["ogwf_d"] = f"{d_['mean']:+.3f} (95% CI {d_['lo']:+.3f} to {d_['hi']:+.3f}; {pv(d_['p'])}, {d_['n']} groups)"
            d2 = paired(fam.rec_fisher, fam.rec_detector)
            M["ogwf_d_fi"] = f"{d2['mean']:+.3f}"
    st = os.path.join(RES, "t1_staleness.csv")
    if os.path.exists(st):
        x = pd.read_csv(st).groupby("reference").mean(numeric_only=True)
        for k_ in ["old", "recent"]:
            M[f"st_{k_}_age"] = f(x.loc[k_, "ref_age_days"], 1)
            M[f"st_{k_}_pnov"] = pct(x.loc[k_, "benign_pnov_le_001"], 1)
            M[f"st_{k_}_auc"] = f(x.loc[k_, "auroc_nov"], 3)
            for r_ in ["novelty", "detector", "fisher", "wfisher0.7"]:
                M[f"st_{k_}_{r_.replace('.', '')}"] = f(x.loc[k_, f"rec_{r_}"], 3)
        M["st_nseeds"] = words(pd.read_csv(st).seed.nunique())


def dist_macros():
    """Independent and coordinated workers on two machines (dist/dist_bench.py)."""
    fr = []
    for h in ["cloud", "mac"]:
        p = os.path.join(RES, f"dist_bench_{h}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["host"] = h
            fr.append(d)
    if not fr:
        return
    d = pd.concat(fr, ignore_index=True)
    ind = d[d["mode"] == "indep"].groupby(["host", "workers"]).agg(
        fps=("flows_per_s", "median"), load=("load", "median"), lo=("load_tenth_min", "median"),
        hi=("load_tenth_max", "median")).reset_index()
    for h in ["cloud", "mac"]:
        g = ind[ind.host == h].set_index("workers")
        if not len(g):
            continue
        b = g.fps.loc[1]
        for P, x in g.iterrows():
            M[f"dist_{h}_fps{P}"] = f"{x.fps:,.0f}"
            M[f"dist_{h}_su{P}"] = f"{x.fps / b:.2f}"
            M[f"dist_{h}_eff{P}"] = f"{x.fps / b / P:.2f}"
        M[f"dist_{h}_pmax"] = str(int(g.index.max()))
        M[f"dist_{h}_cores"] = str(int(d[d.host == h].cores.iloc[0]))
    M["dist_ind_load_rng"] = f"{ind.load.min():.3f} to {ind.load.max():.3f}"
    M["dist_ind_tenth_rng"] = f"{ind.lo.min():.2f} to {ind.hi.max():.2f}"
    ck = os.path.join(RES, "dist_check.csv")
    if os.path.exists(ck):
        v = pd.read_csv(ck).set_index("host").max_abs_priority_difference
        e = int(np.floor(np.log10(v.loc["mac"])))
        M["dist_check_mac"] = f"{v.loc['mac'] / 10 ** e:.0f} × 10^−{-e}^"
    c = d[d["mode"] == "coord"].groupby(["host", "workers", "batch", "delay_ms", "step"]).agg(
        fps=("flows_per_s", "median"), load=("load", "median"), lo=("load_tenth_min", "median"),
        hi=("load_tenth_max", "median"), st=("staleness", "median"), up=("updates", "median")).reset_index()
    pw, fu = c[c.step == "perworker"], c[c.step == "full"]
    M["dist_div_load_rng"] = f"{pw.load.min():.3f} to {pw.load.max():.3f}"
    M["dist_div_tenth_rng"] = f"{pw.lo.min():.2f} to {pw.hi.max():.2f}"
    M["dist_delays"] = ", ".join(str(int(x)) for x in sorted(c.delay_ms.unique())[:-1]) + \
        f" and {int(c.delay_ms.max())}"
    a = fu[(fu.batch == 256) & (fu.delay_ms <= 10)]
    M["dist_full256_load_max"] = f"{a.load.max():.2f}"
    M["dist_full256_tenth"] = f"{a.lo.min():.2f} to {a.hi.max():.2f}"
    m = fu[fu.host == "mac"]
    if len(m):
        for bs in [1024, 4096]:
            x = m[m.batch == bs]
            M[f"dist_full{bs}_load_rng"] = f"{x.load.min():.1f} to {x.load.max():.1f}"
        x = m[(m.batch == 256) & (m.delay_ms == m.delay_ms.max())].iloc[0]
        M["dist_full256_d50_load"] = f"{x.load:.1f}"
        mp = pw[pw.host == "mac"]
        y = mp[(mp.batch == 256) & (mp.delay_ms == mp.delay_ms.max())].iloc[0]
        y0 = mp[(mp.batch == 256) & (mp.delay_ms == 0)].iloc[0]
        M["dist_d50_stale"] = f"{y.st:.0f}"
        M["dist_d50_applied"] = pct(y.up / y0.up)
        M["dist_mac_stale0"] = f"{y0.st:.1f}"
        M["dist_div_fps_mac_rng"] = f"{mp.fps.min():,.0f} to {mp.fps.max():,.0f}"
    cl = pw[(pw.host == "cloud") & (pw.batch == 256) & (pw.delay_ms == 0)]
    if len(cl):
        M["dist_cloud_stale0"] = f"{cl.st.iloc[0]:.1f}"
    sc = os.path.join(RES, "extra_shards_coord.csv")
    if os.path.exists(sc):
        x = pd.read_csv(sc)
        sc2 = os.path.join(RES, "extra_shards_coord_small.csv")
        if os.path.exists(sc2):
            x = pd.concat([x, pd.read_csv(sc2).query("scheme != 'global'")])
        g = x.groupby(["protocol", "scheme"])[["recall", "load", "load_shard_min", "load_shard_max"]].mean()
        M["shco_nseeds"] = words(x.seed.nunique())
        for (pr, sch), v in g.iterrows():
            k = f"shco_{pr[:2].lower()}_{sch}"
            M[k + "_rec"] = f"{v.recall:.3f}"
            M[k + "_load"] = f"{v.load:.2f}"
            M[k + "_shrng"] = f"{v.load_shard_min:.2f} to {v.load_shard_max:.2f}"
        sm = g[g.index.get_level_values("scheme").str.contains("coord(?:16|64)_", regex=True)]
        if len(sm):
            t1 = sm.loc["T1-benign-time"]
            t2 = sm.loc["T2-day"]
            M["shco_small_t1_load_rng"] = f"{t1.load.min():.2f} to {t1.load.max():.2f}"
            M["shco_small_t2_rec_rng"] = f"{t2.recall.min():.3f} to {t2.recall.max():.3f}"


def small_budget_macros():
    """Small budgets and weighted Fisher on the chronological streams (revision_small_budget.py)."""
    p = os.path.join(RES, "extra_small_budget.csv")
    if not os.path.exists(p):
        return
    x = pd.read_csv(p)
    x = x[x.rule == "online"]
    g = x.groupby(["protocol", "budget", "priority"])[["load", "load_min10", "load_max10", "recall", "r_warm"]].mean()
    M["sb_nseeds"] = words(x.seed.nunique())
    for (pr, c, name), v in g.iterrows():
        k = f"sb_{pr[:2].lower()}_{name.replace('.', '')}_{int(round(c * 10000))}"
        M[k + "_rec"] = f"{v.recall:.3f}"
        M[k + "_load"] = f"{v.load:.2f}"
        M[k + "_tenth"] = f"{v.load_min10:.2f} to {v.load_max10:.2f}"
    sm = g.loc[[i for i in g.index if i[1] < 0.01]]
    M["sb_load_rng"] = f"{sm.load.min():.2f} to {sm.load.max():.2f}"
    M["sb_tenth_max"] = f"{sm.load_max10.max():.1f}"
    M["sb_rwarm_rng"] = f"{sm.r_warm.min():.0f} to {sm.r_warm.max():.0f}"
    M["sb_rec_max"] = f"{sm.recall.max():.3f}"


def sync_macros():
    """Coordinator with workers that wait for the reply (dist/dist_bench.py --sync, cloud VM, 4 workers)."""
    p = os.path.join(RES, "dist_sync_cloud.csv")
    if not os.path.exists(p):
        return
    x = pd.read_csv(p)
    p64 = os.path.join(RES, "dist_sync64_cloud.csv")
    if os.path.exists(p64):
        x = pd.concat([x, pd.read_csv(p64)])
    x = x[x.step == "perworker"].groupby(["batch", "delay_ms"])[["flows_per_s", "load", "load_tenth_min",
                                                              "load_tenth_max", "staleness"]].median()
    for (bs, dl), v in x.iterrows():
        M[f"dsync_{bs}_{dl}_fps"] = f"{v.flows_per_s:,.0f}"
    lo_, hi_ = f"{x.load.min():.3f}", f"{x.load.max():.3f}"
    M["dsync_load_rng"] = lo_ if lo_ == hi_ else f"{lo_} to {hi_}"
    M["dsync_tenth_rng"] = f"{x.load_tenth_min.min():.2f} to {x.load_tenth_max.max():.2f}"
    M["dsync_stale"] = f"{x.staleness.median():.1f}"
    M["dsync_delays"] = ", ".join(str(int(d)) for d in sorted(x.index.get_level_values(1).unique())[:-1]) + \
        f" and {int(x.index.get_level_values(1).max())}"


def build():
    os_macros()
    for mod in ("xd_macros", "misc_macros", "extra_macros", "adapt_macros", "lowpi_macros", "ctrl_macros", "stream_macros", "check_macros", "round7_macros", "dist_macros", "small_budget_macros", "sync_macros"):
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

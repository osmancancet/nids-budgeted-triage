"""Manuscript tables (pandoc pipe tables with captions) built from results/ and results/revision/.

Writes results/revision/manuscript_tables.json: {name: markdown}. Tables whose inputs are missing are
skipped with a message.
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, REV = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "revision")
CIC = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023"]
DS = CIC + ["UNSW-NB15", "ToN-IoT"]
MODELS = ["LGBM", "RF", "MLP", "LR"]
MN = {"LGBM": "LightGBM", "RF": "Random forest", "MLP": "MLP", "LR": "Logistic regression"}
PRN = {"detector": "Detector", "novelty": "Novelty", "combined": "Bonferroni", "fisher": "Fisher",
       "wfisher0.6": "Weighted Fisher, w = 0.6", "wfisher0.7": "Weighted Fisher, w = 0.7",
       "wfisher0.8": "Weighted Fisher, w = 0.8", "wfisher0.9": "Weighted Fisher, w = 0.9",
       "stouffer": "Stouffer", "stacked": "Learned combination", "borda": "Rank aggregation",
       "uncertainty": "Calibrated uncertainty", "split0.1": "Split budget, 10%", "split0.25": "Split budget, 25%",
       "split0.5": "Split budget, 50%", "proxy_switch": "Label-free switch"}
T = {}


def r(name):
    p = os.path.join(REV, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def ms(v, k=3):
    return f"{np.mean(v):.{k}f} ± {np.std(v, ddof=1):.{k}f}"


def q3(v, k=2):
    v = pd.Series(v)
    return f"{v.median():.{k}f} [{v.quantile(.1):.{k}f}, {v.quantile(.9):.{k}f}]"


def table(caption, header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = [caption, "", "| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- Table 2
def t_data():
    rows = []
    for f, ds in [("cic2017", "CIC-IDS2017"), ("cic2018", "CSE-CIC-IDS2018"), ("aci", "ACI-IoT-2023"),
                  ("unsw", "UNSW-NB15"), ("ton", "ToN-IoT")]:
        d = pd.read_csv(os.path.join(RES, f"data_summary_{f}.csv"))
        x = d.iloc[0]
        rows.append([ds] + [str(x[c]) for c in d.columns[1:]])
    T["data_raw"] = pd.concat([pd.read_csv(os.path.join(RES, f"data_summary_{f}.csv")) for f in
                               ["cic2017", "cic2018", "aci", "unsw", "ton"]]).to_dict("records")


# ---------------------------------------------------------------- Table 3 / A2 / A3
def t_xd():
    d = pd.read_csv(os.path.join(RES, "xd_detection.csv"))
    m = d[d.group.isna()]
    rows = []
    for model in MODELS:
        for setting, lab in [("in-domain", "In-domain"), ("cross-dataset", "Cross-dataset")]:
            x = m[(m.model == model) & (m.setting == setting) & m.source.isin(CIC)]
            g = x.groupby("seed").mean(numeric_only=True)
            rows.append([MN[model], lab, ms(g.auroc), ms(g.ece_raw), ms(g.ece_iso), ms(g.fpr_inflation, 1),
                         ms(g.tpr_at_alpha), ms(g["prec_at_pi0.01"])])
    T["xd"] = table("**Table 3.** In-domain and cross-dataset performance on the three CICFlowMeter datasets. "
                    "Values are means ± standard deviations over five seeds; cross-dataset values are first "
                    "averaged over the six directed pairs, so they average pairs that flood and pairs that fall "
                    "silent (Appendix Table A3 gives each pair). FPR/α is the realised false-positive rate at the "
                    "source threshold divided by α = 1%; TPR is measured at the same threshold; Prec is the "
                    "precision implied at 1% attack prevalence.",
                    ["Model", "Setting", "AUROC", "ECE (raw)", "ECE (isotonic)", "FPR/α", "TPR", "Prec"], rows,
                    ["l", "l"] + ["r"] * 6)
    # A2: in-domain, all datasets
    rows = []
    for ds in DS:
        for model in MODELS:
            x = m[(m.model == model) & (m.setting == "in-domain") & (m.source == ds)]
            rows.append([ds, MN[model], f"{x.auroc.mean():.3f}", f"{x.ece_raw.mean():.3f}", f"{x.ece_iso.mean():.3f}",
                         f"{x.fpr_inflation.mean():.2f}", f"{x.tpr_at_alpha.mean():.3f}",
                         f"{x['prec_at_pi0.01'].mean():.3f}", f"{x['prec_at_pi0.001'].mean():.3f}"])
    T["A2"] = table("**Table A2.** In-domain results for every dataset and model (mean over five seeds).",
                    ["Dataset", "Model", "AUROC", "ECE (raw)", "ECE (iso.)", "FPR/α", "TPR", "Prec (1%)",
                     "Prec (0.1%)"], rows, ["l", "l"] + ["r"] * 7)
    rows = []
    x = m[m.setting == "cross-dataset"]
    for model in MODELS:
        for (s, t), g in x[x.model == model].groupby(["source", "target"]):
            rows.append([MN[model], f"{s} → {t}", f"{g.auroc.mean():.3f}", f"{g.auroc.min():.2f} to {g.auroc.max():.2f}",
                         f"{g.ece_raw.mean():.3f}", f"{g.ece_platt.mean():.3f}", f"{g.ece_iso.mean():.3f}",
                         f"{g.fpr_inflation.mean():.1f}", f"{g.tpr_at_alpha.mean():.3f}"])
    T["A3"] = table("**Table A3.** Cross-dataset results for every directed pair and model (mean over five seeds; "
                    "the AUROC range is over seeds).",
                    ["Model", "Pair", "AUROC", "AUROC range", "ECE (raw)", "ECE (Platt)", "ECE (iso.)", "FPR/α", "TPR"],
                    rows, ["l", "l"] + ["r"] * 7)
    g = pd.read_csv(os.path.join(RES, "xd_group_recall_lgbm.csv"))
    g = g[g.source != g.target]
    rows = [[f"{a.source} → {a.target}", a.group, f"{int(a.n):,}", f"{a.rec:.3f}"] for a in g.itertuples()]
    T["A4"] = table("**Table A4.** Cross-dataset recall per coarse attack group at the source 1%-FPR threshold "
                    "(LightGBM, mean over five seeds; n is the number of target test flows in the group). Other holds "
                    "Heartbleed in CIC-IDS2017 and ARP Spoofing in ACI-IoT-2023.",
                    ["Pair", "Group", "n", "Recall"], rows, ["l", "l", "r", "r"])


# ---------------------------------------------------------------- Table 4 / A6 / A7
def t_os():
    d = r("os_detection_all.csv")
    og = r("extra_group.csv")
    rows = []
    L = d[d.model == "LGBM"].groupby(["dataset", "held_out"]).mean(numeric_only=True).reset_index()
    for ds in DS:
        x = L[L.dataset == ds]
        rows.append([ds, "LightGBM", len(x), f"{x.recall_seen_at_alpha.median():.3f}",
                     f"{x.recall_unseen_at_alpha.median():.3f}", int((x.recall_unseen_at_alpha < .5).sum()),
                     int((x["cm_0.05"] > .5).sum()), f"{x.auroc_det.median():.3f}", f"{x.auroc_nov.median():.3f}"])
    for model in MODELS:
        x = d[d.model == model].groupby(["dataset", "held_out"]).mean(numeric_only=True).reset_index()
        if x.empty:
            continue
        rows.append(["All datasets", MN[model], len(x), f"{x.recall_seen_at_alpha.median():.3f}",
                     f"{x.recall_unseen_at_alpha.median():.3f}", int((x.recall_unseen_at_alpha < .5).sum()),
                     int((x["cm_0.05"] > .5).sum()), f"{x.auroc_det.median():.3f}", f"{x.auroc_nov.median():.3f}"])
    if og is not None:
        G = og.groupby(["dataset", "held_out_group"]).mean(numeric_only=True).reset_index()
        rows.append(["CICFlowMeter (groups)", "LightGBM", len(G), "", f"{G.recall_unseen.median():.3f}",
                     int((G.recall_unseen < .5).sum()), int((G.cm_005 > .5).sum()), f"{G.auroc_det.median():.3f}",
                     f"{G.auroc_nov.median():.3f}"])
    T["os"] = table("**Table 4.** Held-out attack families (Protocol OS) and groups (Protocol OG). Recall is "
                    "measured at the source 1%-FPR threshold; medians are over held-out families or groups, each "
                    "averaged over five seeds. \"<0.5\" counts the families detected less than half of the time; "
                    "\"Conf.\" counts those with more than half of their flows given an isotonic-calibrated attack "
                    "probability below 0.05. The AUROC columns compare held-out with benign flows, using the "
                    "detector score and the benign-referenced novelty score.",
                    ["Data", "Model", "Units", "Recall (seen)", "Recall (held out)", "<0.5", "Conf.",
                     "AUROC det.", "AUROC nov."], rows, ["l", "l"] + ["r"] * 7)
    # A6 per family (LGBM + recall of other models)
    piv = d.groupby(["dataset", "held_out", "model"]).recall_unseen_at_alpha.mean().unstack()
    rows = []
    for a in L.itertuples():
        pr = piv.loc[(a.dataset, a.held_out)]
        rows.append([a.dataset, a.held_out, f"{a.stream_pi * 100:.2f}"] +
                    [f"{pr[mo]:.3f}" if mo in pr and not np.isnan(pr[mo]) else "" for mo in MODELS] +
                    [f"{a._asdict()['_10']:.3f}" if False else f"{getattr(a, '_14', np.nan):.3f}" if False else
                     f"{L.loc[(L.dataset == a.dataset) & (L.held_out == a.held_out), 'cm_0.05'].iloc[0]:.3f}",
                     f"{a.auroc_det:.3f}", f"{a.auroc_nov:.3f}"])
    T["A6"] = table("**Table A6.** Per-family results of Protocol OS (mean over five seeds). π is the realised "
                    "attack prevalence of the held-out-family stream in percent; recall is at the source "
                    "1%-FPR threshold for each model; the last three columns are for LightGBM. Family names are as labelled in each dataset (e.g. "
                    "'Infilteration' in CSE-CIC-IDS2018).",
                    ["Dataset", "Held-out family", "π (%)"] + [MN[mo] for mo in MODELS] +
                    ["Conf. miss", "AUROC det.", "AUROC nov."], rows, ["l", "l"] + ["r"] * 8)
    rows = []
    for model in MODELS:
        x = d[d.model == model].groupby(["dataset", "held_out"]).mean(numeric_only=True)
        if x.empty:
            continue
        rows.append([MN[model]] + [str(int((x[f"cm_{c}"] > .5).sum())) for c in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]])
    T["A7"] = table("**Table A7.** Number of held-out families (of 47) with more than half of their flows given "
                    "an isotonic-calibrated attack probability below the cutoff.",
                    ["Model", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"], rows)


# ---------------------------------------------------------------- Table 5
def t_prio():
    o = r("os_priorities.csv")
    x = r("xd_priorities.csv")
    N = json.load(open(os.path.join(REV, "revision_numbers.json")))
    P = pd.DataFrame(N["os_priorities"]).set_index(["priority", "budget"])
    rules = ["detector", "novelty", "combined", "fisher", "wfisher0.7", "stouffer", "stacked", "borda",
             "split0.1", "uncertainty", "proxy_switch"]
    xm = None
    if x is not None:
        x = x[x.model == "LGBM"]
        xm = x.groupby(["regime", "source", "target", "budget", "priority"]).recall.mean() \
            .groupby(["regime", "budget", "priority"]).mean()
    rows = []
    for pr in rules:
        row = [PRN[pr]]
        for reg in ["In-domain", "Cross-dataset"]:
            v = xm.get((reg, 0.01, pr), np.nan) if xm is not None else np.nan
            row.append("" if np.isnan(v) else f"{v:.3f}")
        for c in [0.01, 0.02, 0.05]:
            row.append(f"{P.loc[(pr, c), 'all']:.3f}")
        row += [f"{P.loc[(pr, 0.01), 'caught']:.3f}", f"{P.loc[(pr, 0.01), 'missed']:.3f}"]
        rows.append(row)
    rows.append(["Random queue", "0.010", "0.010", "0.010", "0.020", "0.050", "0.010", "0.010"])
    T["prio"] = table("**Table 5.** Attack recall inside the review queue (exact top-b fraction of a stream with "
                      "1% attack prevalence; LightGBM). In-domain and cross-dataset values are for b = 1% and are "
                      "averaged over dataset pairs; unseen-family values are averaged over the 47 held-out families "
                      "and split, for b = 1%, into families the detector catches or misses at its threshold "
                      "(groups defined on other seeds). Of the split budgets, only the 10% novelty share is shown; "
                      "the 25% and 50% shares did no better (Appendix Table A23 lists all rules).",
                      ["Ranking rule", "In-domain", "Cross-dataset", "Unseen b = 1%", "Unseen b = 2%",
                       "Unseen b = 5%", "Caught", "Missed"], rows)


# ---------------------------------------------------------------- Table 6
def t_budget():
    N = json.load(open(os.path.join(REV, "revision_numbers.json")))
    xt = pd.read_csv(os.path.join(RES, "xd_triage.csv"))
    xt = xt[(xt.model == "LGBM") & (xt.priority == "fisher") & (xt.budget == 0.01)]
    w = r("os_warmup.csv")
    w = w[(w.model == "LGBM") & (w.priority == "fisher") & (w.budget == 0.01)]
    xw = r("xd_warmup.csv")
    rows = []
    for lab, reg in [("In-domain", "in-domain"), ("Cross-dataset", "cross-dataset")]:
        z = xt[xt.setting == reg]
        row = [lab, q3(z.load_src), f"{z.recall_src.mean():.3f}"]
        if xw is not None:
            y = xw[(xw.model == "LGBM") & (xw.priority == "fisher") & (xw.budget == 0.01) & (xw.warmup == 0.2) &
                   (xw.regime == lab)]
            row += [q3(y.load), f"{y.recall.mean():.3f}"]
        else:
            row += ["", ""]
        rows.append(row)
    if "rule" in w:
        s, y = w[w.rule == "source"], w[(w.rule == "warmup") & (w.warmup == 0.2)]
        rows.append(["Unseen family", q3(s.load), f"{s.recall.mean():.3f}", q3(y.load), f"{y.recall.mean():.3f}"])
    T["budget"] = table("**Table 6.** Queue load (realised queue size divided by the planned size; median "
                        "[10th, 90th percentile] over runs) and attack recall inside the queue for the Fisher "
                        "priority at a 1% budget on streams with 1% prevalence (LightGBM). The warm-up quantile "
                        "uses the first 20% of each stream. Under cross-dataset shift the source threshold's recall comes "
                        "from a queue many times the planned size (load column). Behaviour on non-stationary streams is in "
                        "Table 7 and Appendix Table A9.",
                        ["Regime", "Source: load", "Source: recall", "Warm-up: load", "Warm-up: recall"], rows,
                        ["l", "r", "r", "r", "r"])
    # A9: non-stationary streams, OS and XD
    rows = []
    for src, tag in [("os_nonstationary.csv", "Unseen family"), ("xd_nonstationary.csv", None)]:
        s = r(src)
        if s is None:
            continue
        s = s[(s.model == "LGBM") & (s.budget == 0.01)]
        if tag:
            s = s.assign(regime=tag)
        for (reg, sc, rule), g in s.groupby(["regime", "scenario", "rule"]):
            rows.append([reg, {"burst_in_warmup": "Burst in warm-up", "late_campaign": "Late campaign"}[sc],
                         rule.capitalize() if rule != "warmup" else "Warm-up", q3(g.load), f"{g.recall.mean():.3f}"])
    T["A9"] = table("**Table A9.** Budget rules on non-stationary streams (Fisher priority, 1% budget, LightGBM). "
                    "\"Burst in warm-up\": 5% of the warm-up window is attacks and 1% of the rest; \"Late "
                    "campaign\": no attacks in the window and 1% after it. Load is median [10th, 90th percentile].",
                    ["Regime", "Scenario", "Rule", "Load", "Recall"], rows, ["l", "l", "l", "r", "r"])
    p = r("os_prevalence.csv")
    p = p[p.model == "LGBM"]
    rows = []
    for (pi, c, pr), g in p.groupby(["target_pi", "budget", "priority"]):
        rows.append([f"{100 * pi:g}%", f"{100 * g.real_pi.mean():.2f}%", f"{100 * c:g}%", PRN[pr], q3(g.load),
                     f"{g.recall.mean():.3f}", f"{g.precision.mean():.3f}"])
    T["A10"] = table("**Table A10.** Warm-up quantile under different attack prevalences (held-out-family streams, "
                     "LightGBM). Realised prevalence can fall below the target for families with few test flows. Recall and precision are "
                     "averaged separately over families with different realised prevalences, so the row means need "
                     "not satisfy precision = recall × π / (load × b).",
                     ["Target π", "Realised π", "Budget", "Priority", "Load", "Recall", "Precision"], rows,
                     ["r", "r", "r", "l", "r", "r", "r"])


# ---------------------------------------------------------------- Table A23
def t_allrules():
    from scipy.stats import wilcoxon
    o = r("os_priorities.csv")
    if o is None:
        return
    N = json.load(open(os.path.join(REV, "revision_numbers.json")))
    P = pd.DataFrame(N["os_priorities"]).set_index(["priority", "budget"])
    o = o[o.model == "LGBM"]
    fr = {b: o[o.budget == b].groupby(["dataset", "held_out", "priority"]).recall.mean().unstack() for b in (0.01, 0.02, 0.05)}
    rules = [r_ for r_ in PRN if r_ not in ("detector", "proxy_switch")]
    ps, dd = {}, {}
    for r_ in rules:
        d = (fr[0.01][r_] - fr[0.01]["detector"]).dropna()
        ps[r_], dd[r_] = wilcoxon(d[d != 0]).pvalue, d.mean()
    order = sorted(ps, key=ps.get)
    holm, run = {}, 0.0
    for i, r_ in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[r_]))
        holm[r_] = run
    fp = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
    rows = []
    for r_ in ["detector"] + rules + ["proxy_switch"]:
        row = [PRN[r_]] + [f"{P.loc[(r_, b), 'all']:.3f}" for b in (0.01, 0.02, 0.05)]
        row += [f"{P.loc[(r_, 0.01), 'caught']:.3f}", f"{P.loc[(r_, 0.01), 'missed']:.3f}"]
        row += ["", "", ""] if r_ not in ps else [f"{dd[r_]:+.3f}", fp(ps[r_]), fp(holm[r_])]
        rows.append(row)
    T["A23"] = table("**Table A23.** All ranking rules on the held-out families (Protocol OS, LightGBM, 47 families, "
                     "mean over five seeds). Recall is that of the exact top-b fraction of the stream; caught and "
                     "missed families are defined as in Table 5. The last three columns compare each rule with the "
                     "detector at b = 1%: mean paired difference over families, Wilcoxon signed-rank p-value and the "
                     "Holm-adjusted p-value over the " + str(len(rules)) + " rules of the family.",
                     ["Ranking rule", "b = 1%", "b = 2%", "b = 5%", "Caught, 1%", "Missed, 1%", "Difference",
                      "p", "p (Holm)"], rows, ["l"] + ["r"] * 8)


# ---------------------------------------------------------------- Table 7
def t_temporal():
    t = r("extra_temporal.csv")
    s = r("extra_temporal_stream.csv")
    if t is None:
        return
    rows = []
    names = {"T1-benign-time": "T1, benign split by time", "T1-random": "T1, random split",
             "T2-day": "T2, split by capture day", "T2-random": "T2, random split"}
    for pr in ["T1-benign-time", "T1-random", "T2-day", "T2-random"]:
        g = t[t.protocol == pr]
        row = [names[pr], f"{g.auroc.mean():.3f}", f"{g.ece_iso.mean():.3f}", f"{g.fpr_inflation.mean():.1f}",
               f"{g.tpr.mean():.3f}"]
        for rule in ["source", "warmup", "rolling", "online"]:
            z = s[(s.protocol == pr) & (s.rule == rule) & (s.priority == "fisher") & (s.budget == 0.01)]
            row.append(f"{z.load.mean():.2f} ({z.load_min10.mean():.2f} to {z.load_max10.mean():.2f})")
        rows.append(row)
    T["temporal"] = table("**Table 7.** Chronological protocols on CSE-CIC-IDS2018 (LightGBM, mean over five "
                          "seeds). Detection metrics are at the source 1%-FPR threshold. The last four columns give "
                          "the queue load of the Fisher priority at a 1% budget on the chronological test stream "
                          "with 1% attack prevalence, overall and (in parentheses) the lowest and highest load in "
                          "the ten consecutive tenths of the stream.",
                          ["Protocol", "AUROC", "ECE (iso.)", "FPR/α", "TPR", "Source", "Warm-up", "Rolling",
                           "Online"], rows, ["l"] + ["r"] * 8)


# ---------------------------------------------------------------- Table 8 + appendix extras
def t_cost():
    c = r("extra_cost.csv")
    if c is None:
        return
    rows = []
    def us(comp, m=None, n=None, th=1):
        z = c[(c.threads == th) & (c.component == comp)]
        if m is not None:
            z = z[z.model == m]
        if n is not None:
            z = z[z.n_ref == n]
        return z
    for m in MODELS:
        a, b = us("detector", m, th=1), us("detector", m, th=4)
        if len(a) and len(b):
            rows.append([f"{MN[m]} inference", "", f"{a.us_per_flow.iloc[0]:.2f}", f"{b.us_per_flow.iloc[0]:.2f}", ""])
    for algo in ["brute", "kd_tree", "ball_tree", "hnsw"]:
        for n in ([1000, 20000, 100000] if algo in ("kd_tree", "ball_tree") else [1000, 5000, 20000, 50000, 100000]):
            a, b = us("knn", algo, n, 1), us("knn", algo, n, 4)
            if len(a) and len(b):
                rc = b.recall_at5.iloc[0] if "recall_at5" in b and not np.isnan(b.recall_at5.iloc[0]) else np.nan
                rows.append([{"brute": "kNN, brute force", "kd_tree": "kNN, k-d tree", "ball_tree": "kNN, ball tree",
                              "hnsw": "kNN, HNSW (approximate)"}[algo], f"{n:,}",
                             f"{a.us_per_flow.iloc[0]:.2f}", f"{b.us_per_flow.iloc[0]:.2f}",
                             "" if np.isnan(rc) else f"{rc:.3f}"])
    a, b = us("conformal+fisher", th=1), us("conformal+fisher", th=4)
    if len(a) and len(b):
        rows.append(["Conformal p-values and Fisher", "", f"{a.us_per_flow.iloc[0]:.3f}", f"{b.us_per_flow.iloc[0]:.3f}", ""])
    T["cost"] = table("**Table 8.** Per-flow cost of each component of the triage layer on 50,000 CSE-CIC-IDS2018 "
                      "test flows (median of three repetitions for detectors). The last column is the share of "
                      "true five nearest neighbours returned by the approximate index; it depends on the sampled reference "
                      "flows and is therefore not monotone in the reference size (Section 4.11).",
                      ["Component", "Reference size", "µs per flow, 1 thread", "µs per flow, 4 threads",
                       "Neighbour recall"], rows, ["l", "r", "r", "r", "r"])


def t_extras():
    f = r("extra_features.csv")
    if f is not None:
        if "kind" not in f:
            f["kind"] = np.nan
        x = f[f.kind.isna() & (f.source != f.target)]
        rows = []
        for (s, t), g in x.groupby(["source", "target"]):
            a, b = g[g.features == "all"], g[g.features == "robust"]
            rows.append([f"{s} → {t}", f"{a.auroc.mean():.3f}", f"{b.auroc.mean():.3f}", f"{a.fpr_inflation.mean():.1f}",
                         f"{b.fpr_inflation.mean():.1f}", f"{a.tpr.mean():.3f}", f"{b.tpr.mean():.3f}"])
        dom = f[f.kind == "benign_domain_auc"]
        for (s, t), g in dom.groupby(["source", "target"]):
            rows.append([f"Benign {s} vs {t}", f"{g[g.features == 'all'].auroc.mean():.3f}",
                         f"{g[g.features == 'robust'].auroc.mean():.3f}", "", "", "", ""])
        T["A11"] = table("**Table A11.** Cross-dataset LightGBM with all 77 features and with the 24 tool-stable "
                         "features (mean over five seeds). The last rows give the AUROC of a classifier trained to "
                         "separate the benign flows of two datasets.",
                         ["Pair", "AUROC (all)", "AUROC (stable)", "FPR/α (all)", "FPR/α (stable)", "TPR (all)",
                          "TPR (stable)"], rows)
    s = r("extra_subsample.csv")
    if s is not None:
        rows = []
        for (d, p, fr), g in s.groupby(["dataset", "protocol", "benign_frac"]):
            rows.append([d, {"in-domain": "In-domain", "cross-dataset": "Cross-dataset", "unseen": "Unseen family"}[p],
                         f"{fr:g}", f"{g.auroc.mean():.3f}" if g.auroc.notna().any() else "",
                         f"{g.fpr_inflation.mean():.2f}" if g.fpr_inflation.notna().any() else "",
                         f"{g.recall_unseen.mean():.3f}" if "recall_unseen" in g and g.recall_unseen.notna().any() else "",
                         f"{g.rec_detector.mean():.3f}", f"{g.rec_fisher.mean():.3f}", f"{g.load_warmup.median():.2f}"])
        T["A12"] = table("**Table A12.** Effect of thinning the benign training flows of the subsampled copies "
                         "(LightGBM, five seeds). Queue recall and the warm-up load are for a 1% budget.",
                         ["Dataset", "Protocol", "Benign share", "AUROC", "FPR/α", "Recall (held out)",
                          "Queue recall, detector", "Queue recall, Fisher", "Warm-up load"], rows,
                         ["l", "l"] + ["r"] * 7)
    rf = r("extra_reference.csv")
    if rf is not None:
        fam = rf.groupby(["dataset", "held_out", "n_ref", "contamination"]).mean(numeric_only=True).reset_index()
        import glob as _glob
        wff = sorted(_glob.glob(os.path.join(REV, "extra_reference_wf_s*.csv")))
        wfm = {}
        if wff:
            wd = pd.concat(pd.read_csv(f_) for f_ in wff)
            wd = wd.groupby(["dataset", "held_out", "n_ref", "contamination"]).mean(numeric_only=True).reset_index()
            wfm = {(n_, c_): g_.rec_wfisher.mean() for (n_, c_), g_ in wd.groupby(["n_ref", "contamination"]) if n_ == 20000}
        rows = []
        for (n, c), g in fam.groupby(["n_ref", "contamination"]):
            w_ = wfm.get((n, c))
            rows.append([f"{int(n):,}", f"{100 * c:g}%", len(g), f"{g.auroc_nov.mean():.3f}", f"{g.rec_detector.mean():.3f}",
                         f"{g.rec_fisher.mean():.3f}", "" if w_ is None else f"{w_:.3f}",
                         f"{g.load_warmup.median():.2f}", f"{g.knn_us.median():.1f}"])
        T["A13"] = table("**Table A13.** Size and contamination of the benign novelty reference (Protocol OS, "
                         "LightGBM, seeds 0 to 2). Contamination replaces the stated share of the reference with "
                         "flows of the held-out family from the training period; families with too few such flows "
                         "are omitted at that level. Where a dataset has fewer benign training flows than the stated size, all of "
                         "them form the reference, and the row gives that size. Weighted Fisher (w = 0.7) comes from a separate "
                         "run with exact neighbour search on the same families and seeds and is given for the 20,000-flow "
                         "reference and its contaminated versions.",
                         ["Reference size", "Contamination", "Families", "AUROC nov.", "Queue recall, detector",
                          "Queue recall, Fisher", "Queue recall, weighted Fisher", "Warm-up load", "kNN µs per flow"],
                         rows, ["r"] * 9)
    v = r("xd_validity.csv")
    if v is not None:
        o = r("os_validity.csv").assign(regime="Unseen family")
        rows = []
        for (reg, m), g in pd.concat([v, o]).groupby(["regime", "model"]):
            rows.append([reg, MN[m], f"{g.fpr_pdet.mean():.4f}", f"{g.fpr_pB.mean():.4f}", f"{g.fpr_pB.max():.4f}"])
        T["A8"] = table("**Table A8.** Share of benign flows flagged by the conformal rules at α = 1% "
                        "($p_{\\det} \\le 0.01$ and $p_B \\le 0.01$). The conformal guarantee bounds the expected "
                        "share by 0.01 under Assumption S.",
                        ["Regime", "Model", "p_det, mean", "p_B, mean", "p_B, max"], rows, ["l", "l", "r", "r", "r"])
    g = r("extra_group.csv")
    if g is not None:
        G = g.groupby(["dataset", "held_out_group"]).mean(numeric_only=True).reset_index()
        gw = r("extra_group_wf.csv")
        wfg = {} if gw is None else {k: f"{v:.3f}" for k, v in
                                     gw.groupby(["dataset", "held_out_group"]).rec_wfisher.mean().items()}
        rows = [[a.dataset, a.held_out_group, f"{int(a.n_unseen):,}", f"{a.recall_unseen:.3f}", f"{a.cm_005:.3f}",
                 f"{a.auroc_det:.3f}", f"{a.auroc_nov:.3f}", f"{a.rec_detector:.3f}", f"{a.rec_fisher:.3f}",
                 wfg.get((a.dataset, a.held_out_group), "")]
                for a in G.itertuples()]
        T["A14"] = table("**Table A14.** Held-out coarse groups (Protocol OG, LightGBM, mean over five seeds). "
                         "Weighted Fisher (w = 0.7) comes from a separate run on the same groups and seeds.",
                         ["Dataset", "Held-out group", "Test flows", "Recall", "Conf. miss", "AUROC det.",
                          "AUROC nov.", "Queue recall, detector", "Queue recall, Fisher",
                          "Queue recall, weighted Fisher"], rows, ["l", "l"] + ["r"] * 8)


def t_zero():
    z = json.load(open(os.path.join(REV, "zero_features.json")))
    from revision_extra import ROBUST
    zf = sorted(set(sum(z.values(), [])))
    feats = zf + [f for f in ROBUST if f not in zf]
    rows = [[f] + ["✓" if f in z[d] else "" for d in CIC] + ["✓" if f in ROBUST else ""] for f in feats]
    T["A5"] = table("**Table A5.** Features that are identically zero in the benign flows of each CICFlowMeter "
                    "dataset (✓), and the 24 tool-stable features used in Section 4.7 (last column).",
                    ["Feature"] + CIC + ["Tool-stable"], rows, ["l", "l", "l", "l", "l"])


def t_adapt():
    a = r("xd_adapt.csv")
    if a is None:
        return
    cv = r("xd_adapt_curve.csv")
    pp = a.groupby(["source", "target", "budget", "variant"]).mean(numeric_only=True).reset_index()
    g = pp.groupby(["budget", "variant"]).mean(numeric_only=True)
    names = [("base_fisher", "Source pipeline, Fisher priority"),
             ("base_detector", "Source pipeline, detector priority"),
             ("TN_wfisher", "TN: per-network quantile features"),
             ("TR_wfisher", "TR: target novelty reference and calibration"),
             ("TNTR_wfisher", "TN and TR combined"),
             ("FB", "FB: analyst feedback"),
             ("FB_noexplore", "FB without random exploration")]
    rows = []
    for v, nm in names:
        row = [nm, "none" if not v.startswith("FB") else "reviewed flows"]
        for c in [0.01, 0.02, 0.05]:
            if (c, v) in g.index:
                z = g.loc[(c, v)]
                row.append(f"{z.recall:.3f} / {z.precision:.3f}")
            else:
                row.append("")
        rows.append(row)
    fc = os.path.join(REV, "xd_coral.csv")
    if os.path.exists(fc):
        co = pd.read_csv(fc).groupby(["source", "target", "budget", "variant"]).mean(numeric_only=True)
        co = co.reset_index().groupby(["budget", "variant"]).mean(numeric_only=True)
        for v, nm in [("CORAL_detector", "CORAL feature alignment, detector priority"),
                      ("CORAL_wfisher", "CORAL feature alignment, weighted Fisher")]:
            rows.append([nm, "none"] + [f"{co.loc[(c, v)].recall:.3f} / {co.loc[(c, v)].precision:.3f}"
                                         if (c, v) in co.index else "" for c in [0.01, 0.02, 0.05]])
    c4 = cv[cv.segment == 4].groupby(["source", "target", "budget", "variant"]).mean(numeric_only=True)
    c4 = c4.reset_index().groupby(["budget", "variant"]).mean(numeric_only=True)
    row = ["FB, last fifth of the stream", "reviewed flows"]
    for c in [0.01, 0.02, 0.05]:
        z = c4.loc[(c, "FB")]
        row.append(f"{z.recall:.3f} / {z.precision:.3f}")
    rows.append(row)
    T["adapt"] = table("**Table 9.** Adapting the triage pipeline to a new network (six cross-dataset pairs, "
                       "LightGBM, mean over pairs of five seeds). Each cell gives queue recall / queue precision on "
                       "the 80% of the target stream after the warm-up window, at 1% attack prevalence. The CORAL rows come "
                       "from a separate run on the same pairs and seeds.",
                       ["Variant", "Target labels", "1% budget", "2% budget", "5% budget"], rows,
                       ["l", "l", "r", "r", "r"])
    rows = []
    for (src, tgt), z in pp[pp.budget == 0.01].groupby(["source", "target"]):
        z = z.set_index("variant")
        rows.append([src, tgt] + [f"{z.loc[v, 'recall']:.3f}" for v, _ in names] +
                    [f"{z.loc['FB', 'labels']:.0f}", f"{z.loc['FB', 'n_attack_labels']:.0f}"])
    T["A15"] = table("**Table A15.** Queue recall at a 1% budget for each directed pair (LightGBM, mean over five "
                     "seeds) and number of reviewed flows (and attacks among them) that the feedback loop collected.",
                     ["Source", "Target", "Fisher", "Detector", "TN", "TR", "TN+TR", "FB", "FB, no expl.",
                      "Labels", "Attack labels"], rows, ["l", "l"] + ["r"] * 9)


def t_ctrl():
    import glob
    fs = sorted(glob.glob(os.path.join(REV, "xd_adapt_ctrl*.csv")))
    if not fs:
        return
    c = pd.concat([pd.read_csv(x) for x in fs])
    a = pd.read_csv(os.path.join(REV, "xd_adapt.csv"))
    a = a[(a.budget == 0.01) & (a.variant == "FB")]
    d = pd.concat([a, c]).groupby(["source", "target", "variant"]).recall.mean().unstack()
    cols = [("FB", "Feedback"), ("RL", "Random labels"), ("US", "Uncertainty sampling"),
            ("FB_n10", "Feedback, 10% flipped"), ("FB_n20", "Feedback, 20% flipped"),
            ("PU", "Unreviewed as benign")]
    cols = [cv for cv in cols if cv[0] in d.columns]
    rows = [[s_, t_] + [f"{r[v]:.3f}" if pd.notna(r[v]) else "" for v, _ in cols] for (s_, t_), r in d.iterrows()]
    rows.append(["Mean", ""] + [f"{d[v].mean():.3f}" if d[v].notna().all() else "" for v, _ in cols])
    T["A16"] = table("**Table A16.** Controls for the feedback loop: queue recall at a 1% budget for each directed "
                     "pair (LightGBM, mean over five seeds). Random labels and uncertainty sampling retrain on the "
                     "same number of target flows as feedback, chosen uniformly at random or closest to a predicted "
                     "attack probability of 0.5; the flipped columns flip the stated share of analyst verdicts" + (
                     ", and the last column adds the randomly drawn flows to training as benign without review"
                     + ("." if c[c.variant == "PU"].shape[0] >= 30 else
                        " (" + ("seed " if c[c.variant == 'PU'].seed.nunique() == 1 else "seeds ") + ", ".join(str(int(x)) for x in sorted(c[c.variant == 'PU'].seed.unique())) + " only; blank cells were not run).") if "PU" in d.columns else "."),
                     ["Source", "Target"] + [n for _, n in cols], rows, ["l", "l"] + ["r"] * len(cols))


def t_sync():
    f = os.path.join(REV, "stream_sync.csv")
    if not os.path.exists(f):
        return
    s = pd.read_csv(f)
    g = s.groupby(["mode", "workers", "batch"]).agg(
        fps=("flows_per_s", "median"), load=("load_union", "median"), lo=("load_tenth_min", "median"),
        hi=("load_tenth_max", "median"), wait=("lock_wait_share", "median"), reps=("rep", "nunique"),
        fpsw=("flows_per_s_wall", "median")).reset_index()
    b_thr = g[(g["mode"] == "threads") & (g.workers == 1)].fps.iloc[0]
    b_sh = g[(g["mode"] == "shards") & (g.workers == 1)].fps.iloc[0]
    name = {"threads": "Threads in one process", "shards": "Independent shards", "shared": "Shards, shared threshold"}
    rows = []
    for m_ in ["threads", "shards", "shared"]:
        for x in g[g["mode"] == m_].itertuples():
            base = b_thr if m_ == "threads" else b_sh
            rows.append([name[m_], str(int(x.workers)), f"{int(x.batch):,}", f"{x.fps:,.0f}",
                         "" if np.isnan(x.fpsw) else f"{x.fpsw:,.0f}", f"{x.fps / base:.2f}",
                         f"{x.load:.2f}", "" if np.isnan(x.lo) else f"{x.lo:.2f} to {x.hi:.2f}",
                         "" if m_ != "shared" else f"{100 * x.wait:.2f}"])
    T["A22"] = table("**Table A22.** Parallel execution of the triage layer on the CSE-CIC-IDS2018 test stream "
                     "(LightGBM, HNSW reference, weighted Fisher priority, 1% budget; median of "
                     f"{int(g.reps.max())} repetitions; one machine with four cores). Throughput counts the time from the "
                     "end of the warm-up to the last flow of the slowest worker; the next column also counts loading the "
                     "models and the warm-up in each worker. Speed-up is relative to one thread for threads and to one "
                     "shard for shards. With a shared threshold, workers flag each micro-batch with the "
                     "current threshold and update it under a lock; lock wait is the largest share of a worker's "
                     "time spent waiting for the lock.",
                     ["Mode", "Workers", "Batch", "Flows per s", "Incl. loading", "Speed-up", "Load", "Load per tenth",
                      "Lock wait (%)"], rows, ["l", "r", "r", "r", "r", "r", "r", "r", "r"])


def t_stream():
    f = os.path.join(REV, "stream_batch.csv")
    if os.path.exists(f):
        b = pd.read_csv(f)
        sy = pd.read_csv(os.path.join(REV, "stream_sync.csv"))
        one = sy[(sy.workers == 1) & (sy.batch == 4096)].groupby("mode").flows_per_s.median()
        one_lo, one_hi = one.min(), one.max()
        rows = [[f"{int(x.batch):,}", f"{x.flows_per_s:,.0f}", f"{x.us_per_flow:.1f}", f"{x.p50_ms:.1f}",
                 f"{x.p99_ms:.1f}", f"{x.load:.2f}"] for x in b.itertuples()]
        T["A17"] = table("**Table A17.** Streaming throughput of the full triage layer (CSE-CIC-IDS2018 test stream, "
                         "LightGBM, HNSW reference of 20,000 benign flows, weighted Fisher priority, online threshold "
                         "update; one process, one thread; p50 and p99 are batch latencies). Load is the queue size divided by the "
                         "1% budget. Each row is one run that also records the latency of every batch; the one-worker rows of "
                         f"Table A22, medians of repeated runs at batches of 4,096, give {one_lo:,.0f} to {one_hi:,.0f} flows per "
                         "second. Runs on several cores are in Table A22.",
                         ["Batch", "Flows per s", "µs per flow", "p50 (ms)", "p99 (ms)", "Load"], rows,
                         ["r", "r", "r", "r", "r", "r"])
    f = os.path.join(REV, "os_mixed.csv")
    if os.path.exists(f):
        x = pd.read_csv(f)
        x = x[x["mode"] == "topk"]
        g = x.groupby(["pi_u", "budget", "rule", "dataset", "held_out"])[["recall_unseen", "recall_known", "precision"]].mean()
        g = g.groupby(["pi_u", "budget", "rule"]).mean()
        names = {"detector": "Detector", "fisher": "Fisher", "wfisher0.7": "Weighted Fisher, w = 0.7"}
        rows = [[f"{pu:.1%}", f"{1 - pu - 0.01 + pu:.1%}" if False else f"{(0.01 - pu):.1%}", f"{c:.0%}", names[r_],
                 f"{v.recall_unseen:.3f}", f"{v.recall_known:.3f}", f"{v.precision:.3f}"]
                for (pu, c, r_), v in g.iterrows()]
        T["A18"] = table("**Table A18.** Held-out families arriving together with known attacks (Protocol OS, "
                         "LightGBM, mean over families of five seeds). Each stream holds all benign test flows, the "
                         "held-out family and the known attack families of the same dataset at the stated "
                         "prevalences; the queue is the top share of the stream given by the budget.",
                         ["Unseen prevalence", "Known prevalence", "Budget", "Priority", "Recall (unseen)",
                          "Recall (known)", "Precision"], rows, ["r", "r", "r", "l", "r", "r", "r"])
    f = os.path.join(REV, "os_novalt.csv")
    if os.path.exists(f):
        x = pd.read_csv(f)
        a = x.groupby(["novelty", "dataset", "held_out"]).auroc_nov.mean().groupby("novelty").mean()
        g = x.groupby(["novelty", "budget", "rule", "dataset", "held_out"]).recall.mean().groupby(["novelty", "budget", "rule"]).mean().unstack()
        names = {"knn": "kNN distance (k = 5)", "iforest": "Isolation forest", "lof": "Local outlier factor", "maha": "Mahalanobis distance"}
        rows = []
        for k in ["knn", "iforest", "lof", "maha"]:
            for c in [0.01, 0.02, 0.05]:
                v = g.loc[(k, c)]
                rows.append([names[k] if c == 0.01 else "", f"{a[k]:.3f}" if c == 0.01 else "", f"{c:.0%}",
                             f"{v['novelty']:.3f}", f"{v['detector']:.3f}", f"{v['fisher']:.3f}", f"{v['wfisher0.7']:.3f}"])
        T["A19"] = table("**Table A19.** Novelty scorers inside the same conformal wrapper (Protocol OS, LightGBM, "
                         "47 held-out families, mean over five seeds). Each scorer is fitted on the benign training "
                         "flows; recall is that of the top share of the stream given by the budget. The table comes from a "
                         "separate run of all four scorers, so its kNN rows can differ from Table 5 in the third decimal.",
                         ["Novelty score", "AUROC nov.", "Budget", "Novelty", "Detector", "Fisher", "Weighted Fisher"],
                         rows, ["l", "r", "r", "r", "r", "r", "r"])
    f = os.path.join(REV, "extra_gamma.csv")
    if os.path.exists(f):
        x = pd.read_csv(f)
        g = x.groupby(["protocol", "rule"])[["load", "load_min10", "load_max10", "recall"]].mean()
        lab = lambda r_: ("batched, 4,096 flows, g = 0.02" if r_.startswith("batched") else "g = " + r_[1:])
        g = g.reset_index()
        g["_o"] = g.rule.map(lambda r_: 99.0 if r_.startswith("batched") else float(r_[1:]))
        g = g.sort_values(["protocol", "_o"]).set_index(["protocol", "rule"])
        rows = [[pr.split("-")[0], lab(r_), f"{v.load:.2f}", f"{v.load_min10:.2f} to {v.load_max10:.2f}", f"{v.recall:.3f}"]
                for (pr, r_), v in g.iterrows()]
        T["A20"] = table("**Table A20.** Step size of the online update on the chronological streams (Fisher priority, "
                         "1% budget, LightGBM, mean over five seeds). The step is g times the spread between the median "
                         "and the 99.9th percentile of the warm-up priorities; the main experiments use g = 0.02. The "
                         "range gives the lowest and highest load among the ten consecutive tenths of the stream, averaged over "
                         "seeds. The table comes from a separate run, so the g = 0.02 rows can differ from Table 7 in "
                         "the second decimal.",
                         ["Protocol", "Update", "Load", "Load range over tenths", "Recall"], rows,
                         ["l", "l", "r", "r", "r"])

    f = os.path.join(REV, "extra_shards_het.csv")
    if os.path.exists(f):
        x = pd.read_csv(f)
        ok = x.groupby(["seed", "protocol"]).scheme.nunique()
        x = x.set_index(["seed", "protocol"]).loc[ok[ok == 4].index].reset_index()
        fx = os.path.join(REV, "extra_shards_fix.csv")
        if os.path.exists(fx):
            x = pd.concat([x, pd.read_csv(fx).query("scheme != 'global'")])
        fc = os.path.join(REV, "extra_shards_coord.csv")
        if os.path.exists(fc):
            x = pd.concat([x, pd.read_csv(fc).query("scheme.str.endswith('perworker')", engine="python")])
        names = {"global": "One threshold", "protocol": "One shard per protocol",
                 "skew4": "Four shards, attacks on one", "sync4": "Four shards, shared batched threshold",
                 "protocol_sync256": "Per protocol, shared threshold every 256 flows",
                 "protocol_share": "Per protocol, budgets from warm-up share",
                 "skew4_sync256": "Attacks on one, shared threshold every 256 flows",
                 "skew4_share": "Attacks on one, budgets from warm-up share",
                 "protocol_coord256_perworker": "Per protocol, coordinator, 256-flow batches",
                 "protocol_coord1024_perworker": "Per protocol, coordinator, 1,024-flow batches",
                 "skew4_coord256_perworker": "Attacks on one, coordinator, 256-flow batches",
                 "skew4_coord1024_perworker": "Attacks on one, coordinator, 1,024-flow batches"}
        rows = []
        for pr in ["T1-benign-time", "T2-day"]:
            for sc in ["global", "protocol", "skew4", "sync4", "protocol_sync256", "skew4_sync256",
                       "protocol_share", "skew4_share", "protocol_coord256_perworker",
                       "protocol_coord1024_perworker", "skew4_coord256_perworker", "skew4_coord1024_perworker"]:
                v = x[(x.protocol == pr) & (x.scheme == sc)]
                if v.empty:
                    continue
                rows.append([pr.split("-")[0] if sc == "global" else "", names[sc], str(v.seed.nunique()),
                             str(int(v.n_shards.iloc[0])), f"{v.load.mean():.2f}",
                             f"{v.load_shard_min.min():.2f} to {v.load_shard_max.max():.2f}",
                             f"{v.recall.mean():.3f}", f"{v.precision.mean():.3f}",
                             f"{100 * (v.precision * v.load * 0.01 / v.recall).mean():.2f}%"])
        T["A21"] = table("**Table A21.** Online budget on heterogeneous shards of the chronological streams "
                         "(CSE-CIC-IDS2018, LightGBM, weighted Fisher priority, 1% budget). Each shard has its own "
                         "warm-up threshold and online update unless the threshold is shared; the shared batched threshold is "
                         "updated after every 4,096 flows of the merged stream, the other shared threshold after every "
                         "256 flows with the same per-flow step. With budgets from the warm-up share, one warm-up "
                         "threshold is taken over the merged stream, each shard's budget is the share of its own "
                         "warm-up flows above it, and each shard then runs its own update with that budget. With the coordinator, "
                         "each shard flags a batch with the threshold it received after its own previous batch and sends its "
                         "excess to one shared threshold, with the step divided by the number of shards (Section 4.11). Load is the queue size divided by "
                         "the budget; shard loads give the lowest and highest shard over all seeds. Recall, precision and load are "
                         "measured after the warm-up windows; because attacks cluster in time, the attack share of "
                         "that part of the stream (last column, precision × load × budget / recall, mean over seeds) "
                         "differs from the 1% of the whole stream, which is why precision and recall do not match "
                         "as they would at 1% prevalence.",
                         ["Protocol", "Sharding", "Seeds", "Shards", "Load", "Shard loads", "Recall", "Precision",
                          "Attack share"],
                         rows, ["l", "l", "r", "r", "r", "r", "r", "r", "r"])


# ---------------------------------------------------------------- Tables A24 and A25
MACH = {"cloud": "Cloud VM, 4 cores", "mac": "Laptop, 10 cores"}


def dist_frames():
    out = []
    for k in ["cloud", "mac"]:
        d = r(f"dist_bench_{k}.csv")
        if d is not None:
            d = d.copy()
            d["host"] = k
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else None


def t_dist():
    d = dist_frames()
    if d is None:
        return
    ind = d[d["mode"] == "indep"].groupby(["host", "workers"]).agg(
        fps=("flows_per_s", "median"), load=("load", "median"), lo=("load_tenth_min", "median"),
        hi=("load_tenth_max", "median"), wlo=("load_worker_min", "median"), whi=("load_worker_max", "median"),
        reps=("rep", "nunique")).reset_index()
    rows = []
    for h in ["cloud", "mac"]:
        g = ind[ind.host == h]
        if not len(g):
            continue
        b = g[g.workers == 1].fps.iloc[0]
        for x in g.itertuples():
            rows.append([MACH[h], str(int(x.workers)), f"{x.fps:,.0f}", f"{x.fps / b:.2f}",
                         f"{x.fps / b / x.workers:.2f}", f"{x.load:.3f}", f"{x.lo:.2f} to {x.hi:.2f}",
                         f"{x.wlo:.2f} to {x.whi:.2f}"])
    T["A24"] = table("**Table A24.** Independent workers on two machines (CSE-CIC-IDS2018 test stream of Table A22, "
                     "LightGBM, HNSW reference, weighted Fisher priority, 1% budget, batches of 4,096 flows; median of "
                     f"{int(ind.reps.max())} repetitions). Each worker is a separate process with its own models, "
                     "warm-up and threshold. Throughput counts the time from the end of the warm-up to the last flow "
                     "of the slowest worker. Efficiency is the speed-up divided by the number of workers.",
                     ["Machine", "Workers", "Flows per s", "Speed-up", "Efficiency", "Load", "Load per tenth",
                      "Worker loads"], rows, ["l", "r", "r", "r", "r", "r", "r", "r"])
    c = d[d["mode"] == "coord"]
    g = c.groupby(["host", "workers", "batch", "delay_ms", "step"]).agg(
        fps=("flows_per_s", "median"), load=("load", "median"), lo=("load_tenth_min", "median"),
        hi=("load_tenth_max", "median"), st=("staleness", "median"), up=("updates", "median"),
        reps=("rep", "nunique")).reset_index()
    rows = []
    for h in ["cloud", "mac"]:
        for (P, s), gg in g[g.host == h].groupby(["workers", "batch"]):
            full_up = gg[(gg.delay_ms == 0)].up.max()
            for dl in sorted(gg.delay_ms.unique()):
                fu = gg[(gg.delay_ms == dl) & (gg.step == "full")]
                pw = gg[(gg.delay_ms == dl) & (gg.step == "perworker")]
                if not (len(fu) and len(pw)):
                    continue
                fu, pw = fu.iloc[0], pw.iloc[0]
                rows.append([MACH[h], str(int(P)), f"{int(s):,}", str(int(dl)), f"{pw.st:.1f}",
                             f"{100 * pw.up / full_up:.0f}", f"{fu.load:.2f}", f"{fu.lo:.2f} to {fu.hi:.2f}",
                             f"{pw.load:.3f}", f"{pw.lo:.2f} to {pw.hi:.2f}", f"{pw.fps:,.0f}"])
    T["A25"] = table("**Table A25.** One review threshold held by a coordinator process and shared by all workers "
                     "over TCP (same stream and settings as Table A24; median of "
                     f"{int(g.reps.max())} repetitions). After each batch a worker flags it with the last threshold it "
                     "has received, sends its excess over the budget and goes on without waiting; the coordinator "
                     "updates the threshold and answers each worker in order, holding every reply for the stated "
                     "delay. Staleness is the mean number of updates applied by the coordinator between issuing a "
                     "threshold and receiving the batch flagged with it; updates applied is the share of all batch "
                     "updates the coordinator processed before the stream ended. The update uses either the per-flow "
                     "step of a single process (full step) or that step divided by the number of workers (step/P); "
                     "the last column is the throughput with step/P.",
                     ["Machine", "Workers", "Batch", "Delay (ms)", "Staleness", "Updates applied (%)", "Load, full step",
                      "Per tenth, full step", "Load, step/P", "Per tenth, step/P", "Flows per s"], rows,
                     ["l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"])


if __name__ == "__main__":
    for f in (t_data, t_xd, t_os, t_prio, t_allrules, t_budget, t_temporal, t_cost, t_extras, t_zero, t_adapt, t_ctrl, t_stream, t_sync, t_dist):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("fail", f.__name__, repr(e))
    json.dump(T, open(os.path.join(REV, "manuscript_tables.json"), "w"), indent=1, ensure_ascii=False)

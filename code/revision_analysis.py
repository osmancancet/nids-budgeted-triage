"""Revision analyses computed from the score dumps of revision_dump.py.

  python revision_analysis.py os    # unseen-family analyses (all models)
  python revision_analysis.py xd    # in-domain / cross-dataset analyses

Every quantity is computed from stored per-flow scores of models trained
exactly as in run_experiments.py; no model is retrained here. Outputs go to
results/revision/*.csv.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, binomtest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.path.join(ROOT, "data", "dumps")
OUT = os.path.join(ROOT, "results", "revision")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.01
BUDGETS = [0.01, 0.02, 0.05]
WARMUPS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
PREVS = [0.001, 0.005, 0.01, 0.02, 0.05]
CM_CUTS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
SPLIT_SHARES = [0.1, 0.25, 0.5]
W_FISHER = [0.6, 0.7, 0.8, 0.9]


# ------------------------------------------------------------------ helpers
def pvals(cal, s):
    cal = np.sort(np.asarray(cal, dtype=np.float64))
    n = len(cal)
    ge = n - np.searchsorted(cal, np.asarray(s, dtype=np.float64), side="left")
    return (1.0 + ge) / (n + 1.0)


def make_stream(y, pi, rng, attack_mask=None):
    """Identical to run_experiments.make_stream."""
    ben = np.flatnonzero(y == 0)
    att = np.flatnonzero((y == 1) if attack_mask is None else attack_mask)
    n_att = min(len(att), int(round(pi / (1 - pi) * len(ben))))
    idx = np.concatenate([ben, rng.choice(att, n_att, replace=False)])
    return rng.permutation(idx)


class Ctx:
    """Conformal p-values, calibrators and learned combiners fitted on the source calibration split."""

    def __init__(self, cal_det, cal_nov, cal_y):
        cal_det = cal_det.astype(np.float64)
        cal_nov = cal_nov.astype(np.float64)
        b = cal_y == 0
        self.cd, self.cn = cal_det[b], cal_nov[b]
        self.tau = np.quantile(self.cd, 1 - ALPHA)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(cal_det, cal_y)
        # learned (stacked) combination of the two p-values, fitted on source calibration data
        pd_, pn_ = pvals(self.cd, cal_det), pvals(self.cn, cal_nov)
        Z = np.c_[-np.log(pd_), -np.log(pn_)]
        self.stack = LogisticRegression(C=1.0, max_iter=1000).fit(Z, cal_y)
        self.cal_det, self.cal_nov, self.cal_y = cal_det, cal_nov, cal_y

    def p(self, det, nov):
        return pvals(self.cd, det), pvals(self.cn, nov)


def priorities(ctx, det, nov, extra=False):
    det = det.astype(np.float64)
    nov = nov.astype(np.float64)
    pdt, pnv = ctx.p(det, nov)
    tie = 1e-9 * det + 1e-12 * nov
    ld, ln = -np.log(pdt), -np.log(pnv)
    pr = {"detector": ld + 1e-9 * det,
          "novelty": ln + 1e-12 * nov,
          "combined": -np.log(np.minimum(1.0, 2.0 * np.minimum(pdt, pnv))) + tie,
          "fisher": -np.log(chi2.sf(2 * (ld + ln), df=4)) + tie}
    if extra:
        for w in W_FISHER:
            pr[f"wfisher{w}"] = w * ld + (1 - w) * ln + tie
        pr["stouffer"] = norm.isf(pdt) + norm.isf(pnv) + tie
        pr["stacked"] = ctx.stack.decision_function(np.c_[ld, ln]) + tie
        # rank aggregation (Borda): mean of within-stream ranks of the two p-values
        rd = pd.Series(ld).rank(method="average").to_numpy()
        rn = pd.Series(ln).rank(method="average").to_numpy()
        pr["borda"] = (rd + rn) / 2 + tie
        # model uncertainty (maximum-softmax style) on isotonic-calibrated probabilities
        pi = ctx.iso.predict(det)
        pr["uncertainty"] = -np.abs(pi - 0.5) + 1e-12 * det
    return pr, pdt, pnv


def topk_mask(s, c):
    k = int(np.floor(c * len(s)))
    m = np.zeros(len(s), bool)
    if k > 0:
        m[np.argpartition(-s, k - 1)[:k]] = True
    return m


def split_budget_mask(det_pr, nov_pr, c, share):
    """Reserve `share` of the budget for the most novel flows not already queued by the detector."""
    n = len(det_pr)
    k = int(np.floor(c * n))
    kn = int(round(share * k))
    m = np.zeros(n, bool)
    if k - kn > 0:
        m[np.argpartition(-det_pr, k - kn - 1)[:k - kn]] = True
    if kn > 0:
        nv = np.where(m, -np.inf, nov_pr)
        m[np.argpartition(-nv, kn - 1)[:kn]] = True
    return m


def queue_stats(q, y, prefix=""):
    tp = (q & (y == 1)).sum()
    return {prefix + "recall": tp / max((y == 1).sum(), 1), prefix + "precision": tp / max(q.sum(), 1)}


def warmup_threshold(s, w_frac, c):
    w = int(w_frac * len(s))
    return np.quantile(s[:w], 1 - c)


@njit(cache=True)
def _online(s, c, t0, gamma):
    t = t0
    flags = np.empty(len(s), np.bool_)
    for i in range(len(s)):
        f = s[i] >= t
        flags[i] = f
        t += gamma * ((1.0 if f else 0.0) - c)
    return flags


def online_quantile(s, c, t0, gamma):
    """Label-free online quantile tracking (the update of adaptive conformal inference,
    Gibbs and Candes 2021, applied to the queue rate): t <- t + gamma * (flag - c)."""
    return _online(np.ascontiguousarray(s, dtype=np.float64), float(c), float(t0), float(gamma))


def rolling_quantile_flags(s, c, window, step):
    """Threshold re-estimated every `step` flows as the (1-c)-quantile of the last `window` flows."""
    flags = np.zeros(len(s), bool)
    start = window
    for i in range(start, len(s), step):
        t = np.quantile(s[i - window:i], 1 - c)
        flags[i:i + step] = s[i:i + step] >= t
    return flags, start


# --------------------------------------------------------------- OS analyses
def load_os():
    files = sorted(glob.glob(os.path.join(DUMP, "os", "*.npz")))
    models = os.environ.get("MODELS")
    if models:
        files = [f for f in files if os.path.basename(f)[:-4].split("__")[-2] in models.split(",")]
    for f in files:
        z = np.load(f, allow_pickle=False)
        base = os.path.basename(f)[:-4].split("__")
        yield dict(dataset=str(z["dataset"]), held_out=str(z["held_out"]), model=base[-2],
                   seed=int(base[-1][1:])), z


def run_os():
    det_rows, tri_rows, cm_rows, wu_rows, prev_rows, prio_rows, ns_rows, comp_rows, val_rows = ([] for _ in range(9))
    for key, z in load_os():
        seed = key["seed"]
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        det, nov, y, unseen = z["det"].astype(np.float64), z["nov"].astype(np.float64), z["y"], z["unseen"]
        p_iso = ctx.iso.predict(det[unseen])
        row = dict(key, n_unseen=int(unseen.sum()),
                   recall_unseen_at_alpha=(det[unseen] >= ctx.tau).mean(),
                   recall_seen_at_alpha=(det[(y == 1) & ~unseen] >= ctx.tau).mean(),
                   fpr_at_alpha=(det[y == 0] >= ctx.tau).mean())
        for c in CM_CUTS:
            row[f"cm_{c}"] = (p_iso < c).mean()
        # stream identical to the original protocol (seed + 11)
        rng = np.random.default_rng(seed + 11)
        idx = make_stream(y, 0.01, rng, attack_mask=unseen)
        ys = y[idx]
        row["stream_pi"] = ys.mean()
        row["auroc_det"] = roc_auc_score(ys, det[idx])
        row["auroc_nov"] = roc_auc_score(ys, nov[idx])
        det_rows.append(row)
        pr, pdt, pnv = priorities(ctx, det[idx], nov[idx], extra=True)
        # validity of the conformal false-alarm guarantee on benign flows (same network, unseen attacks)
        pB = np.minimum(1, 2 * np.minimum(pdt, pnv))
        b = ys == 0
        val_rows.append(dict(key, regime="Unseen family", fpr_pdet=(pdt[b] <= ALPHA).mean(),
                             fpr_pB=(pB[b] <= ALPHA).mean()))
        # top-b recall for every priority, plus split-budget rules
        for c in BUDGETS:
            for name, s in pr.items():
                q = topk_mask(s, c)
                prio_rows.append(dict(key, budget=c, priority=name, **queue_stats(q, ys)))
            for sh in SPLIT_SHARES:
                q = split_budget_mask(pr["detector"], pr["novelty"], c, sh)
                prio_rows.append(dict(key, budget=c, priority=f"split{sh}", **queue_stats(q, ys)))
            # label-free proxy rule: use Fisher only when the warm-up window shows an excess of
            # flows that are novel but not detector-flagged (one-sided binomial test vs. source rate)
            w = int(0.2 * len(ys))
            ref = np.mean((pvals(ctx.cn, ctx.cal_nov[ctx.cal_y == 0]) <= 0.01) &
                          (pvals(ctx.cd, ctx.cal_det[ctx.cal_y == 0]) > 0.1))
            cnt = int(((pnv[:w] <= 0.01) & (pdt[:w] > 0.1)).sum())
            pt = binomtest(cnt, w, max(ref, 1e-6), alternative="greater").pvalue
            use_f = pt < 0.01
            q = topk_mask(pr["fisher"] if use_f else pr["detector"], c)
            prio_rows.append(dict(key, budget=c, priority="proxy_switch", proxy_fired=bool(use_f),
                                  proxy_p=pt, **queue_stats(q, ys)))
        # warm-up sensitivity (common evaluation segment: last 50% of the stream)
        n = len(ys)
        ev = slice(n // 2, n)
        # source-threshold rule: quantile of the priority on a source calibration stream at 1% prevalence
        cidx = make_stream(ctx.cal_y, 0.01, np.random.default_rng(seed + 1000))
        cpr, _, _ = priorities(ctx, ctx.cal_det[cidx], ctx.cal_nov[cidx])
        for c, pname in [(c, p) for c in BUDGETS for p in ["detector", "fisher"]]:
            s = pr[pname]
            for wf in WARMUPS:
                t = warmup_threshold(s, wf, c)
                q = s[ev] >= t
                wu_rows.append(dict(key, rule="warmup", budget=c, priority=pname, warmup=wf, n_warm=int(wf * n),
                                    n_eval=n - n // 2, load=q.mean() / c, recall=q[ys[ev] == 1].mean(),
                                    tie_share=np.mean(s[ev] == t)))
            t = np.quantile(cpr[pname], 1 - c)
            q = s[ev] >= t
            wu_rows.append(dict(key, rule="source", budget=c, priority=pname, warmup=0.0, n_warm=0,
                                n_eval=n - n // 2, load=q.mean() / c, recall=q[ys[ev] == 1].mean(),
                                tie_share=np.mean(s[ev] == t)))
        # queue precision and false-alarm composition with the label-free rule (warm-up 20%)
        w = int(0.2 * n)
        for c in BUDGETS:
            for name in ["detector", "fisher", "combined"]:
                s = pr[name]
                t = np.quantile(s[:w], 1 - c)
                q = s[w:] >= t
                yy = ys[w:]
                fa = q & (yy == 0)
                nov_driven = (pnv[w:] < pdt[w:])
                comp_rows.append(dict(key, regime="Unseen family", budget=c, priority=name,
                                      load=q.mean() / c, recall=q[yy == 1].mean(),
                                      precision=(q & (yy == 1)).sum() / max(q.sum(), 1),
                                      fa_per_tp=fa.sum() / max((q & (yy == 1)).sum(), 1),
                                      fa_novelty_share=(fa & nov_driven).sum() / max(fa.sum(), 1)))
        # non-stationary streams: (a) attack burst inside the warm-up window (adversarial
        # flooding to inflate the threshold), (b) campaign starting after warm-up
        for scen in ["burst_in_warmup", "late_campaign"]:
            rng2 = np.random.default_rng(seed + 23)
            ben = rng2.permutation(np.flatnonzero(y == 0))
            att = rng2.permutation(np.flatnonzero(unseen))
            nb = len(ben)
            nw = int(0.2 * nb)
            if scen == "burst_in_warmup":
                # warm-up at 5% attack prevalence, remainder at 1%
                a1 = min(len(att), int(round(0.05 / 0.95 * nw)))
                a2 = min(len(att) - a1, int(round(0.01 / 0.99 * (nb - nw))))
            else:
                a1 = 0
                a2 = min(len(att), int(round(0.01 / 0.99 * (nb - nw))))
            wi = rng2.permutation(np.r_[ben[:nw], att[:a1]])
            ri = rng2.permutation(np.r_[ben[nw:], att[a1:a1 + a2]])
            sidx = np.r_[wi, ri]
            yy = y[sidx]
            prs, _, _ = priorities(ctx, det[sidx], nov[sidx])
            for c in BUDGETS:
                s = prs["fisher"]
                ww = len(wi)
                rest = slice(ww, len(s))
                yr = yy[rest]
                t = np.quantile(s[:ww], 1 - c)
                q = s[rest] >= t
                ns_rows.append(dict(key, scenario=scen, budget=c, rule="warmup", load=q.mean() / c,
                                    recall=q[yr == 1].mean()))
                fl = online_quantile(s, c, t, gamma=0.02 * (np.quantile(s[:ww], 0.999) - np.quantile(s[:ww], 0.5)))
                q = fl[rest]
                ns_rows.append(dict(key, scenario=scen, budget=c, rule="online", load=q.mean() / c,
                                    recall=q[yr == 1].mean()))
                fl, st = rolling_quantile_flags(s, c, window=ww, step=500)
                q = fl[rest]
                ns_rows.append(dict(key, scenario=scen, budget=c, rule="rolling", load=q.mean() / c,
                                    recall=q[yr == 1].mean()))
        # prevalence sensitivity with the label-free rule
        for pi in PREVS:
            rng3 = np.random.default_rng(seed + 31)
            sidx = make_stream(y, pi, rng3, attack_mask=unseen)
            yy = y[sidx]
            real_pi = yy.mean()
            prs, _, _ = priorities(ctx, det[sidx], nov[sidx])
            w = int(0.2 * len(yy))
            for c in BUDGETS:
                for name in ["detector", "fisher"]:
                    s = prs[name]
                    t = np.quantile(s[:w], 1 - c)
                    q = s[w:] >= t
                    yr = yy[w:]
                    prev_rows.append(dict(key, regime="Unseen family", target_pi=pi, real_pi=real_pi,
                                          budget=c, priority=name, load=q.mean() / c,
                                          recall=q[yr == 1].mean(),
                                          precision=(q & (yr == 1)).sum() / max(q.sum(), 1)))
        print("os", key, flush=True)
    for nm, rows in [("os_detection_all", det_rows), ("os_priorities", prio_rows), ("os_warmup", wu_rows),
                     ("os_nonstationary", ns_rows), ("os_prevalence", prev_rows),
                     ("os_queue_composition", comp_rows), ("os_validity", val_rows)]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, nm + ".csv"), index=False)


# --------------------------------------------------------------- XD analyses
def load_xd():
    for f in sorted(glob.glob(os.path.join(DUMP, "xd", "*.npz"))):
        src, model, s = os.path.basename(f)[:-4].split("__")
        yield dict(source=src, model=model, seed=int(s[1:])), np.load(f, allow_pickle=False)


def run_xd():
    rows, wu_rows, prev_rows, comp_rows, val_rows, cost_rows, prio_rows, ns_rows = ([] for _ in range(8))
    for key, z in load_xd():
        seed = key["seed"]
        ctx = Ctx(z["cal_det"], z["cal_nov"], z["cal_y"])
        for tgt in z["targets"]:
            tgt = str(tgt)
            det, nov, y = z[f"{tgt}__det"].astype(np.float64), z[f"{tgt}__nov"].astype(np.float64), z[f"{tgt}__y"]
            reg = "In-domain" if tgt == key["source"] else "Cross-dataset"
            k2 = dict(key, target=tgt, regime=reg)
            n_t = len(y)
            cost_rows.append(dict(k2, n=n_t, det_us=1e6 * float(z[f"{tgt}__t_det"]) / n_t,
                                  nov_us=1e6 * float(z[f"{tgt}__t_nov"]) / n_t, fit_s=float(z["fit_s"])))
            rows.append(dict(k2, auroc=roc_auc_score(y, det), fpr_inflation=(det[y == 0] >= ctx.tau).mean() / ALPHA,
                             tpr=(det[y == 1] >= ctx.tau).mean()))
            rng = np.random.default_rng(seed + 7)
            idx = make_stream(y, 0.01, rng)
            ys = y[idx]
            extra = key["model"] == "LGBM"
            pr, pdt, pnv = priorities(ctx, det[idx], nov[idx], extra=extra)
            b = ys == 0
            pB = np.minimum(1, 2 * np.minimum(pdt, pnv))
            val_rows.append(dict(k2, fpr_pdet=(pdt[b] <= ALPHA).mean(), fpr_pB=(pB[b] <= ALPHA).mean()))
            n = len(ys)
            if extra:
                for c in BUDGETS:
                    for name, s in pr.items():
                        prio_rows.append(dict(k2, budget=c, priority=name, **queue_stats(topk_mask(s, c), ys)))
                    for sh in SPLIT_SHARES:
                        q = split_budget_mask(pr["detector"], pr["novelty"], c, sh)
                        prio_rows.append(dict(k2, budget=c, priority=f"split{sh}", **queue_stats(q, ys)))
            ev = slice(n // 2, n)
            for c, pname in [(c, p) for c in BUDGETS for p in ["detector", "fisher"]]:
                s = pr[pname]
                for wf in WARMUPS:
                    t = warmup_threshold(s, wf, c)
                    q = s[ev] >= t
                    wu_rows.append(dict(k2, budget=c, priority=pname, warmup=wf, n_warm=int(wf * n), n_eval=n - n // 2,
                                        load=q.mean() / c, recall=q[ys[ev] == 1].mean(),
                                        tie_share=np.mean(s[ev] == t)))
            w = int(0.2 * n)
            for c in BUDGETS:
                for name in ["detector", "fisher", "combined"]:
                    s = pr[name]
                    t = np.quantile(s[:w], 1 - c)
                    q = s[w:] >= t
                    yy = ys[w:]
                    fa = q & (yy == 0)
                    comp_rows.append(dict(k2, budget=c, priority=name, load=q.mean() / c,
                                          recall=q[yy == 1].mean(),
                                          precision=(q & (yy == 1)).sum() / max(q.sum(), 1),
                                          fa_per_tp=fa.sum() / max((q & (yy == 1)).sum(), 1),
                                          fa_novelty_share=(fa & (pnv[w:] < pdt[w:])).sum() / max(fa.sum(), 1)))
            # attack burst in the warm-up window (in-domain and cross-dataset)
            rng2 = np.random.default_rng(seed + 23)
            ben = rng2.permutation(np.flatnonzero(y == 0))
            att = rng2.permutation(np.flatnonzero(y == 1))
            nb = len(ben)
            nw = int(0.2 * nb)
            for scen, pw in [("burst_in_warmup", 0.05), ("late_campaign", 0.0)]:
                a1 = min(len(att), int(round(pw / (1 - pw) * nw)))
                a2 = min(len(att) - a1, int(round(0.01 / 0.99 * (nb - nw))))
                wi = rng2.permutation(np.r_[ben[:nw], att[:a1]])
                ri = rng2.permutation(np.r_[ben[nw:], att[a1:a1 + a2]])
                sidx = np.r_[wi, ri]
                yy = y[sidx]
                prs, _, _ = priorities(ctx, det[sidx], nov[sidx])
                s = prs["fisher"]
                ww = len(wi)
                yr = yy[ww:]
                for c in BUDGETS:
                    t = np.quantile(s[:ww], 1 - c)
                    q = s[ww:] >= t
                    ns_rows.append(dict(k2, scenario=scen, budget=c, rule="warmup", load=q.mean() / c,
                                        recall=q[yr == 1].mean()))
                    fl = online_quantile(s, c, t, gamma=0.02 * (np.quantile(s[:ww], 0.999) - np.quantile(s[:ww], 0.5)))
                    ns_rows.append(dict(k2, scenario=scen, budget=c, rule="online", load=fl[ww:].mean() / c,
                                        recall=fl[ww:][yr == 1].mean()))
                    fl, _ = rolling_quantile_flags(s, c, window=ww, step=500)
                    ns_rows.append(dict(k2, scenario=scen, budget=c, rule="rolling", load=fl[ww:].mean() / c,
                                        recall=fl[ww:][yr == 1].mean()))
            for pi in PREVS:
                rng3 = np.random.default_rng(seed + 31)
                sidx = make_stream(y, pi, rng3)
                yy = y[sidx]
                prs, _, _ = priorities(ctx, det[sidx], nov[sidx])
                w = int(0.2 * len(yy))
                for c in BUDGETS:
                    for name in ["detector", "fisher"]:
                        s = prs[name]
                        t = np.quantile(s[:w], 1 - c)
                        q = s[w:] >= t
                        yr = yy[w:]
                        prev_rows.append(dict(k2, target_pi=pi, real_pi=yy.mean(), budget=c, priority=name,
                                              load=q.mean() / c, recall=q[yr == 1].mean(),
                                              precision=(q & (yr == 1)).sum() / max(q.sum(), 1)))
        print("xd", key, flush=True)
    for nm, r in [("xd_detection_check", rows), ("xd_warmup", wu_rows), ("xd_prevalence", prev_rows),
                  ("xd_queue_composition", comp_rows), ("xd_validity", val_rows), ("xd_cost", cost_rows),
                  ("xd_priorities", prio_rows), ("xd_nonstationary", ns_rows)]:
        pd.DataFrame(r).to_csv(os.path.join(OUT, nm + ".csv"), index=False)


if __name__ == "__main__":
    {"os": run_os, "xd": run_xd}[sys.argv[1]]()

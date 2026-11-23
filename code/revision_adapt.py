"""Adaptation to a new network (cross-dataset pairs, LightGBM, five seeds).

Three ways of using the target network's own traffic, compared with the
source-trained pipeline on the same target streams (1% attack prevalence,
first 20% of each stream as warm-up, evaluation on the remaining 80%):

  TN  label-free: features of each network are mapped to their own empirical
      quantiles (source: training split; target: unlabeled warm-up window),
      and the detector and novelty reference are trained in that space;
  TR  label-free: the kNN novelty reference and both conformal calibrations
      are taken from the unlabeled target warm-up window;
  FB  analyst feedback: the flows that analysts review inside the budget
      (queue plus a fixed exploration share drawn at random) are labeled and
      added to the source training data; the detector is retrained after the
      warm-up and after every further fifth of the stream.

All settings below are fixed in advance and not tuned on results.

  python revision_adapt.py        -> results/revision/xd_adapt.csv, xd_adapt_curve.csv
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import QuantileTransformer

import run_experiments as R
from nidslib import Detector, KNNNovelty, LogScaler, make_model
from revision_analysis import pvals, make_stream, OUT

BUDGETS = [0.01, 0.02, 0.05]
WARM = 0.2
CUTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # feedback retraining points
EXPLORE = 0.2                           # share of the budget reviewed at random
TGT_WEIGHT = 0.2                        # total weight of labeled target flows relative to the source
W = 0.7                                 # weighted Fisher weight (nested choice, Section 4.3)


def fisher(pd_, pn_):
    return -np.log(chi2.sf(2 * (-np.log(pd_) - np.log(pn_)), df=4))


def wfisher(pd_, pn_, w=W):
    return -w * np.log(pd_) - (1 - w) * np.log(pn_)


def evaluate(q, y):
    tp = (q & (y == 1)).sum()
    return dict(recall=tp / max((y == 1).sum(), 1), precision=tp / max(q.sum(), 1), n_flag=int(q.sum()))


def warm_rule(s, n_w, c):
    t = np.quantile(s[:n_w], 1 - c)
    return s[n_w:] >= t


def qt(X, seed):
    return QuantileTransformer(n_quantiles=1000, subsample=200000, random_state=seed).fit(X)


class FB:
    """Feedback loop: retrain on source + labels of reviewed target flows."""

    def __init__(self, seed, Xs, ys, Xt, yt, s0, c, explore):
        self.seed, self.Xs, self.ys = seed, Xs, ys
        self.Xt, self.yt, self.c, self.eps = Xt, yt, c, explore
        self.s0 = s0
        self.rng = np.random.default_rng(seed + 501)

    def review(self, s_seg, idx, thr=None):
        """Queue of a segment: priority above threshold (or top share retrospectively) + random exploration."""
        n = len(idx)
        k_rand = int(round(self.eps * self.c * n))
        if thr is None:
            k = int(np.floor((1 - self.eps) * self.c * n))
            q = np.zeros(n, bool)
            if k > 0:
                q[np.argpartition(-s_seg, k - 1)[:k]] = True
        else:
            q = s_seg >= thr
        free = np.flatnonzero(~q)
        q[self.rng.choice(free, min(k_rand, len(free)), replace=False)] = True
        return q

    def run(self):
        n = len(self.yt)
        b = [int(f * n) for f in CUTS]
        lab = np.zeros(n, bool)
        flags = np.zeros(n, bool)
        seg0 = np.arange(b[0], b[1])
        lab[seg0[self.review(self.s0[seg0], seg0)]] = True
        prev_scores, prev_idx = None, seg0
        curve, fits = [], 0
        for k in range(1, len(CUTS) - 1):
            # retrain with every label collected so far
            L = np.flatnonzero(lab)
            Xtr = np.vstack([self.Xs, self.Xt[L]])
            ytr = np.r_[self.ys, self.yt[L]]
            wt = np.r_[np.ones(len(self.ys)), np.full(len(L), TGT_WEIGHT * len(self.ys) / max(len(L), 1))]
            det = Detector("LGBM", self.seed)
            det.scaler = LogScaler().fit(Xtr)
            det.clf = make_model("LGBM", self.seed).fit(det.scaler.transform(Xtr), ytr, sample_weight=wt)
            fits += 1
            # label-free threshold: quantile of the new scores on the previous segment
            s_prev = det.score(self.Xt[prev_idx])
            thr = np.quantile(s_prev, 1 - (1 - self.eps) * self.c)
            seg = np.arange(b[k], b[k + 1])
            s_seg = det.score(self.Xt[seg])
            q = self.review(s_seg, seg, thr)
            flags[seg] = q
            lab[seg[q]] = True
            curve.append(dict(segment=k, n_labels=int(lab[:b[k]].sum()),
                              n_attack_labels=int((lab[:b[k]] & (self.yt[:b[k]] == 1)).sum()),
                              **evaluate(q, self.yt[seg])))
            prev_idx = seg
        ev = slice(b[1], n)
        out = evaluate(flags[ev], self.yt[ev])
        out.update(load=flags[ev].mean() / self.c, n_labels=int(lab.sum()),
                   n_attack_labels=int((lab & (self.yt == 1)).sum()), fits=fits)
        return out, curve


def main():
    data = {n: R.load(n) for n in R.CIC}
    feats = data[R.CIC[0]][1]
    rows, curves = [], []
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    srcs = os.environ.get("SOURCES", ",".join(R.CIC)).split(",")
    tag = os.environ.get("TAG", "")
    for seed in seeds:
        parts = {n: R.split(data[n][0], seed) for n in R.CIC}
        for src in srcs:
            df = data[src][0]
            Xs_all, ys_all, ps = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
            Xs, ys = Xs_all[ps == 0], ys_all[ps == 0]
            Xc, yc = Xs_all[ps == 1], ys_all[ps == 1]
            # baseline source model (identical to the main pipeline)
            t0 = time.time()
            base = R.Fitted("LGBM", seed, Xs, ys, Xc, yc)
            # TN: quantile-normalised source model
            qs = qt(Xs, seed)
            det_tn = Detector("LGBM", seed).fit(qs.transform(Xs), ys)
            nov_tn = KNNNovelty(seed=seed).fit(qs.transform(Xs)[ys == 0])
            cal_tn_d = det_tn.score(qs.transform(Xc))[yc == 0]
            cal_tn_n = nov_tn.score(qs.transform(Xc))[yc == 0]
            pc, nc = base.scores(Xc)
            cal_d, cal_n = pc[yc == 0], nc[yc == 0]
            print("fit", seed, src, "%.0fs" % (time.time() - t0), flush=True)
            for tgt in [t for t in R.CIC if t != src]:
                tdf = data[tgt][0]
                tm = parts[tgt] == 2
                Xt0, yt0 = tdf.loc[tm, feats].to_numpy(np.float32), tdf.loc[tm, "label"].to_numpy()
                idx = make_stream(yt0, 0.01, np.random.default_rng(seed + 7))
                Xt, yt = Xt0[idx], yt0[idx]
                n = len(yt)
                n_w = int(WARM * n)
                ye = yt[n_w:]
                key = dict(source=src, target=tgt, seed=seed, n=n, n_warm=n_w, n_attacks_eval=int(ye.sum()))
                # scores
                det, nov = base.scores(Xt)
                qtg = qt(Xt[:n_w], seed)                       # target map from the unlabeled warm-up only
                Zt = qtg.transform(Xt)
                det2, nov2 = det_tn.score(Zt), nov_tn.score(Zt)
                # TR: novelty reference and calibration from the unlabeled warm-up window
                n_ref = int(0.7 * n_w)
                nov_tr = KNNNovelty(seed=seed).fit(Xt[:n_ref])
                ntr = nov_tr.score(Xt)
                nov_tr2 = KNNNovelty(seed=seed).fit(Zt[:n_ref])
                ntr2 = nov_tr2.score(Zt)
                pr = {}
                pd0, pn0 = pvals(cal_d, det), pvals(cal_n, nov)
                pr["base_detector"] = det
                pr["base_fisher"] = fisher(pd0, pn0) + 1e-9 * det
                pr["base_wfisher"] = wfisher(pd0, pn0) + 1e-9 * det
                pd1, pn1 = pvals(cal_tn_d, det2), pvals(cal_tn_n, nov2)
                pr["TN_detector"] = det2
                pr["TN_wfisher"] = wfisher(pd1, pn1) + 1e-9 * det2
                # TR calibration: warm-up flows held out from the reference
                pd2 = pvals(det[n_ref:n_w], det)
                pn2 = pvals(ntr[n_ref:n_w], ntr)
                pr["TR_novelty"] = ntr
                pr["TR_wfisher"] = wfisher(pd2, pn2) + 1e-9 * det
                pd3 = pvals(det2[n_ref:n_w], det2)
                pn3 = pvals(ntr2[n_ref:n_w], ntr2)
                pr["TNTR_wfisher"] = wfisher(pd3, pn3) + 1e-9 * det2
                for c in BUDGETS:
                    for name, s in pr.items():
                        q = warm_rule(s, n_w, c)
                        rows.append(dict(key, budget=c, variant=name, load=q.mean() / c, labels=0, **evaluate(q, ye)))
                    for eps in ([EXPLORE, 0.0] if c == 0.01 else [EXPLORE]):
                        t1 = time.time()
                        fb = FB(seed, Xs, ys, Xt, yt, pr["base_fisher"], c, eps)
                        out, curve = fb.run()
                        name = "FB" if eps > 0 else "FB_noexplore"
                        rows.append(dict(key, budget=c, variant=name, labels=out.pop("n_labels"), **out))
                        for cv in curve:
                            curves.append(dict(key, budget=c, variant=name, **cv))
                        print("fb", seed, src, tgt, c, name, "recall %.3f" % out["recall"], "%.0fs" % (time.time() - t1),
                              flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"xd_adapt{tag}.csv"), index=False)
            pd.DataFrame(curves).to_csv(os.path.join(OUT, f"xd_adapt_curve{tag}.csv"), index=False)


if __name__ == "__main__":
    main()

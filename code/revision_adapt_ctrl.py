"""Controls for the analyst-feedback loop of revision_adapt.py (1% budget, same pairs, seeds and streams).

  RL       the same number of target labels per fifth, drawn uniformly at random instead of from the queue;
           the queue itself is the top of the retrained model's ranking at the full budget
  US       the same number of target labels per fifth, taken from the flows whose predicted attack
           probability is closest to 0.5 (uncertainty sampling); queue as in RL
  FB_n10   feedback loop as in FB, with 10% of the analyst labels flipped
  FB_n20   as FB_n10 with 20% flipped
  FB_d1    feedback loop as in FB, with analyst verdicts arriving one fifth of the stream late
           (labels from a segment are used only from the retraining after the next segment)
  PU       label-free: the flows drawn for RL are added to training as benign without review

  python revision_adapt_ctrl.py   -> results/revision/xd_adapt_ctrl.csv
"""
import os
import numpy as np
import pandas as pd

import run_experiments as R
from nidslib import Detector, LogScaler, make_model
from revision_analysis import pvals, make_stream, OUT
from revision_adapt import FB, CUTS, EXPLORE, TGT_WEIGHT, WARM, fisher, evaluate

C = 0.01


class Ctrl(FB):
    def __init__(self, *a, mode="FB", noise=0.0, p0=None, delay=0, **k):
        super().__init__(*a, **k)
        self.mode, self.noise, self.p0, self.delay = mode, noise, p0, delay
        r = np.random.default_rng(self.seed + 911)
        self.ytrain = self.yt.copy()
        if noise > 0:
            flip = r.random(len(self.yt)) < noise
            self.ytrain[flip] = 1 - self.ytrain[flip]
        if mode == "PU":
            self.ytrain[:] = 0

    def pick(self, idx, prob):
        """Flows labelled in a segment for the RL and US controls."""
        m = int(round(self.c * len(idx)))
        if self.mode in ("RL", "PU"):
            return idx[self.rng.choice(len(idx), m, replace=False)]
        return idx[np.argsort(np.abs(prob - 0.5))[:m]]

    def run(self):
        n = len(self.yt)
        b = [int(f * n) for f in CUTS]
        lab = np.zeros(n, bool)
        seg_of = np.searchsorted(np.array(b[1:]), np.arange(n), side="right")
        flags = np.zeros(n, bool)
        seg0 = np.arange(b[0], b[1])
        if self.mode in ("RL", "US", "PU"):
            lab[self.pick(seg0, self.p0[seg0])] = True
        else:
            lab[seg0[self.review(self.s0[seg0], seg0)]] = True
        prev_idx = seg0
        for k in range(1, len(CUTS) - 1):
            L = np.flatnonzero(lab & (seg_of <= k - 1 - self.delay))
            Xtr = np.vstack([self.Xs, self.Xt[L]])
            ytr = np.r_[self.ys, self.ytrain[L]]
            wt = np.r_[np.ones(len(self.ys)), np.full(len(L), TGT_WEIGHT * len(self.ys) / max(len(L), 1))]
            det = Detector("LGBM", self.seed)
            det.scaler = LogScaler().fit(Xtr)
            det.clf = make_model("LGBM", self.seed).fit(det.scaler.transform(Xtr), ytr, sample_weight=wt)
            seg = np.arange(b[k], b[k + 1])
            s_prev, s_seg = det.score(self.Xt[prev_idx]), det.score(self.Xt[seg])
            if self.mode in ("RL", "US", "PU"):
                q = s_seg >= np.quantile(s_prev, 1 - self.c)
                lab[self.pick(seg, s_seg)] = True
            else:
                q = self.review(s_seg, seg, np.quantile(s_prev, 1 - (1 - self.eps) * self.c))
                lab[seg[q]] = True
            flags[seg] = q
            prev_idx = seg
        ev = slice(b[1], n)
        out = evaluate(flags[ev], self.yt[ev])
        out.update(load=flags[ev].mean() / self.c, labels=int(lab.sum()),
                   n_attack_labels=int((lab & (self.yt == 1)).sum()))
        return out


def main():
    data = {n: R.load(n) for n in R.CIC}
    feats = data[R.CIC[0]][1]
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    tag = os.environ.get("TAG", "")
    rows = []
    for seed in seeds:
        parts = {n: R.split(data[n][0], seed) for n in R.CIC}
        for src in R.CIC:
            df = data[src][0]
            Xa, ya, ps = df[feats].to_numpy(np.float32), df["label"].to_numpy(), parts[src]
            Xs, ys, Xc, yc = Xa[ps == 0], ya[ps == 0], Xa[ps == 1], ya[ps == 1]
            base = R.Fitted("LGBM", seed, Xs, ys, Xc, yc)
            pc, nc = base.scores(Xc)
            cal_d, cal_n = pc[yc == 0], nc[yc == 0]
            for tgt in [t for t in R.CIC if t != src]:
                tdf = data[tgt][0]
                tm = parts[tgt] == 2
                Xt0, yt0 = tdf.loc[tm, feats].to_numpy(np.float32), tdf.loc[tm, "label"].to_numpy()
                idx = make_stream(yt0, 0.01, np.random.default_rng(seed + 7))
                Xt, yt = Xt0[idx], yt0[idx]
                det, nov = base.scores(Xt)
                s0 = fisher(pvals(cal_d, det), pvals(cal_n, nov)) + 1e-9 * det
                key = dict(source=src, target=tgt, seed=seed, budget=C)
                variants = [("RL", dict(mode="RL")), ("US", dict(mode="US")),
                            ("FB_n10", dict(noise=0.1)), ("FB_n20", dict(noise=0.2)), ("FB_d1", dict(delay=1)),
                            ("PU", dict(mode="PU"))]
                only = os.environ.get("VARIANTS")
                if only:
                    variants = [v for v in variants if v[0] in only.split(",")]
                for name, kw in variants:
                    out = Ctrl(seed, Xs, ys, Xt, yt, s0, C, EXPLORE, p0=det, **kw).run()
                    rows.append(dict(key, variant=name, **out))
                    print(seed, src, tgt, name, "recall %.3f" % out["recall"], flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f"xd_adapt_ctrl{tag}.csv"), index=False)


if __name__ == "__main__":
    main()

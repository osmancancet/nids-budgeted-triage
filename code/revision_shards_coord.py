"""Shared threshold held by a coordinator on heterogeneous shards of the chronological streams
(same streams, seeds, detector and priority as revision_shards_fix.py).

Each shard collects its flows in time order into batches of bs flows. When a batch is full, the shard
flags it with the threshold it received after its own previous batch and sends its excess over the
budget; the coordinator applies the update and returns the new threshold to that shard only. Other
shards keep using older values until their own next batch, as in the asynchronous coordinator of
dist/dist_bench.py with no network delay. The step is the per-flow step of a single process (full) or
that step divided by the number of shards (perworker).

  python revision_shards_coord.py            -> results/revision/extra_shards_coord.csv (batches 256, 1024)
  python revision_shards_coord.py small      -> results/revision/extra_shards_coord_small.csv (batches 16, 64)
"""
import os
import sys
import numpy as np
import pandas as pd

import run_experiments as R
from revision_analysis import priorities, online_quantile
from revision_extra import fit_eval, OUT, SEEDS

SMALL = len(sys.argv) > 1 and sys.argv[1] == "small"
BATCHES = [16, 64] if SMALL else [256, 1024]
OUTFILE = "extra_shards_coord_small.csv" if SMALL else "extra_shards_coord.csv"
from prepare import CIC_FEATURES
from revision_shards_fix import C, warm


def run_coord(s, shard, bs, divide):
    w, t, g = warm(s)
    keys = np.unique(shard)
    if divide:
        g = g / len(keys)
    f = np.zeros(len(s), bool)
    f[:w] = s[:w] >= t
    tk = {k: t for k in keys}
    buf = {k: [] for k in keys}
    for i in range(w, len(s)):
        k = shard[i]
        buf[k].append(i)
        if len(buf[k]) == bs:
            ix = np.array(buf[k])
            f[ix] = s[ix] >= tk[k]
            t += g * (f[ix].sum() - C * len(ix))
            tk[k] = t
            buf[k] = []
    for k in keys:
        if buf[k]:
            ix = np.array(buf[k])
            f[ix] = s[ix] >= tk[k]
    post = np.zeros(len(s), bool)
    post[w:] = True
    return f, post


def main():

    df = pd.read_parquet(os.path.join(R.PROC, "CSE-CIC-IDS2018-temporal.parquet"))
    feats = CIC_FEATURES + ["Protocol"]
    days = sorted(df.day.unique())
    rows = []
    for seed in SEEDS[:4]:
        rng = np.random.default_rng(seed)
        parts = []
        for fam_, g in df.groupby("family"):
            if fam_ != "Benign" and len(g) > 30000:
                g = g.iloc[np.sort(rng.choice(len(g), 30000, replace=False))]
            parts.append(g)
        d = pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)
        if (d.label == 0).sum() > 200000:
            bi = np.flatnonzero(d.label.to_numpy() == 0)
            d = d.drop(index=rng.choice(bi, len(bi) - 200000, replace=False)).reset_index(drop=True)
        X, y, ts = d[feats].to_numpy(np.float32), d.label.to_numpy(), d.ts.to_numpy()
        for proto in ["T1-benign-time", "T2-day"]:
            part = np.empty(len(d), np.int8)
            if proto.startswith("T1"):
                ben = np.flatnonzero(y == 0)
                a, b = int(0.5 * len(ben)), int(0.7 * len(ben))
                part[ben[:a]], part[ben[a:b]], part[ben[b:]] = 0, 1, 2
                for _, idx in d[d.label == 1].groupby("family").indices.items():
                    idx = rng.permutation(d.index[d.label == 1].to_numpy()[idx])
                    a, b = int(round(.5 * len(idx))), int(round(.7 * len(idx)))
                    part[idx[:a]], part[idx[a:b]], part[idx[b:]] = 0, 1, 2
            else:
                early = np.isin(d.day.to_numpy(), days[:5])
                tr_pool = rng.permutation(np.flatnonzero(early))
                k = int(0.7 * len(tr_pool))
                part[tr_pool[:k]], part[tr_pool[k:]] = 0, 1
                part[~early] = 2
            F, ctx = fit_eval("LGBM", seed, X[part == 0], y[part == 0], X[part == 1], y[part == 1])
            te = np.flatnonzero(part == 2)
            te = te[np.argsort(ts[te], kind="stable")]
            det, nov = F.scores(X[te])
            yt = y[te]
            ben, att = np.flatnonzero(yt == 0), np.flatnonzero(yt == 1)
            n_att = min(len(att), int(round(0.01 / 0.99 * len(ben))))
            sel = np.sort(np.r_[ben, rng.choice(att, n_att, replace=False)])
            ys = yt[sel]
            s = priorities(ctx, det[sel], nov[sel], extra=True)[0]["wfisher0.7"]
            proto_te = X[te][sel][:, feats.index("Protocol")]
            schemes = {"global": np.zeros(len(s), int), "protocol": proto_te.astype(int)}
            sk = np.zeros(len(s), int)
            bi = np.flatnonzero(ys == 0)
            sk[bi] = np.arange(len(bi)) % 4
            schemes["skew4"] = sk
            runs = [("global", schemes["global"])]
            for base in ["protocol", "skew4"]:
                for bs in BATCHES:
                    for stp in ["full", "perworker"]:
                        runs.append((f"{base}_coord{bs}_{stp}", schemes[base]))
            for name, shard in runs:
                if name == "global":
                    w, t, g = warm(s)
                    fl = online_quantile(s, C, t, g)
                    post = np.zeros(len(s), bool); post[w:] = True
                else:
                    _, bs, stp = name.split("_")
                    fl, post = run_coord(s, shard, int(bs[5:]), stp == "perworker")
                loads = [fl[post & (shard == k)].mean() / C for k in np.unique(shard)]
                shares = [(shard[post] == k).mean() for k in np.unique(shard)]
                rows.append(dict(seed=seed, protocol=proto, scheme=name, n_shards=len(loads),
                                 load=fl[post].mean() / C, load_shard_min=min(loads), load_shard_max=max(loads),
                                 size_share_min=min(shares), size_share_max=max(shares),
                                 recall=fl[post & (ys == 1)].mean(),
                                 precision=(fl[post] & (ys[post] == 1)).sum() / max(fl[post].sum(), 1)))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, OUTFILE), index=False)



if __name__ == "__main__":
    main()

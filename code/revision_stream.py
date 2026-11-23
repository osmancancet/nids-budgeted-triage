"""Streaming throughput of the triage layer (CSE-CIC-IDS2018, LightGBM, seed 0).

The full per-flow path of Section 3 is run on micro-batches of a test stream:
log scaling, LightGBM inference, approximate kNN novelty (HNSW, 20,000 benign reference
flows), conformal p-values, weighted Fisher priority (w = 0.7) and the online quantile
update of the review threshold, which carries its state from batch to batch.

  batch     micro-batch sizes 256 to 16,384 flows, one worker, one thread
  shards    the stream is split by flow index into 1, 2 or 4 shards, each served by its own
            process with its own copy of the models and its own budget threshold; reports
            aggregate throughput and the queue load of each shard and of the union
  retrain   wall time of one LightGBM retraining on the source training split (1 and 4 threads)

  python revision_stream.py   -> results/revision/stream_batch.csv, stream_shards.csv, stream_misc.csv
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "1")
import run_experiments as R  # noqa: E402
from nidslib import Detector, LogScaler, make_model  # noqa: E402
from revision_analysis import pvals, make_stream, online_quantile, OUT  # noqa: E402

C, W, N_REF, K = 0.01, 0.7, 20000, 5
N_STREAM = int(os.environ.get("N_STREAM", 400000))
WARM = 20000


def build(seed=0):
    import faiss
    df, feats = R.load("CSE-CIC-IDS2018")
    X, y = df[feats].to_numpy(np.float32), df["label"].to_numpy()
    pt = R.split(df, seed)
    Xtr, ytr, Xc, yc, Xte, yte = X[pt == 0], y[pt == 0], X[pt == 1], y[pt == 1], X[pt == 2], y[pt == 2]
    det = Detector("LGBM", seed).fit(Xtr, ytr)
    det.clf.set_params(n_jobs=1)
    ben = Xtr[ytr == 0]
    sc = LogScaler().fit(ben)
    ref = sc.transform(ben[np.random.default_rng(seed).choice(len(ben), N_REF, replace=False)]).astype(np.float32)
    idx = faiss.IndexHNSWFlat(ref.shape[1], 32)
    idx.add(ref)
    idx.hnsw.efSearch = 64
    mc = yc == 0
    Zc = sc.transform(Xc[mc]).astype(np.float32)
    dc, _ = idx.search(Zc, K)
    cal_d = np.sort(det.score(Xc[mc]))
    cal_n = np.sort(np.sqrt(np.maximum(dc, 0)).mean(1))
    # test stream at 1% attack prevalence, tiled to N_STREAM flows if needed
    s_idx = make_stream(yte, 0.01, np.random.default_rng(seed + 7))
    s_idx = np.resize(s_idx, N_STREAM)
    state = dict(det=det, sc=sc, idx=faiss.serialize_index(idx), cal_d=cal_d, cal_n=cal_n)
    return state, Xte[s_idx], yte[s_idx], (Xtr, ytr, ben, sc, Xte)


class Triage:
    def __init__(self, state):
        import faiss
        faiss.omp_set_num_threads(1)
        self.det, self.sc = state["det"], state["sc"]
        self.idx = faiss.deserialize_index(state["idx"])
        self.idx.hnsw.efSearch = 64
        self.cal_d, self.cal_n = state["cal_d"], state["cal_n"]
        self.t, self.gamma = None, None

    def priority(self, X):
        pd_ = pvals(self.cal_d, self.det.score(X))
        d, _ = self.idx.search(self.sc.transform(X).astype(np.float32), K)
        pn_ = pvals(self.cal_n, np.sqrt(np.maximum(d, 0)).mean(1))
        return -W * np.log(pd_) - (1 - W) * np.log(pn_)

    def warm(self, X):
        s = self.priority(X)
        self.t = float(np.quantile(s, 1 - C))
        self.gamma = 0.02 * (np.quantile(s, 0.999) - np.quantile(s, 0.5))

    def step(self, X):
        s = self.priority(X)
        f = online_quantile(s, C, self.t, self.gamma)
        # carry the threshold to the next batch
        self.t += self.gamma * (f.sum() - C * len(f))
        return f


def serve(state, X, batch):
    tr = Triage(state)
    tr.warm(X[:WARM])
    tr.step(X[WARM:WARM + batch])           # warm caches and numba
    lat, flags = [], np.zeros(len(X) - WARM, bool)
    t0 = time.perf_counter()
    for i in range(WARM, len(X), batch):
        a = time.perf_counter()
        flags[i - WARM:i - WARM + batch] = tr.step(X[i:i + batch])
        lat.append(time.perf_counter() - a)
    wall = time.perf_counter() - t0
    return dict(n=len(X) - WARM, wall=wall, lat=np.array(lat), flags=flags)


def _worker(args):
    path, lo, hi, batch = args
    state, X = pickle.load(open(path, "rb"))
    r = serve(state, X[lo:hi], batch)
    return dict(lo=lo, hi=hi, n=r["n"], wall=r["wall"], flags=r["flags"],
                p50=np.median(r["lat"]), p99=np.quantile(r["lat"], 0.99))


def main():
    state, X, y, (Xtr, ytr, ben, sc_b, Xte) = build()
    rows = []
    for b in [256, 1024, 4096, 16384]:
        r = serve(state, X, b)
        rows.append(dict(batch=b, flows=r["n"], flows_per_s=r["n"] / r["wall"],
                         us_per_flow=1e6 * r["wall"] / r["n"],
                         p50_ms=1e3 * np.median(r["lat"]), p99_ms=1e3 * np.quantile(r["lat"], 0.99),
                         load=r["flags"].mean() / C))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stream_batch.csv"), index=False)

    path = os.path.join(OUT, ".stream_state.pkl")
    pickle.dump((state, X), open(path, "wb"))
    rows = []
    ctx = mp.get_context("spawn")
    for P in [1, 2, 4]:
        edges = np.linspace(0, len(X), P + 1).astype(int)
        jobs = [(path, edges[i], edges[i + 1], 4096) for i in range(P)]
        with ctx.Pool(P) as pool:
            pool.map(_worker, [(path, 0, WARM + 8192, 4096)] * P)   # start-up and imports
            t0 = time.perf_counter()
            res = pool.map(_worker, jobs)
            wall = time.perf_counter() - t0
        n = sum(r["n"] for r in res)
        union = np.concatenate([r["flags"] for r in res])
        rows.append(dict(shards=P, flows=n, wall_s=wall, flows_per_s=n / wall,
                         max_shard_wall_s=max(r["wall"] for r in res),
                         p99_ms=1e3 * max(r["p99"] for r in res),
                         load_union=union.mean() / C,
                         load_shard_min=min(r["flags"].mean() for r in res) / C,
                         load_shard_max=max(r["flags"].mean() for r in res) / C))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stream_shards.csv"), index=False)
    os.remove(path)

    misc = []
    for th in [1, 4]:
        t0 = time.perf_counter()
        m = make_model("LGBM", 0)
        m.set_params(n_jobs=th)
        sc = LogScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        misc.append(dict(item="retrain_lgbm_s", threads=th, value=time.perf_counter() - t0, n_train=len(ytr)))
    tr = Triage(state)
    misc.append(dict(item="state_mb_hnsw", threads=np.nan, value=len(state["idx"]) / 2 ** 20))
    misc.append(dict(item="state_mb_calibration", threads=np.nan,
                     value=(state["cal_d"].nbytes + state["cal_n"].nbytes) / 2 ** 20))
    misc.append(dict(item="state_mb_lgbm", threads=np.nan,
                     value=len(pickle.dumps(state["det"])) / 2 ** 20))
    del tr
    # HNSW: build (refresh) time, neighbour recall and search cost over three reference samples
    import faiss
    from sklearn.neighbors import NearestNeighbors
    faiss.omp_set_num_threads(1)
    Q = sc_b.transform(Xte[:2000]).astype(np.float32)
    for n_ref in [20000, 50000, 100000]:
        for rs in range(3):
            Zr = sc_b.transform(ben[np.random.default_rng(rs).choice(len(ben), min(n_ref, len(ben)), replace=False)]).astype(np.float32)
            exact = NearestNeighbors(n_neighbors=K).fit(Zr).kneighbors(Q, return_distance=False)
            t0 = time.perf_counter()
            ix = faiss.IndexHNSWFlat(Zr.shape[1], 32)
            ix.add(Zr)
            bt = time.perf_counter() - t0
            for ef in [64, 128, 256]:
                ix.hnsw.efSearch = ef
                t0 = time.perf_counter(); _, I = ix.search(Q, K); dt = time.perf_counter() - t0
                rec = np.mean([len(set(a) & set(b)) / K for a, b in zip(I, exact)])
                misc.append(dict(item="hnsw", threads=1, n_ref=n_ref, rep=rs, ef=ef, build_s=bt,
                                 recall_at5=rec, us_per_flow=1e6 * dt / len(Q)))
        print("hnsw", n_ref, flush=True)
    pd.DataFrame(misc).to_csv(os.path.join(OUT, "stream_misc.csv"), index=False)
    print(misc, flush=True)


if __name__ == "__main__":
    main()

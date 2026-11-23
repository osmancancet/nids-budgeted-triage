"""Parallel execution of the triage layer: threads inside one process, independent shards, and
shards that share one review threshold (CSE-CIC-IDS2018, LightGBM, seed 0, the stream of
revision_stream.py).

  threads   one process; LightGBM and Faiss use T threads (T = 1, 2, 4); micro-batches of 4,096
  shards    P processes (P = 1, 2, 3, 4), each with its own models and its own threshold
  shared    four processes that share one threshold in shared memory: after every micro-batch of
            s flows (s = 256, 1,024, 4,096) a worker flags the batch with the current threshold and
            then applies its update under a lock; reports throughput, lock wait and queue load

  python revision_sync.py   -> results/revision/stream_sync.csv
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "1")
from revision_stream import build, Triage, WARM, C, N_STREAM  # noqa: E402
from revision_analysis import online_quantile, OUT  # noqa: E402

B_INDEP = 4096


def _threads(args):
    path, T = args
    import faiss
    state, X = pickle.load(open(path, "rb"))
    tr = Triage(state)
    faiss.omp_set_num_threads(T)
    tr.det.clf.set_params(n_jobs=T)
    tr.warm(X[:WARM])
    tr.step(X[WARM:WARM + B_INDEP])
    t0 = time.perf_counter()
    fl = []
    for i in range(WARM, len(X), B_INDEP):
        fl.append(tr.step(X[i:i + B_INDEP]))
    wall = time.perf_counter() - t0
    f = np.concatenate(fl)
    return dict(n=len(f), wall=wall, load=f.mean() / C)


SH = None


def _init(sh):
    global SH
    SH = sh


def _shard(args):
    path, lo, hi, s, use_shared = args
    shared = SH if use_shared else None
    state, X = pickle.load(open(path, "rb"))
    X = X[lo:hi]
    tr = Triage(state)
    tr.warm(X[:WARM])
    tr.priority(X[WARM:WARM + 1024])
    if shared is not None:
        t_sh, lock, ready = shared
        with lock:
            if ready.value == 0:            # the first worker publishes its warm-up threshold
                t_sh.value = tr.t
                ready.value = 1
    fl, wait = [], 0.0
    t0 = time.perf_counter()
    for i in range(WARM, len(X), s):
        x = X[i:i + s]
        if shared is None:
            fl.append(tr.step(x))
            continue
        sc = tr.priority(x)
        f = sc >= t_sh.value
        a = time.perf_counter()
        with lock:
            wait += time.perf_counter() - a
            t_sh.value += tr.gamma * (f.sum() - C * len(f))
        fl.append(f)
    wall = time.perf_counter() - t0
    return dict(n=sum(len(f) for f in fl), wall=wall, wait=wait, flags=np.concatenate(fl))


def run_shards(path, n, P, s, shared):
    ctx = mp.get_context("fork")
    edges = np.linspace(0, n, P + 1).astype(int)
    if shared:
        sh = (ctx.Value("d", 0.0, lock=False), ctx.Lock(), ctx.Value("i", 0, lock=False))
    jobs = [(path, edges[i], edges[i + 1], s, bool(shared)) for i in range(P)]
    with ctx.Pool(P, initializer=_init, initargs=(sh if shared else None,)) as pool:
        t0 = time.perf_counter()
        res = pool.map(_shard, jobs)
        wall = time.perf_counter() - t0
    u = np.concatenate([r["flags"] for r in res])
    tenths = [u[c].mean() / C for c in np.array_split(np.arange(len(u)), 10)]
    nf = sum(r["n"] for r in res)
    return dict(flows=nf, wall_s=wall, flows_per_s=nf / max(r["wall"] for r in res), flows_per_s_wall=nf / wall,
                max_worker_s=max(r["wall"] for r in res), lock_wait_share=max(r["wait"] / r["wall"] for r in res),
                load_union=u.mean() / C, load_shard_min=min(r["flags"].mean() for r in res) / C,
                load_shard_max=max(r["flags"].mean() for r in res) / C,
                load_tenth_min=min(tenths), load_tenth_max=max(tenths))


def main():
    state, X, y, _ = build()
    path = os.path.join(OUT, ".sync_state.pkl")
    pickle.dump((state, X), open(path, "wb"))
    rows = []
    reps = int(os.environ.get("REPS", 3))
    for rep in range(reps):
        for T in [1, 2, 4]:
            with mp.get_context("spawn").Pool(1) as pool:
                r = pool.map(_threads, [(path, T)])[0]
            rows.append(dict(rep=rep, mode="threads", workers=T, batch=B_INDEP, flows=r["n"], wall_s=r["wall"],
                             flows_per_s=r["n"] / r["wall"], load_union=r["load"]))
            print(rows[-1], flush=True)
        for P in [1, 2, 3, 4]:
            rows.append(dict(rep=rep, mode="shards", workers=P, batch=B_INDEP,
                             **run_shards(path, len(X), P, B_INDEP, False)))
            print({k: v for k, v in rows[-1].items()}, flush=True)
        for s in [256, 1024, 4096]:
            rows.append(dict(rep=rep, mode="shared", workers=4, batch=s, **run_shards(path, len(X), 4, s, True)))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "stream_sync.csv"), index=False)
    os.remove(path)


if __name__ == "__main__":
    main()

"""Distributed execution of the triage layer from a portable bundle (see export_bundle.py).

Every worker is a separate process with its own copy of the models, and it scores its slice of the
stream in micro-batches with the weighted Fisher priority (w = 0.7, 1% budget, HNSW kNN novelty).

  indep   P workers, each with its own warm-up threshold and online update; no communication
  coord   P workers and one coordinator process that holds the review threshold; they talk over
          TCP (multiprocessing.connection). After each micro-batch of s flows a worker flags the
          batch with the last threshold it has received, sends its excess (flags - c*s) to the
          coordinator and continues without waiting; the coordinator applies t <- t + gamma*excess
          and returns the new t to that worker after a one-way delay of d ms, which emulates the
          network between nodes. The worker uses a reply when it arrives (asynchronous updates).
          The step is either the per-flow step g of the single-process update ("full") or g / P
          ("perworker"), which offsets the P - 1 updates a worker misses between two replies.
  sync    as coord, but a worker waits for the reply to its update before flagging its next batch
          (--sync); every batch then pays one round trip including the delay.

Reported per run: throughput over the slowest worker (processing time after model loading and the
warm-up), throughput including start-up, queue load of the union, load per tenth of the stream, and
the mean staleness (number of updates the coordinator had applied since issuing the threshold a
worker used for the batch it reports).

  python dist_bench.py BUNDLE_DIR OUT.csv [--workers 1,2,4,8] [--delays 0,1,10,50] [--batches 256,4096]
"""
import os
import sys
import time
import platform
import argparse
import threading
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.connection import Listener, Client

os.environ.setdefault("OMP_NUM_THREADS", "1")
C, W, K, WARM = 0.01, 0.7, 5, 20000
AUTH = b"triage"


def online(s, c, t, g):
    f = np.empty(len(s), bool)
    for i in range(len(s)):
        f[i] = s[i] >= t
        t += g * ((1.0 if f[i] else 0.0) - c)
    return f, t


try:
    from numba import njit

    @njit(cache=True)
    def _online(s, c, t, g):
        f = np.empty(len(s), np.bool_)
        for i in range(len(s)):
            f[i] = s[i] >= t
            t += g * ((1.0 if f[i] else 0.0) - c)
        return f, t

    def online(s, c, t, g):  # noqa: F811
        return _online(np.ascontiguousarray(s, np.float64), c, t, g)
except ImportError:
    pass


def pvals(cal, s):
    n = len(cal)
    return (1.0 + n - np.searchsorted(cal, s, side="left")) / (n + 1.0)


class Triage:
    def __init__(self, d):
        import faiss
        import lightgbm as lgb
        faiss.omp_set_num_threads(1)
        self.b = lgb.Booster(model_file=os.path.join(d, "lgbm.txt"))
        z = np.load(os.path.join(d, "bundle.npz"))
        self.dm, self.ds, self.nm, self.ns = z["det_mean"], z["det_scale"], z["nov_mean"], z["nov_scale"]
        self.idx = faiss.deserialize_index(z["idx"])
        self.idx.hnsw.efSearch = 64
        self.cal_d, self.cal_n = z["cal_d"], z["cal_n"]

    @staticmethod
    def _t(X, m, s):
        L = np.sign(X) * np.log1p(np.abs(X.astype(np.float64)))
        return np.clip((L - m) / s, -50, 50)

    def priority(self, X):
        pdt = pvals(self.cal_d, self.b.predict(self._t(X, self.dm, self.ds), num_threads=1))
        d, _ = self.idx.search(self._t(X, self.nm, self.ns).astype(np.float32), K)
        pnv = pvals(self.cal_n, np.sqrt(np.maximum(d, 0)).mean(1))
        return -W * np.log(pdt) - (1 - W) * np.log(pnv)

    def warm(self, X):
        s = self.priority(X)
        return float(np.quantile(s, 1 - C)), 0.02 * (np.quantile(s, 0.999) - np.quantile(s, 0.5))


def coordinator(port, nworkers, t0, delay_ms, q):
    lst = Listener(("127.0.0.1", port), backlog=128, authkey=AUTH)
    state = {"t": t0, "n": 0, "stale": 0.0}
    lock = threading.Lock()

    def serve(conn):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            if msg is None:
                conn.close()
                return
            exc, g, used_n = msg
            with lock:
                state["stale"] += state["n"] - used_n
                state["t"] += g * exc
                state["n"] += 1
                t, n = state["t"], state["n"]
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            try:
                conn.send((t, n))
            except OSError:
                return
    ths = []
    for _ in range(nworkers):
        th = threading.Thread(target=serve, args=(lst.accept(),), daemon=True)
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    lst.close()
    q.put((state["n"], state["stale"] / max(state["n"], 1)))


def worker(args):
    d, lo, hi, s, port, t_shared, g_shared, sync = args
    a0 = time.perf_counter()
    z = np.load(os.path.join(d, "bundle.npz"))
    X = z["X"][lo:hi]
    tr = Triage(d)
    t, g = tr.warm(X[:WARM])
    if port is not None:
        t, g = t_shared, g_shared
        conn = Client(("127.0.0.1", port), authkey=AUTH)
    tr.priority(X[WARM:WARM + 1024])
    t1 = time.perf_counter()
    fl, stale, last_n, sent_n = [], [], 0, []
    for i in range(WARM, len(X), s):
        sc = tr.priority(X[i:i + s])
        if port is None:
            f, t = online(sc, C, t, g)
        else:
            while conn.poll():
                t, last_n = conn.recv()
            f = sc >= t
            conn.send((float(f.sum() - C * len(f)), g, last_n))
            if sync:  # wait for the updated threshold before the next batch
                t, last_n = conn.recv()
        fl.append(f)
    if port is not None:
        conn.send(None)
        conn.close()
    t2 = time.perf_counter()
    return dict(flags=np.concatenate(fl), proc=t2 - t1, total=t2 - a0)


def run(d, n, P, s, mode, delay, port, step="full"):
    sync = mode == "sync"
    if sync:
        mode = "coord"
    edges = np.linspace(0, n, P + 1).astype(int)
    ctx = mp.get_context("spawn")
    tsh = gsh = None
    if mode == "coord":
        z = np.load(os.path.join(d, "bundle.npz"))
        tr = Triage(d)
        tsh, gsh = tr.warm(z["X"][:WARM])
        if step == "perworker":
            gsh = gsh / P
        q = ctx.Queue()
        cp = ctx.Process(target=coordinator, args=(port, P, tsh, delay, q))
        cp.start()
        time.sleep(0.5)
    jobs = [(d, edges[i], edges[i + 1], s, port if mode == "coord" else None, tsh, gsh, sync) for i in range(P)]
    a = time.perf_counter()
    with ctx.Pool(P) as pool:
        res = pool.map(worker, jobs)
    wall = time.perf_counter() - a
    napplied = st = None
    if mode == "coord":
        napplied, st = q.get()
        cp.join()
    u = np.concatenate([r["flags"] for r in res])
    tenths = [u[c].mean() / C for c in np.array_split(np.arange(len(u)), 10)]
    return dict(flows=len(u), flows_per_s=len(u) / max(r["proc"] for r in res),
                flows_per_s_total=len(u) / wall, load=u.mean() / C,
                load_tenth_min=min(tenths), load_tenth_max=max(tenths),
                load_worker_min=min(r["flags"].mean() for r in res) / C,
                load_worker_max=max(r["flags"].mean() for r in res) / C,
                updates=napplied, staleness=st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("out")
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--delays", default="0,1,10,50")
    ap.add_argument("--batches", default="256,1024,4096")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sync", action="store_true", help="only synchronous coordinated runs")
    a = ap.parse_args()
    d = a.bundle
    n = len(np.load(os.path.join(d, "bundle.npz"))["y"])
    if a.check:
        tr = Triage(d)
        X = np.load(os.path.join(d, "bundle.npz"))["X"][:50000]
        ref = np.load(os.path.join(d, "check_priority.npy"))
        print("max abs priority difference", float(np.max(np.abs(tr.priority(X) - ref))))
    host = dict(machine=platform.node(), cpu=platform.processor() or platform.machine(),
                system=platform.system(), cores=os.cpu_count())
    rows, port = [], 47000
    ws = [int(x) for x in a.workers.split(",")]
    if a.sync:
        for rep in range(a.reps):
            for s in [int(x) for x in a.batches.split(",")]:
                for dl in [int(x) for x in a.delays.split(",")]:
                    for stp in ["full", "perworker"]:
                        port += 1
                        rows.append(dict(host, rep=rep, mode="sync", workers=max(ws), batch=s, delay_ms=dl,
                                         step=stp, **run(d, n, max(ws), s, "sync", dl, port, stp)))
                        print(rows[-1], flush=True)
                        pd.DataFrame(rows).to_csv(a.out, index=False)
        return
    for rep in range(a.reps):
        for P in ws:
            rows.append(dict(host, rep=rep, mode="indep", workers=P, batch=4096, delay_ms=0,
                             **run(d, n, P, 4096, "indep", 0, None)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
        for s in [int(x) for x in a.batches.split(",")]:
            for dl in [int(x) for x in a.delays.split(",")]:
                for stp in ["full", "perworker"]:
                    port += 1
                    rows.append(dict(host, rep=rep, mode="coord", workers=max(ws), batch=s, delay_ms=dl, step=stp,
                                     **run(d, n, max(ws), s, "coord", dl, port, stp)))
                    print(rows[-1], flush=True)
                    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

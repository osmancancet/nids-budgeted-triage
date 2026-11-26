"""The triage layer on workers spread over several machines that share one review threshold.

One machine runs the coordinator (and optionally some workers); every other machine runs workers that
reach the coordinator over the real network. The stream of the bundle is cut by flow index into
TOTAL slices, one per worker; worker k scores slice k in micro-batches with the weighted Fisher
priority, as in dist_bench.py. No delay is injected: the delay is that of the network.

  coord   a worker flags a batch with the last threshold it has received and continues
  sync    a worker waits for the updated threshold before flagging its next batch

The step is the per-flow step divided by TOTAL. Each configuration is repeated --reps times; both sides
must be started with the same --modes, --batches, --reps and --total. At the end of a run every worker
sends its flags and timings to the coordinator, which writes one row per run to OUT.csv.

  machine A:  python dist_nodes.py coord BUNDLE OUT.csv --total 8 --local 4
  machine B:  python dist_nodes.py workers BUNDLE --host <IP of A> --total 8 --first 4 --local 4
"""
import os
import sys
import time
import socket
import platform
import argparse
import threading
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing.connection import Listener, Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dist_bench import Triage, C, WARM, AUTH  # noqa: E402

PORT0 = 47100


def configs(a):
    out = []
    for rep in range(a.reps):
        for mode in a.modes.split(","):
            for s in [int(x) for x in a.batches.split(",")]:
                out.append((rep, mode, s))
    return out


def connect(host, port, tries=600):
    for _ in range(tries):
        try:
            return Client((host, port), authkey=AUTH)
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"cannot reach coordinator at {host}:{port}")


def worker(args):
    d, k, total, s, sync, host, port = args
    import numpy as np
    z = np.load(os.path.join(d, "bundle.npz"))
    edges = np.linspace(0, len(z["y"]), total + 1).astype(int)
    X = z["X"][edges[k]:edges[k + 1]]
    tr = Triage(d)
    tr.priority(X[WARM:WARM + 1024])
    conn = connect(host, port)
    conn.send(("hello", k, platform.node()))
    t, g = conn.recv()
    last_n, fl, rtt = 0, [], []
    t1 = time.perf_counter()
    for i in range(WARM, len(X), s):
        sc = tr.priority(X[i:i + s])
        while conn.poll():
            t, last_n = conn.recv()
        f = sc >= t
        a = time.perf_counter()
        conn.send((float(f.sum() - C * len(f)), last_n))
        if sync:
            t, last_n = conn.recv()
            rtt.append(time.perf_counter() - a)
        fl.append(f)
    proc = time.perf_counter() - t1
    conn.send(("done", k, np.concatenate(fl), proc, float(np.median(rtt)) if rtt else np.nan))
    conn.close()


def run_local(d, ks, total, s, sync, host, port):
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker, args=((d, k, total, s, sync, host, port),)) for k in ks]
    for p in ps:
        p.start()
    return ps


def coordinator(a):
    d = a.bundle
    z = np.load(os.path.join(d, "bundle.npz"))
    tr = Triage(d)
    t0, g0 = tr.warm(z["X"][:WARM])
    g = g0 / a.total
    rows = []
    for j, (rep, mode, s) in enumerate(configs(a)):
        port = PORT0 + j
        lst = Listener((a.bind, port), backlog=128, authkey=AUTH)
        ps = run_local(d, range(a.local), a.total, s, mode == "sync", "127.0.0.1", port) if a.local else []
        state = {"t": t0, "n": 0, "stale": 0.0}
        lock = threading.Lock()
        res, hosts = {}, {}

        def serve(conn):
            _, k, node = conn.recv()
            hosts[k] = node
            conn.send((t0, g))
            while True:
                msg = conn.recv()
                if msg[0] == "done":
                    res[k] = msg
                    return
                exc, used_n = msg
                with lock:
                    state["stale"] += state["n"] - used_n
                    state["t"] += g * exc
                    state["n"] += 1
                    t, n = state["t"], state["n"]
                conn.send((t, n))

        ths = []
        for _ in range(a.total):
            th = threading.Thread(target=serve, args=(lst.accept(),), daemon=True)
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        lst.close()
        for p in ps:
            p.join()
        u = np.concatenate([res[k][2] for k in range(a.total)])
        tenths = [u[c].mean() / C for c in np.array_split(np.arange(len(u)), 10)]
        nodes = sorted(set(hosts.values()))
        rtts = [res[k][4] for k in range(a.total) if not np.isnan(res[k][4])]
        rows.append(dict(rep=rep, mode=mode, batch=s, workers=a.total, nodes=len(nodes),
                         workers_per_node=",".join(str(sum(v == n for v in hosts.values())) for n in nodes),
                         flows=len(u), flows_per_s=len(u) / max(res[k][3] for k in range(a.total)),
                         load=u.mean() / C, load_tenth_min=min(tenths), load_tenth_max=max(tenths),
                         load_worker_min=min(res[k][2].mean() for k in range(a.total)) / C,
                         load_worker_max=max(res[k][2].mean() for k in range(a.total)) / C,
                         updates=state["n"], staleness=state["stale"] / max(state["n"], 1),
                         rtt_ms_median=1000 * float(np.median(rtts)) if rtts else np.nan))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


def workers(a):
    for j, (rep, mode, s) in enumerate(configs(a)):
        ps = run_local(a.bundle, range(a.first, a.first + a.local), a.total, s, mode == "sync", a.host, PORT0 + j)
        for p in ps:
            p.join()
        print(f"done rep={rep} mode={mode} batch={s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["coord", "workers"])
    ap.add_argument("bundle")
    ap.add_argument("out", nargs="?", default="dist_nodes.csv")
    ap.add_argument("--total", type=int, default=8)
    ap.add_argument("--local", type=int, default=4)
    ap.add_argument("--first", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--modes", default="sync,coord")
    ap.add_argument("--batches", default="64,256")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.role == "coord":
        print("coordinator on", socket.gethostname(), flush=True)
        coordinator(a)
    else:
        workers(a)


if __name__ == "__main__":
    main()

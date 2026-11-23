"""Where the time goes in one streaming batch (revision_stream.py setup, batches of 4,096 flows, one core).

Each stage of Triage.step is timed separately over the whole stream: detector (scaling and LightGBM),
detector p-value, novelty scaling, HNSW search, novelty p-value, weighted Fisher and threshold update.
Other jobs on the machine are paused (SIGSTOP) while the timing runs.

  python revision_stream_profile.py   -> results/revision/stream_profile.csv
"""
import os
import signal
import time
import subprocess
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
from revision_stream import build, Triage, WARM, C, K, W  # noqa: E402
from revision_analysis import pvals, online_quantile, OUT  # noqa: E402

B = 4096


def others():
    out = subprocess.run(["pgrep", "-f", "revision_"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def main():
    state, X, _, _ = build()
    tr = Triage(state)
    tr.warm(X[:WARM])
    tr.step(X[WARM:WARM + B])
    T = dict(detector=0.0, p_detector=0.0, scale_novelty=0.0, hnsw=0.0, p_novelty=0.0, fisher=0.0, update=0.0)
    pids = others()
    for p in pids:
        os.kill(p, signal.SIGSTOP)
    try:
        time.sleep(2)
        t_all = time.perf_counter()
        for i in range(WARM, len(X), B):
            x = X[i:i + B]
            a = time.perf_counter(); sd = tr.det.score(x); b = time.perf_counter(); T["detector"] += b - a
            pd_ = pvals(tr.cal_d, sd); a = time.perf_counter(); T["p_detector"] += a - b
            z = tr.sc.transform(x).astype(np.float32); b = time.perf_counter(); T["scale_novelty"] += b - a
            d, _ = tr.idx.search(z, K); a = time.perf_counter(); T["hnsw"] += a - b
            pn_ = pvals(tr.cal_n, np.sqrt(np.maximum(d, 0)).mean(1)); b = time.perf_counter(); T["p_novelty"] += b - a
            s = -W * np.log(pd_) - (1 - W) * np.log(pn_); a = time.perf_counter(); T["fisher"] += a - b
            f = online_quantile(s, C, tr.t, tr.gamma); tr.t += tr.gamma * (f.sum() - C * len(f))
            b = time.perf_counter(); T["update"] += b - a
        wall = time.perf_counter() - t_all
    finally:
        for p in pids:
            os.kill(p, signal.SIGCONT)
    n = len(X) - WARM
    rows = [dict(stage=k, us_per_flow=1e6 * v / n, share=v / wall) for k, v in T.items()]
    rows.append(dict(stage="total", us_per_flow=1e6 * wall / n, share=1.0))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stream_profile.csv"), index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()

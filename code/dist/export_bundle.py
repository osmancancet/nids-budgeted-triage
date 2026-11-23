"""Export the stream of revision_stream.py as a portable bundle for dist_bench.py.

The bundle holds the 400,000-flow CSE-CIC-IDS2018 test stream (seed 0), the LightGBM model as a
text booster, the parameters of both log scalers, the serialized HNSW index over 20,000 benign
reference flows and the two sorted calibration vectors, so that the benchmark runs on any machine
with numpy, lightgbm and faiss-cpu, without the raw data.

  python dist/export_bundle.py OUTDIR
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from revision_stream import build  # noqa: E402


def main(out):
    os.makedirs(out, exist_ok=True)
    state, X, y, _ = build()
    det, sc = state["det"], state["sc"]
    np.savez_compressed(os.path.join(out, "bundle.npz"), X=X.astype(np.float32), y=y.astype(np.int8),
                        det_mean=det.scaler.sc.mean_, det_scale=det.scaler.sc.scale_,
                        nov_mean=sc.sc.mean_, nov_scale=sc.sc.scale_,
                        idx=np.frombuffer(bytes(state["idx"]), np.uint8),
                        cal_d=state["cal_d"], cal_n=state["cal_n"])
    det.clf.booster_.save_model(os.path.join(out, "lgbm.txt"))
    # reference priorities of the first 50,000 flows, to check the port on another machine
    from revision_stream import Triage
    tr = Triage(state)
    np.save(os.path.join(out, "check_priority.npy"), tr.priority(X[:50000]))
    print("exported", X.shape)


if __name__ == "__main__":
    main(sys.argv[1])

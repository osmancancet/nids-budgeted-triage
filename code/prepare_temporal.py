"""CSE-CIC-IDS2018 with capture day and timestamp kept, for the temporal analyses.

Same reading, 5% benign Bernoulli subsampling (same RNG stream as prepare.py),
cleaning and deduplication as prepare.py, but without the per-family caps and
with two extra columns: `day` (capture date) and `ts` (flow start time, seconds
since epoch, parsed from the CICFlowMeter Timestamp field). Duplicates are
removed over features + family, keeping the earliest flow.

Output: data/processed/CSE-CIC-IDS2018-temporal.parquet
"""
import os
import numpy as np
import pandas as pd
from prepare import RAW, OUT, SEED, CIC_FEATURES, GROUPS


def main():
    d = os.path.join(RAW, "CSE-CIC-IDS2018")
    rng = np.random.default_rng(SEED)
    parts = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        for ch in pd.read_csv(os.path.join(d, f), chunksize=250_000, dtype=str,
                              usecols=CIC_FEATURES + ["Protocol", "Label", "Timestamp"]):
            ch = ch[ch["Label"] != "Label"]
            keep = (ch["Label"] != "Benign").to_numpy() | (rng.random(len(ch)) < 0.05)
            ch = ch[keep]
            X = ch[CIC_FEATURES + ["Protocol"]].apply(pd.to_numeric, errors="coerce").astype(np.float64)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            X["family"] = ch["Label"].to_numpy()
            ts = pd.to_datetime(ch["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            X["ts"] = ((ts - pd.Timestamp("1970-01-01")).dt.total_seconds()).fillna(-1).astype("int64").to_numpy()
            X["file"] = f.split("_")[0]
            parts.append(X)
        print("read", f, flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df[df.ts > 0]
    df = df.sort_values("ts", kind="stable")
    feats = CIC_FEATURES + ["Protocol"]
    df = df.drop_duplicates(subset=feats + ["family"], keep="first")
    df["day"] = pd.to_datetime(df.ts, unit="s").dt.strftime("%Y-%m-%d")
    df["label"] = (df["family"] != "Benign").astype(np.int8)
    df["group"] = df["family"].map(GROUPS).fillna(df["family"])
    df = df.reset_index(drop=True)
    df.to_parquet(os.path.join(OUT, "CSE-CIC-IDS2018-temporal.parquet"), index=False)
    print(df.groupby(["day", "family"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()

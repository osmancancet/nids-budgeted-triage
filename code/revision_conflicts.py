"""Feature-identical flows with conflicting labels in the processed datasets (after exact
de-duplication over features and label).

  python revision_conflicts.py   -> results/revision/label_conflicts.csv
"""
import os
import pandas as pd

import run_experiments as R
from revision_analysis import OUT


def main():
    rows = []
    for name in R.ALL:
        df, feats = R.load(name)
        g = df.groupby(feats, sort=False)["label"].transform("nunique")
        conf = g > 1
        rows.append(dict(dataset=name, n=len(df), n_conflict_flows=int(conf.sum()),
                         share=conf.mean(), n_conflict_attack=int((conf & (df.label == 1)).sum())))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "label_conflicts.csv"), index=False)


if __name__ == "__main__":
    main()

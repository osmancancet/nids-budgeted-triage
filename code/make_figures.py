"""Aggregate result CSVs into LaTeX tables, figures and a numbers file."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# reference categorical palette (fixed order) and text inks
C = {"detector": "#2a78d6", "fisher": "#eb6834", "combined": "#1baf7a", "novelty": "#eda100"}
LAB = {"detector": "Detector score", "fisher": "Conformal (Fisher)",
       "combined": "Conformal (Bonferroni)", "novelty": "Novelty only"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "pdf.fonttype": 42})
SHORT = {"CIC-IDS2017": "CIC17", "CSE-CIC-IDS2018": "CIC18", "ACI-IoT-2023": "ACI23",
         "UNSW-NB15": "UNSW", "ToN-IoT": "ToN"}
NUM = {}


def ms(x, d=3):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f}" + (f"$\\pm${x.std(ddof=1):.{d}f}" if len(x) > 1 else "")


def data_table():
    parts = [pd.read_csv(os.path.join(RES, f)) for f in sorted(os.listdir(RES))
             if f.startswith("data_summary_")]
    d = pd.concat(parts)
    rows = []
    for ds in ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023", "UNSW-NB15", "ToN-IoT"]:
        x = d[d.dataset == ds]
        raw = int(x[x.family == "__rows_raw__"].n_final.iloc[0])
        dd = int(x[x.family == "__rows_dedup__"].n_final.iloc[0])
        fam = x[~x.family.str.startswith("__")]
        ben = fam[fam.family.str.lower().isin(["benign", "normal"])].n_final.sum()
        att = fam.n_final.sum() - ben
        nf = (~fam.family.str.lower().isin(["benign", "normal"])).sum()
        rows.append((ds, raw, dd, 100 * (1 - dd / raw), int(ben), int(att), int(nf)))
    with open(os.path.join(TAB, "data.tex"), "w") as f:
        for r in rows:
            f.write(f"{r[0]} & {r[1]:,} & {r[2]:,} & {r[3]:.1f} & {r[4]:,} & {r[5]:,} & {r[6]} \\\\\n")
    NUM["data"] = rows


def xd_tables():
    p = os.path.join(RES, "xd_detection.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    m = d[d.group.isna()] if "group" in d else d
    # main table: per model, in-domain (CIC datasets only for comparability) vs cross-dataset
    cic = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023"]
    lines = []
    for model in ["LGBM", "RF", "MLP", "LR"]:
        for setting in ["in-domain", "cross-dataset"]:
            x = m[(m.model == model) & (m.setting == setting) & m.source.isin(cic)]
            # average over pairs first, then report mean+-sd over seeds
            g = x.groupby("seed").mean(numeric_only=True)
            lines.append(f"{model} & {setting} & {ms(g.auroc)} & {ms(g.ece_raw)} & {ms(g.ece_iso)} & "
                         f"{ms(g.fpr_inflation, 1)} & {ms(g.tpr_at_alpha)} & {ms(g['prec_at_pi0.01'])} \\\\")
            NUM[f"xd_{model}_{setting}"] = g.mean().to_dict()
        lines.append("\\midrule")
    with open(os.path.join(TAB, "xd_main.tex"), "w") as f:
        f.write("\n".join(lines[:-1]) + "\n")
    # in-domain for the non-CIC datasets
    x = m[(m.setting == "in-domain") & ~m.source.isin(cic)]
    NUM["id_other"] = x.groupby(["source", "model"]).mean(numeric_only=True)[
        ["auroc", "ece_iso", "fpr_inflation", "tpr_at_alpha"]].round(4).reset_index().to_dict("records")
    # pairwise LGBM heatmaps
    L = m[(m.model == "LGBM") & m.source.isin(cic)].groupby(["source", "target"]).mean(numeric_only=True)
    if len(L) < 9:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (col, title, fmt, vmax) in zip(axes, [("auroc", "AUROC", "{:.2f}", 1.0),
                                                  ("ece_iso", "ECE (isotonic)", "{:.2f}", None),
                                                  ("fpr_inflation", r"FPR / $\alpha$", "{:.0f}", None)]):
        M = np.array([[L.loc[(s, t), col] for t in cic] for s in cic])
        mat = np.log10(np.maximum(M, 1e-3)) if col == "fpr_inflation" else M
        im = ax.imshow(mat, cmap="Blues" if col != "auroc" else "Blues_r", aspect="auto")
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                dark = (mat[i, j] - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat) + 1e-12)
                dark = dark if col != "auroc" else 1 - dark
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if dark > 0.6 else INK)
        ax.set_xticks(range(3), [SHORT[c] for c in cic])
        ax.set_yticks(range(3), [SHORT[c] for c in cic])
        ax.set_xlabel("Target")
        ax.grid(False)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("Source")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "xd_heatmaps.pdf"))
    plt.close(fig)
    NUM["xd_lgbm_pairs"] = L[["auroc", "ap", "ece_raw", "ece_iso", "fpr_inflation", "tpr_at_alpha",
                              "prec_at_pi0.01"]].round(4).reset_index().to_dict("records")
    # per-group recall across datasets (LGBM)
    gr = d[d.group.notna() & (d.model == "LGBM")].groupby(["source", "target", "group"]).agg(
        rec=("group_recall", "mean"), n=("group_n", "mean")).reset_index()
    gr.to_csv(os.path.join(RES, "xd_group_recall_lgbm.csv"), index=False)
    NUM["xd_group_recall_lgbm"] = gr.round(3).to_dict("records")
    NUM["score_us_per_flow"] = m.groupby("model").score_us_per_flow.median().round(2).to_dict()


def triage_fig():
    parts = []
    for f, reg in [("xd_triage.csv", None), ("os_triage.csv", "Unseen family")]:
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        if reg is None:
            t["regime"] = np.where(t.setting == "in-domain", "In-domain", "Cross-dataset")
        else:
            t["regime"] = reg
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    regs = [r for r in ["In-domain", "Cross-dataset", "Unseen family"] if r in set(t.regime)]
    fig, axes = plt.subplots(1, len(regs), figsize=(7.2, 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    summ = []
    for ax, r in zip(axes, regs):
        x = t[t.regime == r]
        if r != "Unseen family":
            x = x[x.model == "LGBM"]
        for pr in ["detector", "fisher", "combined", "novelty"]:
            g = x[x.priority == pr].groupby("budget").recall_topc
            mu, sd = g.mean(), g.sem()
            ax.plot(100 * mu.index, mu.values, color=C[pr], lw=2, marker="o", ms=4, label=LAB[pr])
            ax.fill_between(100 * mu.index, (mu - sd).clip(0), (mu + sd).clip(upper=1), color=C[pr],
                            alpha=0.12, lw=0)
            for b, v in mu.items():
                summ.append(dict(regime=r, priority=pr, budget=b, recall=v))
        ax.set_title(r, fontsize=9, color=INK)
        ax.set_xlabel("Review budget (% of flows)")
        ax.set_xticks([1, 2, 5])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Attack recall inside the queue")
    axes[1 if len(axes) > 1 else 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "triage_recall.pdf"))
    plt.close(fig)
    S = pd.DataFrame(summ)
    NUM["triage_recall"] = S.round(4).to_dict("records")
    # budget adherence: load with source threshold vs label-free warm-up
    lo = t[(t.priority == "fisher")].copy()
    lo = lo if "model" not in lo else lo[(lo.model == "LGBM") | lo.model.isna()]
    rows = []
    for r in regs:
        y = lo[lo.regime == r]
        for c in sorted(y.budget.unique()):
            z = y[y.budget == c]
            rows.append(dict(regime=r, budget=c,
                             src_med=z.load_src.median(), src_q10=z.load_src.quantile(.1),
                             src_q90=z.load_src.quantile(.9),
                             tgt_med=z.load_tgt.median(), tgt_q10=z.load_tgt.quantile(.1),
                             tgt_q90=z.load_tgt.quantile(.9),
                             rec_src=z.recall_src.mean(), rec_tgt=z.recall_tgt.mean()))
    B = pd.DataFrame(rows)
    NUM["budget_adherence_fisher"] = B.round(3).to_dict("records")
    with open(os.path.join(TAB, "budget.tex"), "w") as f:
        for _, r in B.iterrows():
            f.write(f"{r.regime} & {100*r.budget:.0f}\\% & {r.src_med:.2f} [{r.src_q10:.2f}, {r.src_q90:.2f}] & "
                    f"{r.rec_src:.3f} & {r.tgt_med:.2f} [{r.tgt_q10:.2f}, {r.tgt_q90:.2f}] & {r.rec_tgt:.3f} \\\\\n")
    # load distribution figure
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    pos, ticks = 0, []
    for r in regs:
        y = lo[(lo.regime == r) & (lo.budget == 0.01)]
        for k, (col, colr) in enumerate([("load_src", C["detector"]), ("load_tgt", C["fisher"])]):
            v = np.log2(y[col].clip(lower=1 / 64))
            bp = ax.boxplot(v, positions=[pos + 0.4 * k], widths=0.32, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK))
            bp["boxes"][0].set(facecolor=colr, alpha=0.6, edgecolor=INK2)
        ticks.append(pos + 0.2)
        pos += 1.2
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xticks(ticks, regs)
    ax.set_yticks([-4, -2, 0, 2, 4, 6], ["1/16", "1/4", "1", "4", "16", "64"])
    ax.set_ylim(-6.3, 7.2)
    ax.set_ylabel("Realised load / budget (1% budget)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C["detector"], alpha=.6, label="Source threshold"),
                       Patch(facecolor=C["fisher"], alpha=.6, label="Label-free warm-up")],
              fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "budget_load.pdf"))
    plt.close(fig)
    # triage table (all regimes, all priorities, budgets)
    with open(os.path.join(TAB, "triage.tex"), "w") as f:
        for r in regs:
            x = t[t.regime == r]
            if r != "Unseen family":
                x = x[x.model == "LGBM"]
            for pr in ["detector", "novelty", "combined", "fisher"]:
                vals = []
                for c in [0.01, 0.02, 0.05]:
                    g = x[(x.priority == pr) & (x.budget == c)].recall_topc
                    vals.append(f"{g.mean():.3f}")
                f.write(f"{r if pr == 'detector' else ''} & {LAB[pr]} & " + " & ".join(vals) + " \\\\\n")
            f.write("\\midrule\n")
    # paired comparison fisher vs detector (per configuration)
    from scipy.stats import wilcoxon
    for r in regs:
        x = t[t.regime == r]
        if r != "Unseen family":
            x = x[x.model == "LGBM"]
        keys = [k for k in ["seed", "source", "target", "model", "dataset", "held_out", "budget"] if k in x]
        a = x[x.priority == "fisher"].set_index(keys).recall_topc
        b = x[x.priority == "detector"].set_index(keys).recall_topc
        a, b = a.align(b, join="inner")
        diff = (a - b).dropna()
        try:
            p = wilcoxon(diff[diff != 0]).pvalue
        except ValueError:
            p = np.nan
        NUM[f"fisher_vs_det_{r}"] = dict(n=len(diff), mean_diff=diff.mean(), median_diff=diff.median(),
                                         win=(diff > 0).mean(), loss=(diff < 0).mean(), p=p)


def os_tables():
    p = os.path.join(RES, "os_detection.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    g = d.groupby(["dataset", "held_out"]).mean(numeric_only=True).reset_index()
    g.to_csv(os.path.join(RES, "os_per_family.csv"), index=False)
    NUM["os_per_family"] = g.round(3).to_dict("records")
    rows = []
    with open(os.path.join(TAB, "os_summary.tex"), "w") as f:
        for ds in ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023", "UNSW-NB15", "ToN-IoT"]:
            x = g[g.dataset == ds]
            if x.empty:
                continue
            r = dict(ds=ds, nf=len(x), rec=x.recall_unseen_at_alpha.median(),
                     rec_seen=x.recall_seen_at_alpha.median(),
                     low=(x.recall_unseen_at_alpha < 0.5).sum(),
                     cm=x.confident_miss_iso.median(), cm_max=x.confident_miss_iso.max(),
                     auc=x.auroc_unseen_vs_benign.median(), aucn=x.auroc_nov_unseen_vs_benign.median())
            rows.append(r)
            f.write(f"{ds} & {r['nf']} & {r['rec_seen']:.3f} & {r['rec']:.3f} & {r['low']} & {r['cm']:.3f} "
                    f"({r['cm_max']:.3f}) & {r['auc']:.3f} & {r['aucn']:.3f} \\\\\n")
    NUM["os_summary"] = rows
    # where does conformal fusion help? split by detector weakness (post-hoc, oracle grouping)
    t = pd.read_csv(os.path.join(RES, "os_triage.csv"))
    w = t.pivot_table(index=["dataset", "held_out", "seed", "budget"], columns="priority",
                      values="recall_topc").reset_index()
    w = w.merge(g[["dataset", "held_out", "recall_unseen_at_alpha", "auroc_nov_unseen_vs_benign"]],
                on=["dataset", "held_out"])
    w["gain"] = w.fisher - w.detector
    w["weak"] = w.recall_unseen_at_alpha < 0.5
    agg = w.groupby(["weak", "budget"]).agg(n=("gain", "size"), gain=("gain", "mean"),
                                            det=("detector", "mean"), fis=("fisher", "mean"),
                                            bon=("combined", "mean")).reset_index()
    NUM["os_gain_by_weakness"] = agg.round(4).to_dict("records")
    with open(os.path.join(TAB, "os_gain.tex"), "w") as f:
        for _, r in agg.iterrows():
            lab = "$<0.5$" if r.weak else "$\\ge0.5$"
            f.write(f"{lab} & {int(r.n)//3} & {100*r.budget:.0f}\\% & {r.det:.3f} & {r.bon:.3f} & "
                    f"{r.fis:.3f} & {r.gain:+.3f} \\\\\n")
    NUM["os_overall"] = dict(n_families=len(g), n_below_half=int((g.recall_unseen_at_alpha < .5).sum()),
                             n_confmiss_over_half=int((g.confident_miss_iso > .5).sum()),
                             median_recall=g.recall_unseen_at_alpha.median(),
                             median_recall_seen=g.recall_seen_at_alpha.median())
    # scatter: recall at alpha vs confident misses
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    mk = {"CIC-IDS2017": "o", "CSE-CIC-IDS2018": "s", "ACI-IoT-2023": "^", "UNSW-NB15": "D",
          "ToN-IoT": "v"}
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (ds, m_), c in zip(mk.items(), cols):
        x = g[g.dataset == ds]
        ax.scatter(x.recall_unseen_at_alpha, x.confident_miss_iso, marker=m_, s=26, color=c,
                   edgecolor="white", linewidth=0.8, label=SHORT[ds], zorder=3)
    ax.set_xlabel(r"Recall of held-out family at $\alpha=1\%$")
    ax.set_ylabel("Confident misses ($\\hat p<0.05$)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.2,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "os_scatter.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    data_table()
    xd_tables()
    os_tables()
    triage_fig()
    with open(os.path.join(RES, "numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=float)
    print("ok")

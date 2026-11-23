"""Figures for the revised manuscript (PNG, 300 dpi), from results/ and results/revision/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, REV = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "revision")
FIG = os.path.join(ROOT, "figures", "revision")
os.makedirs(FIG, exist_ok=True)
C = {"detector": "#2a78d6", "fisher": "#eb6834", "wfisher0.7": "#1baf7a", "novelty": "#eda100",
     "source": "#52514e", "warmup": "#2a78d6", "online": "#eb6834", "rolling": "#1baf7a"}
LAB = {"detector": "Detector", "fisher": "Fisher", "wfisher0.7": "Weighted Fisher (w = 0.7)",
       "novelty": "Novelty", "source": "Source threshold", "warmup": "Warm-up quantile",
       "online": "Online update", "rolling": "Rolling quantile"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
SHORT = {"CIC-IDS2017": "CIC17", "CSE-CIC-IDS2018": "CIC18", "ACI-IoT-2023": "ACI23",
         "UNSW-NB15": "UNSW", "ToN-IoT": "ToN"}
CIC = ["CIC-IDS2017", "CSE-CIC-IDS2018", "ACI-IoT-2023"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, metadata={"Software": None})
    plt.close(fig)


def _r(path):
    p = os.path.join(REV, path)
    return pd.read_csv(p) if os.path.exists(p) else None


def fig1_heat():
    d = pd.read_csv(os.path.join(RES, "xd_detection.csv"))
    m = d[d.group.isna() & (d.model == "LGBM")]
    L = m.groupby(["source", "target"]).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (col, title, fmt) in zip(axes, [("auroc", "AUROC", "{:.2f}"), ("ece_iso", "ECE (isotonic)", "{:.2f}"),
                                            ("fpr_inflation", r"FPR / $\alpha$ (log colour)", "{:.1f}")]):
        M = np.array([[L.loc[(s, t), col] for t in CIC] for s in CIC])
        mat = np.log10(np.maximum(M, 1e-3)) if col == "fpr_inflation" else M
        ax.imshow(mat, cmap="Blues_r" if col == "auroc" else "Blues", aspect="auto")
        lo, hi = np.nanmin(mat), np.nanmax(mat)
        for i in range(3):
            for j in range(3):
                dk = (mat[i, j] - lo) / (hi - lo + 1e-12)
                dk = 1 - dk if col == "auroc" else dk
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if dk > 0.6 else INK)
        ax.set_xticks(range(3), [SHORT[c] for c in CIC])
        ax.set_yticks(range(3), [SHORT[c] for c in CIC])
        ax.set_xlabel("Target")
        ax.grid(False)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("Source")
    fig.tight_layout()
    save(fig, "fig1_xd_heatmaps.png")


def fig2_os():
    d = _r("os_detection_all.csv")
    d = d[d.model == "LGBM"]
    g = d.groupby(["dataset", "held_out"]).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    mk = {"CIC-IDS2017": "o", "CSE-CIC-IDS2018": "s", "ACI-IoT-2023": "^", "UNSW-NB15": "D", "ToN-IoT": "v"}
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    ax.axvspan(-0.03, 0.5, ymin=0, ymax=1, color="#f3f2ee", zorder=0)
    for (ds, m_), c in zip(mk.items(), cols):
        x = g[g.dataset == ds]
        ax.scatter(x.recall_unseen_at_alpha, x["cm_0.05"], marker=m_, s=26, color=c, edgecolor="white",
                   linewidth=0.8, label=SHORT[ds], zorder=3)
    ax.axhline(0.5, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel(r"Recall of held-out family at $\alpha = 1\%$")
    ax.set_ylabel(r"Share of flows with $\hat p < 0.05$")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handletextpad=0.2,
              columnspacing=0.8)
    fig.tight_layout()
    save(fig, "fig2_os_scatter.png")


def fig3_recall():
    o = _r("os_priorities.csv")
    x = _r("xd_priorities.csv")
    if x is None:
        return
    o = o[o.model == "LGBM"]
    x = x[x.model == "LGBM"]
    prs = ["detector", "novelty", "fisher", "wfisher0.7"]
    panels = [("In-domain", x[x.regime == "In-domain"], ["source", "target"]),
              ("Cross-dataset", x[x.regime == "Cross-dataset"], ["source", "target"]),
              ("Unseen family", o, ["dataset", "held_out"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, (title, df, unit) in zip(axes, panels):
        for pr in prs:
            u = df[df.priority == pr].groupby(unit + ["budget"]).recall.mean().reset_index()
            s = u.groupby("budget").recall.agg(["mean", "sem"])
            ax.plot(100 * s.index, s["mean"], marker="o", ms=3, color=C[pr], label=LAB[pr])
            ax.fill_between(100 * s.index, s["mean"] - s["sem"], s["mean"] + s["sem"], color=C[pr], alpha=0.15, lw=0)
        ax.plot([1, 2, 5], [0.01, 0.02, 0.05], color=INK2, ls=":", lw=0.8, label="Random queue")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("Review budget (% of flows)")
        ax.set_xticks([1, 2, 5])
    axes[0].set_ylabel("Attack recall inside queue")
    axes[0].set_ylim(0, 1.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig3_recall_budget.png")


def fig4_load():
    xd = pd.read_csv(os.path.join(RES, "xd_triage.csv"))
    xd = xd[(xd.model == "LGBM") & (xd.priority == "fisher") & (xd.budget == 0.01)]
    w = _r("os_warmup.csv")
    w = w[(w.model == "LGBM") & (w.priority == "fisher") & (w.budget == 0.01)]
    xw = _r("xd_warmup.csv")
    rows = []
    for reg, lab in [("in-domain", "In-domain"), ("cross-dataset", "Cross-dataset")]:
        rows.append((lab, "source", xd[xd.setting == reg].load_src.values))
        if xw is not None:
            z = xw[(xw.model == "LGBM") & (xw.priority == "fisher") & (xw.budget == 0.01) & (xw.warmup == 0.2)
                   & (xw.regime == lab)]
            rows.append((lab, "warmup", z.load.values))
    if "rule" in w:
        rows.append(("Unseen family", "source", w[w.rule == "source"].load.values))
        rows.append(("Unseen family", "warmup", w[(w.rule == "warmup") & (w.warmup == 0.2)].load.values))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    regs = ["In-domain", "Cross-dataset", "Unseen family"]
    for i, reg in enumerate(regs):
        for k, rule in enumerate(["source", "warmup"]):
            v = [r[2] for r in rows if r[0] == reg and r[1] == rule]
            if not v or len(v[0]) == 0:
                continue
            v = np.clip(v[0], 1e-2, None)
            pos = i + (k - 0.5) * 0.35
            bp = ax.boxplot(v, positions=[pos], widths=0.28, patch_artist=True, showfliers=False,
                            whis=(10, 90), medianprops=dict(color=INK))
            bp["boxes"][0].set(facecolor=C[rule], alpha=0.7, edgecolor=C[rule])
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(3), regs)
    ax.set_ylabel("Queue load (1 = budget met)")
    for rule in ["source", "warmup"]:
        ax.plot([], [], color=C[rule], lw=6, alpha=0.7, label=LAB[rule])
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig4_budget_load.png")


def fig5_warmup():
    w = _r("os_warmup.csv")
    w = w[(w.model == "LGBM") & (w.priority == "fisher")]
    if "rule" in w:
        w = w[w.rule == "warmup"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, c in zip(axes, [0.01, 0.02, 0.05]):
        z = w[w.budget == c]
        g = z.groupby("warmup").agg(n=("n_warm", "median"), ne=("n_eval", "median"),
                                    q10=("load", lambda v: v.quantile(.1)), q50=("load", "median"),
                                    q90=("load", lambda v: v.quantile(.9)))
        sd = np.sqrt((1 - c) / c * (1 / g["n"] + 1 / g["ne"]))
        ax.fill_between(g["n"], 1 - 1.2816 * sd, 1 + 1.2816 * sd, color=GRID, label="Predicted 80% band (Eq. 5)")
        ax.plot(g["n"], g.q50, color=C["fisher"], marker="o", ms=3, label="Median load")
        ax.plot(g["n"], g.q10, color=C["fisher"], ls="--", lw=0.8, label="10th and 90th percentile")
        ax.plot(g["n"], g.q90, color=C["fisher"], ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"Budget {int(100 * c)}%", fontsize=9, color=INK)
        ax.set_xlabel("Warm-up window (flows)")
    axes[0].set_ylabel("Queue load")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, "fig5_warmup_size.png")


def fig6_temporal():
    s = _r("extra_temporal_stream.csv")
    if s is None:
        return
    s = s[(s.priority == "fisher") & (s.budget == 0.01)]
    protos = ["T1-benign-time", "T1-random", "T2-day", "T2-random"]
    rules = ["source", "warmup", "rolling", "online"]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    for k, rule in enumerate(rules):
        g = s[s.rule == rule].groupby("protocol")[["load", "load_min10", "load_max10"]].mean().reindex(protos)
        x = np.arange(len(protos)) + (k - 1.5) * 0.19
        ax.bar(x, g.load, width=0.17, color=C[rule], label=LAB[rule])
        ax.vlines(x, g.load_min10, g.load_max10, color=INK, lw=0.8)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(range(len(protos)), ["T1 (time)", "T1 (random)", "T2 (day)", "T2 (random)"])
    ax.set_ylabel("Queue load, 1% budget")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig6_temporal_load.png")


def fig7_cost():
    c = _r("extra_cost.csv")
    if c is None:
        return
    k = c[(c.component == "knn") & (c.threads == 4)]
    d = c[(c.component == "detector") & (c.threads == 4)]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    cols = {"brute": "#2a78d6", "kd_tree": "#eda100", "ball_tree": "#1baf7a", "hnsw": "#eb6834"}
    for algo, g in k.groupby("model"):
        ax.plot(g.n_ref, g.us_per_flow, marker="o", ms=3, color=cols.get(algo, INK),
                label={"brute": "Exact, brute force", "kd_tree": "Exact, k-d tree", "ball_tree": "Exact, ball tree",
                       "hnsw": "Approximate, HNSW"}[algo])
    lg = d[d.model == "LGBM"].us_per_flow
    if len(lg):
        ax.axhline(float(lg.iloc[0]), color=INK2, ls="--", lw=0.8, label="LightGBM inference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference size (benign flows)")
    ax.set_ylabel(r"Time per flow ($\mu$s), 4 threads")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    save(fig, "fig7_cost.png")


def fig8_adapt():
    cv, a = _r("xd_adapt_curve.csv"), _r("xd_adapt.csv")
    if cv is None:
        return
    cv = cv[cv.variant == "FB"]
    pp = cv.groupby(["source", "target", "budget", "segment"])[["recall", "precision"]].mean().reset_index()
    base = a[a.variant == "base_fisher"].groupby(["source", "target", "budget"])[["recall", "precision"]].mean()
    base = base.reset_index().groupby("budget").mean(numeric_only=True)
    cols = {0.01: "#2a78d6", 0.02: "#1baf7a", 0.05: "#eb6834"}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, m, lab in [(axs[0], "recall", "Queue recall"), (axs[1], "precision", "Queue precision")]:
        for c, g in pp.groupby("budget"):
            st = g.groupby("segment")[m].agg(["mean", "std", "count"])
            se = st["std"] / np.sqrt(st["count"])
            x = 20 * st.index + 10
            ax.plot(x, st["mean"], marker="o", ms=3, color=cols[c], label=f"Feedback, {c:.0%} budget")
            ax.fill_between(x, st["mean"] - se, st["mean"] + se, color=cols[c], alpha=0.15, lw=0)
            ax.axhline(base.loc[c, m], color=cols[c], ls="--", lw=0.8)
        ax.set_xticks([30, 50, 70, 90], ["20-40%", "40-60%", "60-80%", "80-100%"])
        ax.set_xlabel("Part of the target stream")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK2, ls="--", lw=0.8))
    l.append("Source pipeline (Fisher)")
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig8_adapt.png")


if __name__ == "__main__":
    for f in (fig1_heat, fig2_os, fig3_recall, fig4_load, fig5_warmup, fig6_temporal, fig7_cost, fig8_adapt):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("fail", f.__name__, repr(e))

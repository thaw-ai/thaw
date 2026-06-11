"""Generate the three figures for the re-feed drift paper from the raw
per-pivot JSONs in benchmarks/results/2026-06-10_drift_ablation/.

Run from the repo root:  python3 paper/refeed-drift/figures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "benchmarks/results/2026-06-10_drift_ablation/"
OUT = "paper/refeed-drift/"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "figure.dpi": 200,
})


def sign(x):
    return int(x > 0) - int(x < 0)


def load(name):
    with open(os.path.join(BASE, name + ".json")) as f:
        return json.load(f)["pivots"]


def common_eligible(recs):
    return [r for r in recs
            if sign(r["A_exact1"]) or sign(r["A_exact2"]) or sign(r["A_refeed"])]


# ---------------------------------------------------------------------------
# Figure 1: agreement heatmaps, exact-vs-refeed beside exact-vs-exact
# ---------------------------------------------------------------------------

def fig1():
    recs = common_eligible(load("primary") + load("screened"))
    vals = np.round(np.arange(-1.0, 1.01, 0.25), 2)
    idx = {v: i for i, v in enumerate(vals)}

    def grid(a_key, b_key):
        g = np.zeros((len(vals), len(vals)))
        for r in recs:
            a = round(r[a_key], 2)
            b = round(r[b_key], 2)
            if a in idx and b in idx:
                g[idx[b], idx[a]] += 1
        return g

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, (b_key, title, flips) in zip(axes, [
        ("A_refeed", "re-feed vs exact resume", None),
        ("A_exact2", "exact replica vs exact resume (floor)", None),
    ]):
        g = grid("A_exact1", b_key)
        n = int(g.sum())
        fl = sum(1 for r in recs if sign(r["A_exact1"]) != sign(r[b_key]))
        with np.errstate(divide="ignore"):
            ax.imshow(np.log10(np.where(g > 0, g, np.nan)),
                      origin="lower", cmap="viridis", aspect="equal")
        # shade sign-disagreement region
        for i, bv in enumerate(vals):
            for j, av in enumerate(vals):
                if sign(av) != sign(bv) and not (sign(av) == 0 and sign(bv) == 0):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, edgecolor="crimson",
                                               linewidth=0.35, alpha=0.55))
                if g[i, j] > 0:
                    ax.text(j, i, int(g[i, j]), ha="center", va="center",
                            fontsize=5.5,
                            color="white" if g[i, j] < g.max() / 3 else "black")
        ax.set_title(f"{title}\nsign flips: {fl}/{n} = {fl/n:.1%}")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=6.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([f"{v:g}" for v in vals], fontsize=6.5)
        ax.set_xlabel(r"$\hat{A}_t$ (exact resume, pass 1)")
    axes[0].set_ylabel(r"$\hat{A}_t$ (comparison pass)")
    fig.suptitle("Pooled main configuration, common eligible set (n=622), "
                 "red boxes = sign disagreement", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "fig_quadrants.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: flip rate vs floor across configurations
# ---------------------------------------------------------------------------

def fig2():
    runs = [
        ("Main pooled\n(t=0.7, K=4)", common_eligible(load("primary") + load("screened"))),
        ("Greedy\n(t=0, K=1)", common_eligible(load("temp0"))),
        ("K=8 partial\n(t=0.7)", common_eligible(load("k8"))),
        ("GRPO ckpt\n(t=0.7, K=4)", common_eligible(load("grpo"))),
    ]
    labels, refeed, floor, ns = [], [], [], []
    for label, recs in runs:
        n = len(recs)
        labels.append(label)
        ns.append(n)
        refeed.append(sum(sign(r["A_exact1"]) != sign(r["A_refeed"]) for r in recs) / n)
        floor.append(sum(sign(r["A_exact1"]) != sign(r["A_exact2"]) for r in recs) / n)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    b1 = ax.bar(x - w / 2, [v * 100 for v in refeed], w,
                label="re-feed vs exact", color="#c0392b")
    b2 = ax.bar(x + w / 2, [v * 100 for v in floor], w,
                label="replica floor", color="#7f8c8d")
    for i in range(len(labels)):
        ax.annotate(f"+{(refeed[i]-floor[i])*100:.1f}pp",
                    xy=(x[i], max(refeed[i], floor[i]) * 100 + 1.2),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labels, ns)], fontsize=7.5)
    ax.set_ylabel("credit sign-flip rate (%)")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + "fig_configs.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: ECDF of first-token logprob deltas, replica vs re-feed
# ---------------------------------------------------------------------------

def fig3():
    recs = (load("primary") + load("screened") + load("temp0") + load("grpo"))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for tag, color, label in (
        ("exact2", "#7f8c8d", "exact replica"),
        ("refeed", "#c0392b", "re-feed"),
    ):
        ds = np.array([r[f"greedy_first_logprob_delta_{tag}"] for r in recs
                       if r.get(f"greedy_first_logprob_delta_{tag}") is not None])
        zero_frac = float((ds == 0).mean())
        nz = np.sort(ds[ds > 0])
        y = zero_frac + (1 + np.arange(len(nz))) / len(ds)
        ax.step(np.concatenate([[1e-8], nz]),
                np.concatenate([[zero_frac], y]) * 100,
                where="post", color=color,
                label=f"{label} (exactly 0 at {zero_frac:.1%})")
        ax.scatter([1e-8], [zero_frac * 100], s=18, color=color, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 0.3)
    ax.set_ylim(0, 102)
    ax.set_xlabel("|first-token logprob delta| vs exact pass 1")
    ax.set_ylabel("cumulative % of probes")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("point mass at exactly 0", xy=(1.6e-8, 80), fontsize=7.5,
                color="#444444")
    fig.tight_layout()
    fig.savefig(OUT + "fig_logprob_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", OUT + "fig_quadrants.pdf, fig_configs.pdf, fig_logprob_ecdf.pdf")

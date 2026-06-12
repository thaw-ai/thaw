"""Generate the figures for the re-feed drift paper from the raw per-pivot
JSONs. Second campaign (instrumented, non-overlapping cohorts) is primary:
benchmarks/results/2026-06-11_pressure_test/. Two first-campaign runs (greedy,
GRPO checkpoint) remain valid standalone and are loaded from
benchmarks/results/2026-06-10_drift_ablation/.

Run from the repo root:  python3 paper/refeed-drift/figures.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "benchmarks")
from drift_ablation import dedupe  # noqa: E402

BASE2 = "benchmarks/results/2026-06-11_pressure_test/"
BASE1 = "benchmarks/results/2026-06-10_drift_ablation/"
OUT = "paper/refeed-drift/"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "figure.dpi": 200,
})


def sign(x):
    return int(x > 0) - int(x < 0)


def load(base, name):
    with open(os.path.join(base, name + ".json")) as f:
        return json.load(f)["pivots"]


def common_eligible(recs):
    return [r for r in recs
            if sign(r["A_exact1"]) or sign(r["A_exact2"]) or sign(r["A_refeed"])]


def pooled_main():
    return common_eligible(dedupe(
        load(BASE2, "primary") + load(BASE2, "screened")))


# ---------------------------------------------------------------------------
# Figure 1: agreement heatmaps, re-feed vs replica floor (pooled main)
# ---------------------------------------------------------------------------

def fig1():
    recs = pooled_main()
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
    for ax, (b_key, title) in zip(axes, [
        ("A_refeed", "re-feed vs exact resume"),
        ("A_exact2", "exact replica vs exact resume (floor)"),
    ]):
        g = grid("A_exact1", b_key)
        n = int(g.sum())
        fl = sum(1 for r in recs if sign(r["A_exact1"]) != sign(r[b_key]))
        with np.errstate(divide="ignore"):
            ax.imshow(np.log10(np.where(g > 0, g, np.nan)),
                      origin="lower", cmap="viridis", aspect="equal")
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
        ax.set_title(f"{title}\ndisagreements: {fl}/{n} = {fl/n:.1%}")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=6.5)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels([f"{v:g}" for v in vals], fontsize=6.5)
        ax.set_xlabel(r"$\hat{A}_t$ (exact resume, pass 1)")
    axes[0].set_ylabel(r"$\hat{A}_t$ (comparison pass)")
    fig.suptitle("Pooled main configuration (40 non-overlapping problems, "
                 "common eligible set n=636); red boxes = sign disagreement",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT + "fig_quadrants.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: flip rate vs floor across all seven configurations
# ---------------------------------------------------------------------------

def fig2():
    runs = [
        ("Main pooled\nQwen t=.7 K4", pooled_main()),
        ("K=8\nQwen", common_eligible(load(BASE2, "k8"))),
        ("K=16\nQwen", common_eligible(load(BASE2, "k16"))),
        ("Greedy K=1$^{\\diamond}$\nQwen", common_eligible(load(BASE1, "temp0"))),
        ("GRPO ckpt$^{\\diamond}$\nSimpleRL", common_eligible(load(BASE1, "grpo"))),
        ("Phi-4-mini\nt=.7 K4", common_eligible(load(BASE2, "phi"))),
        ("batch-\ninvariant", common_eligible(load(BASE2, "batchinv"))),
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
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - w / 2, [v * 100 for v in refeed], w,
           label="re-feed vs exact resume", color="#c0392b")
    ax.bar(x + w / 2, [v * 100 for v in floor], w,
           label="replica floor", color="#7f8c8d")
    for i in range(len(labels)):
        exc = (refeed[i] - floor[i]) * 100
        txt = f"+{exc:.1f}pp" if exc > 0 else "0: bitwise\nidentical"
        ax.annotate(txt, xy=(x[i], max(refeed[i], floor[i]) * 100 + 1.3),
                    ha="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labels, ns)], fontsize=7)
    ax.set_ylabel("estimate disagreement rate (%)")
    ax.set_ylim(0, 58)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + "fig_configs.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: ECDF of first-token logprob deltas, replica vs re-feed,
# with the batch-invariant run as the degenerate reference
# ---------------------------------------------------------------------------

def fig3():
    recs = (load(BASE2, "primary") + load(BASE2, "screened")
            + load(BASE2, "k8") + load(BASE2, "phi"))
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
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
    ax.axhline(100, color="#2980b9", linewidth=0.8, linestyle="--")
    ax.annotate("batch-invariant kernels: both at 100% exactly 0",
                xy=(2e-8, 95.5), fontsize=7, color="#2980b9")
    ax.set_xscale("log")
    ax.set_xlim(1e-8, 0.3)
    ax.set_ylim(0, 104)
    ax.set_xlabel("per-pivot max |first-token logprob delta| vs exact pass 1")
    ax.set_ylabel("cumulative % of pivots")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + "fig_logprob_ecdf.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote fig_quadrants.pdf, fig_configs.pdf, fig_logprob_ecdf.pdf")

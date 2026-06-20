# -*- coding: utf-8 -*-
"""
Reproduce Figure 6 — confusion matrices of the two blind large language models
(Claude Sonnet 4.5; GPT-5.5) against the human two-coder consensus. Color encodes
row-normalized recovery; annotations are segment counts.

Input : gold_standard/llm_blind_validation_180.csv  (SEG_ID, human, claude_cat, gpt_cat, ...)
Output: figures/figure6_llm_confusion.{png,pdf}
Requires: pandas, numpy, scikit-learn, matplotlib.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
CSV = os.path.join(ROOT, "gold_standard", "llm_blind_validation_180.csv")
ORDER = ["symbolic", "soft_substantive", "verifiable"]
SHORT = ["Symbolic", "Soft subst.", "Verifiable"]


def panel(human, pred):
    d = pd.DataFrame({"h": human, "p": pred}).dropna()
    d = d[d.p.isin(ORDER) & d.h.isin(ORDER)]
    M = pd.crosstab(d.h, d.p).reindex(index=ORDER, columns=ORDER).fillna(0).astype(int).values
    return M, cohen_kappa_score(d.h, d.p), np.trace(M) / M.sum(), int(M.sum())


def main():
    df = pd.read_csv(CSV)
    panels = [("Claude Sonnet 4.5", *panel(df.human, df.claude_cat)),
              ("GPT-5.5", *panel(df.human, df.gpt_cat))]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for ax, (name, M, k, agree, n) in zip(axes, panels):
        row = M / M.sum(axis=1, keepdims=True)
        im = ax.imshow(row, cmap="Greys", vmin=0, vmax=1, aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]}", ha="center", va="center",
                        color="white" if row[i, j] > 0.55 else "black",
                        fontsize=11, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(3)); ax.set_xticklabels(SHORT, fontsize=8.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(SHORT, fontsize=8.5, rotation=90, va="center")
        ax.set_xlabel("LLM prediction", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Human consensus", fontsize=9)
        ax.set_title(f"{name}\n$\\kappa$ = {k:.2f},  {agree*100:.1f}% agree  ($n$ = {n})", fontsize=9.5)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="0.6", linewidth=0.6)
        ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cb.set_label("Row-normalized recovery", fontsize=8.5); cb.ax.tick_params(labelsize=8)

    outdir = os.path.join(ROOT, "figures")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "figure6_llm_confusion.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "figure6_llm_confusion.pdf"), bbox_inches="tight")
    print("saved figures/figure6_llm_confusion.png / .pdf")
    for name, M, k, agree, n in panels:
        print(f"{name}: kappa={k:.3f} agree={agree:.3f} n={n}")


if __name__ == "__main__":
    main()

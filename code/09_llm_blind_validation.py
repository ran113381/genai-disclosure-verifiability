# -*- coding: utf-8 -*-
"""
Reproduce the Section 3.5 third-party large-language-model blind validation.

Two independent frontier LLMs (Claude Opus 4.6; a GPT-class model) classified
each of the 180 stratified gold segments into {verifiable, soft_substantive,
symbolic} from the segment text ALONE, given only the construct definition
(no keyword dictionary, no human labels, no each other's output), temperature 0.
Their stored per-segment predictions are scored here against the two-coder HUMAN
CONSENSUS (the 174 of 180 segments where blind coders A and B agree).

Input : gold_standard/llm_blind_validation_180.csv
        (SEG_ID, human [= A/B consensus label], claude_cat, claude_score,
         gpt_cat, gpt_score)
Output: prints kappa / % agreement / Spearman rho for each model vs the human
        consensus, and the inter-model agreement.

Requires: pandas, scikit-learn, scipy.
"""
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

CSV = os.path.join(os.path.dirname(__file__), "..", "gold_standard",
                   "llm_blind_validation_180.csv")
ORD = {"symbolic": 0, "soft_substantive": 1, "verifiable": 2}
CATS = set(ORD)


def score(human, pred, tag):
    d = pd.DataFrame({"h": human, "p": pred}).dropna()
    d = d[d["p"].isin(CATS) & d["h"].isin(CATS)]
    k3 = cohen_kappa_score(d["h"], d["p"])
    agree = (d["h"] == d["p"]).mean()
    kb = cohen_kappa_score((d["h"] == "verifiable").astype(int),
                           (d["p"] == "verifiable").astype(int))
    rho, p = spearmanr(d["h"].map(ORD), d["p"].map(ORD))
    print(f"{tag:42s} n={len(d):3d}  kappa(3-class)={k3:.3f}  "
          f"%agree={agree:.3f}  kappa(binary)={kb:.3f}  rho={rho:.3f} (p={p:.1e})")
    return d


def main():
    df = pd.read_csv(CSV)
    print("=" * 78)
    print("Section 3.5 — LLM blind validation vs A/B human consensus (n = 174)")
    print("=" * 78)
    score(df["human"], df["claude_cat"], "Claude Opus 4.6 vs human consensus")
    score(df["human"], df["gpt_cat"],    "GPT-class vs human consensus")
    # inter-model
    dd = df[df["claude_cat"].isin(CATS) & df["gpt_cat"].isin(CATS)]
    k = cohen_kappa_score(dd["claude_cat"], dd["gpt_cat"])
    rho, _ = spearmanr(dd["claude_cat"].map(ORD), dd["gpt_cat"].map(ORD))
    print(f"{'Claude vs GPT (inter-model)':42s} n={len(dd):3d}  kappa(3-class)={k:.3f}  "
          f"%agree={(dd['claude_cat']==dd['gpt_cat']).mean():.3f}  rho={rho:.3f}")
    print("\nConfusion (Claude, rows = human consensus):")
    cm = pd.crosstab(df["human"], df["claude_cat"])
    print(cm.to_string())
    print("\nNote: the original triangulation scripts mistakenly scored the LLM "
          "labels against the\nmachine label (paper1_category), not the human "
          "consensus; this script uses the human\nconsensus, reproducing the "
          "Section 3.5 values (Claude kappa 0.70, GPT 0.54, inter-model 0.68).")


if __name__ == "__main__":
    main()

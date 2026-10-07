# -*- coding: utf-8 -*-
"""
Reproduce the Section 3.5 third-party large-language-model blind validation.

Two independent frontier LLMs (Claude Sonnet 4.5; a GPT-class model) classified
each of the 180 stratified gold segments into {verifiable, soft_substantive,
symbolic} from the segment text ALONE, given only the construct definition
(no keyword dictionary, no human labels, no each other's output), temperature 0.
Their stored per-segment predictions are scored here against the two-coder HUMAN
CONSENSUS (the 174 of 180 segments where blind coders A and B agree).

Input : gold_standard/llm_blind_validation_180.csv
        (SEG_ID, human [= A/B consensus label; NC for the six segments without
         two-coder consensus], claude_cat, claude_score, gpt_cat, gpt_score)
Output: prints kappa / % agreement / Spearman rho for each model vs the human
        consensus (the 174 consensus segments only), and the inter-model
        agreement on (a) all segments both models labeled and (b) those within
        the human consensus.

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
    cons = df[df["human"].isin(CATS)]   # drop NC (no two-coder consensus) rows
    print("=" * 78)
    print("Section 3.5 — LLM blind validation vs A/B human consensus (n = 174)")
    print("=" * 78)
    score(cons["human"], cons["claude_cat"], "Claude Sonnet 4.5 vs human consensus")
    score(cons["human"], cons["gpt_cat"],    "GPT-class vs human consensus")
    # inter-model, on two bases
    both = df[df["claude_cat"].isin(CATS) & df["gpt_cat"].isin(CATS)]
    for tag, dd in [("Claude vs GPT (both valid)", both),
                    ("Claude vs GPT (both valid, in consensus)", both[both["human"].isin(CATS)])]:
        k = cohen_kappa_score(dd["claude_cat"], dd["gpt_cat"])
        rho, _ = spearmanr(dd["claude_cat"].map(ORD), dd["gpt_cat"].map(ORD))
        print(f"{tag:42s} n={len(dd):3d}  kappa(3-class)={k:.3f}  "
              f"%agree={(dd['claude_cat']==dd['gpt_cat']).mean():.3f}  rho={rho:.3f}")
    print("\nConfusion (Claude, rows = human consensus):")
    cm = pd.crosstab(cons["human"], cons["claude_cat"])
    print(cm.to_string())
    print("\nNote: the original triangulation scripts mistakenly scored the LLM "
          "labels against the\nmachine label (paper1_category), not the human "
          "consensus; this script uses the human\nconsensus, reproducing the "
          "Section 3.5 values (Claude kappa 0.70, GPT 0.54;\ninter-model 0.68 on the 156 "
          "segments both models labeled, 0.67 on the\n152 of those within the human "
          "consensus).")


if __name__ == "__main__":
    main()

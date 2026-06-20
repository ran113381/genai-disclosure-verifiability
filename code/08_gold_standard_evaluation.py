# -*- coding: utf-8 -*-
"""
Reproduce the two-stage human-gold-standard evaluation of the measurement pipeline.

Stage 1  Dictionary identification (binary: is a passage a genuine enterprise GenAI
         disclosure?). Source: gold_standard/identification_gold_standard_600.xlsx
         (600 passages, two coders fully adjudicated). Compares the dictionary
         machine_label against the adjudicated human gold_label.

Stage 2  Verifiability scoring (segment-level "verifiable vs. rest"). Source:
         gold_standard/02_gold_standard.xlsx + 01_blind_coder_A/B_COMPLETED.xlsx
         (180 stratified segments). Compares the cue-rule verifiable_segment against
         the two-coder human consensus, and reports inter-coder reliability.

Usage:  python code/08_gold_standard_evaluation.py   (run from the repo root)
Requires: pandas, scikit-learn.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

GS = os.path.join(os.path.dirname(__file__), "..", "gold_standard")


def _to01(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return np.nan


def stage1_identification():
    print("=" * 70, "\nSTAGE 1: dictionary identification (binary), n=600\n", "=" * 70)
    df = pd.read_excel(os.path.join(GS, "identification_gold_standard_600.xlsx"))
    df["gold"] = df["gold_label"].map(_to01)
    df["mach"] = df["machine_label"].map(_to01)
    df["c1"] = df["coder1_label"].map(_to01)
    df["c2"] = df["coder2_label"].map(_to01)
    u = df.dropna(subset=["gold", "mach"])
    print(f"usable n = {len(u)}; gold positive rate = {u.gold.mean():.3f}")
    print(classification_report(u.gold, u.mach,
          target_names=["not-GenAI (0)", "GenAI (1)"], digits=3, zero_division=0))
    print("Cohen's kappa (machine vs human gold):", round(cohen_kappa_score(u.gold, u.mach), 3))
    print("confusion [rows=gold, cols=machine]:\n", confusion_matrix(u.gold, u.mach))
    cc = df.dropna(subset=["c1", "c2"])
    print(f"inter-coder: n={len(cc)}  kappa={cohen_kappa_score(cc.c1, cc.c2):.3f}  "
          f"agreement={(cc.c1 == cc.c2).mean():.3f}")


def stage2_verifiability():
    print("\n" + "=" * 70, "\nSTAGE 2: verifiability cue-rule vs human (binary), n=180\n", "=" * 70)
    def load_coder(f):
        d = pd.read_excel(os.path.join(GS, f), sheet_name=1).iloc[:, [0, 2]]
        d.columns = ["SEG_ID", "human"]
        d["human"] = d["human"].astype(str).str.strip().replace({"soft_substantive": "soft"})
        return d
    A = load_coder("01_blind_coder_A_COMPLETED.xlsx")
    B = load_coder("01_blind_coder_B_COMPLETED.xlsx")
    gs = pd.read_excel(os.path.join(GS, "02_gold_standard.xlsx"))
    m = A.merge(B, on="SEG_ID", suffixes=("_A", "_B")).merge(
        gs[["SEG_ID", "verifiable_segment"]], on="SEG_ID")
    cons = m[m.human_A == m.human_B].copy()          # human consensus = where coders agree
    cons["hum_bin"] = (cons.human_A == "verifiable").astype(int)
    cons["rule_bin"] = cons.verifiable_segment.astype(int)
    print(f"consensus n = {len(cons)} of {len(m)}; human-verifiable rate = {cons.hum_bin.mean():.3f}")
    print(classification_report(cons.hum_bin, cons.rule_bin,
          target_names=["not-verifiable", "verifiable"], digits=3, zero_division=0))
    print("Cohen's kappa (rule vs human):", round(cohen_kappa_score(cons.hum_bin, cons.rule_bin), 3))
    print("Note: the three-way verifiable/soft/symbolic typology is firm-year level; "
          "this segment-level check is binary verifiable-vs-rest. A learned model on the "
          "same five cues barely improves on the rule (kappa ~0.31), indicating the "
          "binding constraint is the five-cue representation, not the rule logic.")


if __name__ == "__main__":
    stage1_identification()
    stage2_verifiability()

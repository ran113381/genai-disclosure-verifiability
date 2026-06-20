# -*- coding: utf-8 -*-
"""
Reproduce Table 5 — benchmarking the cue rule against learned classifiers on the
180-segment human gold standard (binary verifiable-vs-rest on the 174 segments
where both blind coders agree). This is the apples-to-apples comparison with the
cue rule's reported kappa = 0.231.

Methods (all evaluated on the SAME consensus set, same metric):
  - cue rule (deterministic)                 -> kappa 0.231
  - logistic regression on the 5 cues        -> ~0.21
  - gradient-boosted trees on the 5 cues     -> ~0.23
  - logistic regression on TEXT (TF-IDF char n-grams)        -> ~0.45
  - logistic regression on TEXT (frozen Chinese transformer  -> ~0.55
    embeddings, BAAI/bge-large-zh-v1.5; skipped if sentence-transformers
    or the model is unavailable)

Learned rows use out-of-fold predictions under repeated stratified five-fold
cross-validation (20 repeats). Input: gold_standard/02_gold_standard.xlsx +
01_blind_coder_A/B_COMPLETED.xlsx. Requires: pandas, scikit-learn, lightgbm
(+ optional sentence-transformers/torch for the transformer row).
"""
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

GS = os.path.join(os.path.dirname(__file__), "..", "gold_standard")
SEED = 42
CUES = ["has_completion", "has_quant", "has_artifact", "has_partner", "has_current"]


def load_coder(fname):
    d = pd.read_excel(os.path.join(GS, fname), sheet_name=1).iloc[:, [0, 2]]
    d.columns = ["SEG_ID", "human"]
    d["human"] = d["human"].astype(str).str.strip().replace({"soft_substantive": "soft"})
    return d


def main():
    A = load_coder("01_blind_coder_A_COMPLETED.xlsx")
    B = load_coder("01_blind_coder_B_COMPLETED.xlsx")
    gs = pd.read_excel(os.path.join(GS, "02_gold_standard.xlsx"))
    m = (A.merge(B, on="SEG_ID", suffixes=("_A", "_B"))
           .merge(gs[["SEG_ID", "verifiable_segment", "segment_text"] + CUES], on="SEG_ID"))
    cons = m[m.human_A == m.human_B].copy()
    y = (cons.human_A == "verifiable").astype(int).values
    Xcue = cons[CUES].astype(float).values
    texts = cons.segment_text.astype(str).values
    n = len(cons)
    print(f"consensus n = {n}; human-verifiable rate = {y.mean():.3f}\n")

    rows = []

    def add(tag, kappa, f1m, f1v, acc):
        rows.append([tag, round(kappa, 3), round(f1m, 3), round(f1v, 3), round(acc, 3)])
        print(f"{tag:42s} kappa={kappa:.3f}  macroF1={f1m:.3f}  verF1={f1v:.3f}  acc={acc:.3f}")

    # deterministic cue rule
    r = cons.verifiable_segment.astype(int).values
    add("cue rule (5 cues, deterministic)", cohen_kappa_score(y, r),
        f1_score(y, r, average="macro"), f1_score(y, r, pos_label=1), accuracy_score(y, r))

    def cv(make_est, X):
        ks, fm, fv, ac = [], [], [], []
        for rep in range(20):
            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED + rep)
            pred = cross_val_predict(make_est(), X, y, cv=skf)
            ks.append(cohen_kappa_score(y, pred)); fm.append(f1_score(y, pred, average="macro"))
            fv.append(f1_score(y, pred, pos_label=1, zero_division=0)); ac.append(accuracy_score(y, pred))
        return np.mean(ks), np.mean(fm), np.mean(fv), np.mean(ac)

    add("logistic regression (5 cues)", *cv(
        lambda: LogisticRegression(max_iter=1000, class_weight="balanced"), Xcue))
    add("gradient-boosted trees (5 cues)", *cv(
        lambda: lgb.LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.05,
                                   min_child_samples=5, verbose=-1, random_state=SEED), Xcue))
    add("logistic regression (TF-IDF char n-gram)", *cv(
        lambda: Pipeline([("tf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          min_df=2, sublinear_tf=True)),
                          ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]), texts))

    # optional: frozen Chinese transformer embeddings
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "0")
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cpu")
        Xemb = enc.encode(list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        add("logistic regression (bge-large-zh embeddings)", *cv(
            lambda: LogisticRegression(max_iter=3000, class_weight="balanced"), Xemb))
    except Exception as e:
        print(f"[transformer row skipped: {type(e).__name__} — install sentence-transformers + "
              f"download BAAI/bge-large-zh-v1.5 to reproduce kappa ~0.55]")

    out = pd.DataFrame(rows, columns=["method", "kappa", "macro_f1", "verifiable_f1", "accuracy"])
    dst = os.path.join(os.path.dirname(__file__), "..", "data", "table5_benchmark_results.csv")
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\nsaved {os.path.relpath(dst)}")
    print("Reference: full-text zero-shot LLM blind classification reaches kappa = 0.70 (see code/09).")


if __name__ == "__main__":
    main()

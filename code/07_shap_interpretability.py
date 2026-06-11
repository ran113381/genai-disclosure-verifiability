# -*- coding: utf-8 -*-
"""
SHAP interpretability analysis for the GenAI-disclosure verifiability measurement system
(paper 1, submission to Information Processing & Management).

Pipeline
--------
Step 0  Reverse-engineer / verify the exact measurement rules
        (verifiable_score -> verifiable_segment threshold rule; label classes;
         Type_V5 firm-year derivation) and store them in results_summary.json.
A1      Segment level. Features = five rule cues (has_*) + segment length.
        Target = three-class segment type (verifiable / soft_substantive / symbolic),
        derived consistently with the firm-year Type_V5 typology:
            verifiable        : verifiable_segment == 1
            soft_substantive  : label == 'substantive' and verifiable_segment == 0
            symbolic          : otherwise (label in {'generic','strategic'}, not verifiable)
        LightGBM multiclass, 5-fold stratified CV (accuracy / macro-F1 / AUC-ovr),
        SHAP TreeExplainer beeswarm + global bar for the "verifiable" class.
        NOTE: cues are inputs of the rule itself -> near-deterministic performance is
        expected; this analysis is a transparency / contribution decomposition exercise.
A2      Segment level, text only. Features = char n-gram TF-IDF (2-4, min_df=20,
        max_features=3000), NO rule cues. Same target / model / CV.  SHAP top n-grams
        for the "verifiable" class are mapped to the five cue dimensions (convergent
        validity: does a cue-blind learner re-discover the five-dimension design?).
B       Firm-year level. First-disclosure firms with Tobin's Q observable at event
        time -1 and +1. Outcome = TobinQ(t+1) level with TobinQ(t-1) as a baseline
        feature (chosen over delta-Q: same information, markedly more stable CV R^2,
        hence more interpretable SHAP; the delta-Q variant is also reported in JSON).
        Controls-only vs controls+text LightGBM regressions, 5-fold CV R^2, delta R^2,
        SHAP beeswarm (top 15) of the full model.

Usage
-----
D:/python.exe -X utf8 run_shap_analysis.py ^
    --features-pkl "<...>/paper1_verifiability_features.pkl" ^
    --event-panel  "<...>/paper1_v11_strict_event_panel_with_extra_controls.csv" ^
    --cohort       "<...>/cohort_enriched_1123.csv" ^
    --outdir       "<output dir>"

All inputs are read-only; everything is written to --outdir.
"""

import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

warnings.filterwarnings("ignore")

SEED = 42
CUES = ["has_completion", "has_quant", "has_artifact", "has_partner", "has_current"]
CUE_LABELS_EN = {
    "has_completion": "Implementation verb (completion)",
    "has_quant": "Quantified rollout (quant)",
    "has_artifact": "Product/tool artifact (artifact)",
    "has_partner": "Partner/customer (partner)",
    "has_current": "Time anchoring (current)",
}

# ---------------------------------------------------------------- figures ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, outdir, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def shap_values_for_class(explainer, X, class_idx):
    """Return SHAP matrix (n, p) for one class, robust to shap API versions."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # old API: list of (n, p) per class
        return sv[class_idx]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # new API: (n, p, n_classes)
        return sv[:, :, class_idx]
    return sv  # binary / regression


# =============================================================== step 0 ======
def step0_rules(seg, feat, outdir):
    print("=" * 70, "\nSTEP 0: measurement rules\n", "=" * 70)
    rules = {}

    rules["label_distribution"] = seg["label"].value_counts().to_dict()
    rules["verifiable_score_distribution"] = (
        seg["verifiable_score"].value_counts().sort_index().to_dict()
    )
    rules["crosstab_score_x_verifiable_segment"] = (
        pd.crosstab(seg["verifiable_score"], seg["verifiable_segment"])
        .astype(int).to_dict()
    )

    # (1) verifiable_score = simple sum of the five cues
    score_is_sum = bool(
        (seg["verifiable_score"] == seg[CUES].sum(axis=1)).all()
    )
    rules["rule_verifiable_score"] = {
        "statement": "verifiable_score = has_completion + has_quant + has_artifact + has_partner + has_current (unweighted sum, 0-5)",
        "verified_on_all_20703_segments": score_is_sum,
    }

    # (2) verifiable_segment: NOT a pure score cutoff; exact conditional rule
    c, q, a, p, t = (seg[x].astype(bool) for x in CUES)
    rule_pred = (c & (q | a | p)) | (q & a & (p | t))
    n_mismatch = int((rule_pred.astype(int) != seg["verifiable_segment"]).sum())
    rules["rule_verifiable_segment"] = {
        "statement": (
            "verifiable_segment = 1  iff  [has_completion AND (has_quant OR has_artifact OR has_partner)] "
            "OR [has_quant AND has_artifact AND (has_partner OR has_current)]. "
            "I.e. an implementation/completion verb must be corroborated by at least one non-time cue "
            "(quantification, artifact, or partner); absent a completion verb, the segment needs the "
            "quant+artifact pair plus one further cue (partner or time anchor). has_current alone never "
            "qualifies as corroboration; a pure score>=2 cutoff does NOT reproduce the flag "
            "(score 2: only 2018/6426 verifiable; score 3: 1724/2095; score>=4: all verifiable; score<=1: none).",
        ),
        "mismatches_on_20703_segments": n_mismatch,
        "verified_exactly": n_mismatch == 0,
    }

    # (3) label = independent base 'substance' classification (3 classes)
    rules["label_definition"] = (
        "label in {generic, strategic, substantive} is the base substance classification of the "
        "segment text, assigned independently of the five cues (cf. crosstab: every label class "
        "spans the whole score range). 'substantive' marks concrete application/deployment language; "
        "'strategic' marks plan/strategy language; 'generic' marks boilerplate mentions."
    )
    rules["crosstab_label_x_score"] = (
        pd.crosstab(seg["label"], seg["verifiable_score"]).astype(int).to_dict()
    )

    # (4) firm-year aggregation / Type_V5 priority rule (verified by recomputation)
    agg = (
        seg.groupby(["Firm_ID", "Year"])
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "n_sub": int((g.label == "substantive").sum()),
                    "n_ver": int(g.verifiable_segment.sum()),
                    "mean_score": g.verifiable_score.mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    m = feat.merge(agg, on=["Firm_ID", "Year"], how="left")
    rules["rule_firm_year_aggregation"] = {
        "N_Segments": bool((m.N_Segments == m.n).all()),
        "N_Substantive = count(label=='substantive')": bool((m.N_Substantive == m.n_sub).all()),
        "N_Verifiable = count(verifiable_segment==1)": bool((m.N_Verifiable == m.n_ver).all()),
        "Verifiable_Share = N_Verifiable/N_Segments": bool(
            np.allclose(m.Verifiable_Share, m.N_Verifiable / m.N_Segments)
        ),
        "Mean_Verifiable_Score = mean(verifiable_score)": bool(
            np.allclose(m.Mean_Verifiable_Score, m.mean_score)
        ),
        "Type_V5_priority": (
            "verifiable if N_Verifiable>0; else soft_substantive if N_Substantive>0; "
            "else symbolic (strict priority chain; Has_* dummies are mutually exclusive)"
        ),
    }

    # (5) segment-level three-class target used in A1/A2 (mirrors Type_V5 logic)
    rules["segment_three_class_target"] = (
        "verifiable: verifiable_segment==1; soft_substantive: label=='substantive' & "
        "verifiable_segment==0; symbolic: otherwise. (The raw 'label' column is the 3-class "
        "generic/strategic/substantive base classification, NOT the paper typology, so the "
        "paper's verifiable/soft-substantive/symbolic typology is derived as stated.)"
    )

    seg = seg.copy()
    seg["seg_class"] = np.where(
        seg.verifiable_segment == 1,
        "verifiable",
        np.where(seg.label == "substantive", "soft_substantive", "symbolic"),
    )
    rules["segment_three_class_distribution"] = seg["seg_class"].value_counts().to_dict()
    print(json.dumps(to_py(rules), ensure_ascii=False, indent=1)[:2000])
    return rules, seg


# ============================================================ CV helper ======
def cv_classification(model_fn, X_fn, y, n_splits=5):
    """X_fn(train_idx, test_idx) -> X_tr, X_te (allows fold-wise TF-IDF fitting)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    accs, f1s, aucs = [], [], []
    classes = np.unique(y)
    for tr, te in skf.split(np.zeros(len(y)), y):
        X_tr, X_te = X_fn(tr, te)
        mdl = model_fn()
        mdl.fit(X_tr, y[tr])
        pred = mdl.predict(X_te)
        proba = mdl.predict_proba(X_te)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        aucs.append(
            roc_auc_score(y[te], proba, multi_class="ovr", average="macro", labels=classes)
        )
    return {
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
        "auc_ovr_mean": float(np.mean(aucs)), "auc_ovr_sd": float(np.std(aucs)),
        "per_fold": {"accuracy": accs, "macro_f1": f1s, "auc_ovr": aucs},
    }


# =============================================================== A1 ==========
def analysis_a1(seg, outdir):
    print("=" * 70, "\nA1: rule-cue contribution decomposition (segment level)\n", "=" * 70)
    X = seg[CUES].copy()
    X["seg_len"] = seg["segment_text"].str.len()
    X.columns = [CUE_LABELS_EN.get(c, c) for c in CUES] + ["Segment length (chars)"]
    y = seg["seg_class"].values

    def mk():
        return lgb.LGBMClassifier(
            objective="multiclass", n_estimators=300, learning_rate=0.1,
            num_leaves=31, min_child_samples=20, random_state=SEED, verbose=-1,
        )

    metrics = cv_classification(mk, lambda tr, te: (X.iloc[tr], X.iloc[te]), y)
    print("A1 CV:", {k: round(v, 4) for k, v in metrics.items() if k != "per_fold"})

    # SHAP on full-data model (explanation model, standard practice)
    mdl = mk().fit(X, y)
    class_idx = list(mdl.classes_).index("verifiable")
    expl = shap.TreeExplainer(mdl)
    sv = shap_values_for_class(expl, X, class_idx)

    fig = plt.figure(figsize=(8, 4.5))
    shap.summary_plot(sv, X, show=False, max_display=10)
    plt.gcf().axes[0].set_xlabel('SHAP value (impact on "verifiable" class)')
    save_fig(plt.gcf(), outdir, "fig_shap_A1_beeswarm")

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(np.array(X.columns)[order], mean_abs[order], color="#1f77b4")
    ax.set_xlabel('Mean |SHAP value| (impact on "verifiable" class)')
    save_fig(fig, outdir, "fig_shap_A1_bar")

    metrics["mean_abs_shap_verifiable_class"] = dict(
        zip(X.columns, np.round(mean_abs, 5))
    )
    return metrics


# =============================================================== A2 ==========
NGRAM_CUE_PATTERNS = [
    # five-dimension mapping of n-grams (manual classification encoded as patterns)
    ("completion", r"上线|落地|部署|建成|投产|投入|应用|交付|完成|实现|推出|发布|接入|搭建|赋能|使用"),
    ("quant", r"\d|%|％|亿|万(?!科)|千|百|余|同比|增长|效率|超过"),
    ("artifact", r"平台|系统|产品|模型|工具|软件|方案|终端|机器人|助手|智能体|引擎|版本|算法|应用程序|App|GPT|大模型"),
    ("partner", r"客户|合作|伙伴|签约|用户|厂商|供应商|华为|腾讯|百度|阿里"),
    ("current", r"报告期|期内|本期|年度|202\d|年初|年内|目前|当前|本年"),
]
CUE_ZH = {
    "completion": "实施动词", "quant": "可衡量推广", "artifact": "产品工具",
    "partner": "合作伙伴", "current": "时间锚定", "other": "其他/无对应",
}


def map_ngram_to_cues(ng):
    hits = [name for name, pat in NGRAM_CUE_PATTERNS if re.search(pat, ng)]
    return "+".join(hits) if hits else "other"


def analysis_a2(seg, outdir):
    print("=" * 70, "\nA2: text-only re-discovery (char n-gram TF-IDF, segment level)\n", "=" * 70)
    texts = seg["segment_text"].astype(str).values
    y = seg["seg_class"].values

    def mk_vec():
        return TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4), min_df=20, max_features=3000
        )

    def mk():
        return lgb.LGBMClassifier(
            objective="multiclass", n_estimators=400, learning_rate=0.1,
            num_leaves=63, min_child_samples=20, colsample_bytree=0.8,
            subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1,
        )

    # fold-wise vectorizer fitting (no vocabulary leakage into CV metrics)
    def X_fn(tr, te):
        vec = mk_vec()
        return vec.fit_transform(texts[tr]), vec.transform(texts[te])

    metrics = cv_classification(mk, X_fn, y)
    print("A2 CV:", {k: round(v, 4) for k, v in metrics.items() if k != "per_fold"})

    # full-data model for SHAP
    vec = mk_vec()
    Xfull = vec.fit_transform(texts)
    feat_names = vec.get_feature_names_out()
    mdl = mk().fit(Xfull, y)
    class_idx = list(mdl.classes_).index("verifiable")

    # SHAP on a stratified subsample (memory: n_sub x 3000 x 3 classes)
    rng = np.random.RandomState(SEED)
    idx = np.concatenate(
        [
            rng.choice(np.where(y == cl)[0],
                       size=min(2000, (y == cl).sum()), replace=False)
            for cl in np.unique(y)
        ]
    )
    X_sub = np.asarray(Xfull[idx].todense(), dtype=np.float32)
    expl = shap.TreeExplainer(mdl)
    sv = shap_values_for_class(expl, X_sub, class_idx)

    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:25]
    rows = []
    for j in top_idx:
        present = X_sub[:, j] > 0
        direction = float(sv[present, j].mean()) if present.any() else 0.0
        ng = feat_names[j]
        rows.append(
            {
                "ngram": ng,
                "mean_abs_shap": float(mean_abs[j]),
                "mean_shap_when_present": direction,
                "direction": "-> verifiable" if direction > 0 else "-> not verifiable",
                "cue_dimension": map_ngram_to_cues(ng),
                "cue_dimension_zh": "+".join(
                    CUE_ZH[c] for c in map_ngram_to_cues(ng).split("+")
                ),
                "present_share_in_sample": float(present.mean()),
            }
        )
    top_df = pd.DataFrame(rows)
    top_df.to_csv(f"{outdir}/table_A2_top25_ngrams.csv", index=False, encoding="utf-8-sig")
    print(top_df[["ngram", "mean_abs_shap", "direction", "cue_dimension"]].to_string())

    # horizontal bar: top 25 n-grams, color by direction
    fig, ax = plt.subplots(figsize=(8.5, 8))
    dd = top_df.iloc[::-1]
    colors = ["#d62728" if d <= 0 else "#1f77b4" for d in dd["mean_shap_when_present"]]
    ylabels = [f'"{r.ngram}"  [{r.cue_dimension}]' for r in dd.itertuples()]
    ax.barh(np.arange(len(dd)), dd["mean_abs_shap"], color=colors)
    ax.set_yticks(np.arange(len(dd)))
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel('Mean |SHAP value| (impact on "verifiable" class)')
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#1f77b4", label="pushes toward verifiable when present"),
            Patch(color="#d62728", label="pushes away from verifiable when present"),
        ],
        loc="lower right", fontsize=9,
    )
    save_fig(fig, outdir, "fig_shap_A2_topngram")

    # beeswarm (top 20)
    sub_df = pd.DataFrame(X_sub, columns=feat_names)
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(sv, sub_df, show=False, max_display=20)
    plt.gcf().axes[0].set_xlabel('SHAP value (impact on "verifiable" class)')
    save_fig(plt.gcf(), outdir, "fig_shap_A2_beeswarm")

    # convergent-validity tally
    tally = top_df["cue_dimension"].apply(lambda s: "other" if s == "other" else "mapped")
    metrics["top25_mapped_to_five_dims"] = int((tally == "mapped").sum())
    metrics["n_shap_subsample"] = int(len(idx))
    return metrics, top_df


# =============================================================== B ===========
def build_b_dataset(event_panel_path, seg, feat):
    ev = pd.read_csv(event_panel_path)
    e = ev[ev.first_type.notna()].copy()
    piv = (
        e[e.event_time.isin([-1, 1])]
        .pivot_table(index="Firm_ID", columns="event_time", values="TobinQ")
        .rename(columns={-1.0: "Q_m1", 1.0: "Q_p1"})
    )
    ctl = e[e.event_time == 0].set_index("Firm_ID")[
        ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount", "first_ai_year", "first_type"]
    ]
    df = piv.join(ctl, how="inner").dropna(subset=["Q_m1", "Q_p1"])
    df = df.dropna(subset=["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount"])  # complete cases
    df = df.reset_index()
    df["fid"] = df.Firm_ID.astype(int).astype(str).str.zfill(6)

    # firm-year text features measured in the first-disclosure year
    tf = feat.rename(columns={"Firm_ID": "fid", "Year": "first_ai_year"})
    df = df.merge(
        tf[
            ["fid", "first_ai_year", "N_Segments", "N_Verifiable",
             "Verifiable_Share", "Mean_Verifiable_Score"]
        ],
        on=["fid", "first_ai_year"], how="left",
    )
    # five-cue shares aggregated to the firm-year (disclosure year)
    cs = (
        seg.groupby(["Firm_ID", "Year"])[CUES].mean()
        .add_prefix("share_").reset_index()
        .rename(columns={"Firm_ID": "fid", "Year": "first_ai_year"})
    )
    df = df.merge(cs, on=["fid", "first_ai_year"], how="left")
    df["type_verifiable"] = (df.first_type == "verifiable").astype(int)
    df["type_soft_substantive"] = (df.first_type == "soft_substantive").astype(int)
    return df


def cv_r2(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in kf.split(X):
        mdl = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=7,
            min_child_samples=15, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.9, reg_lambda=1.0, random_state=SEED, verbose=-1,
        )
        mdl.fit(X.iloc[tr], y[tr])
        scores.append(r2_score(y[te], mdl.predict(X.iloc[te])))
    return scores


def analysis_b(event_panel_path, seg, feat, outdir):
    print("=" * 70, "\nB: incremental contribution to market reaction (firm-year)\n", "=" * 70)
    df = build_b_dataset(event_panel_path, seg, feat)
    n = len(df)
    print(f"B sample: n={n} first-disclosure firms with Q(t-1) and Q(t+1) and complete controls")

    controls = ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount", "Q_m1", "first_ai_year"]
    textf = [
        "N_Segments", "N_Verifiable", "Verifiable_Share", "Mean_Verifiable_Score",
        "type_verifiable", "type_soft_substantive",
        "share_has_completion", "share_has_quant", "share_has_artifact",
        "share_has_partner", "share_has_current",
    ]
    pretty = {
        "Size": "Firm size (ln assets)", "Lev": "Leverage", "SoE": "State-owned",
        "IndepDir": "Independent director ratio", "MDA_CharCount": "MD&A length (chars)",
        "Q_m1": "Baseline Tobin's Q (t-1)", "first_ai_year": "First disclosure year",
        "N_Segments": "N GenAI segments", "N_Verifiable": "N verifiable segments",
        "Verifiable_Share": "Verifiable share", "Mean_Verifiable_Score": "Mean verifiability score",
        "type_verifiable": "Type: verifiable", "type_soft_substantive": "Type: soft-substantive",
        "share_has_completion": "Cue share: implementation verb",
        "share_has_quant": "Cue share: quantified rollout",
        "share_has_artifact": "Cue share: product/tool",
        "share_has_partner": "Cue share: partner/customer",
        "share_has_current": "Cue share: time anchoring",
    }

    out = {"n": n}
    for target_name, yv in [
        ("TobinQ_t+1_level", df["Q_p1"].values),
        ("TobinQ_change_t+1_minus_t-1", (df["Q_p1"] - df["Q_m1"]).values),
    ]:
        s_ctl = cv_r2(df[controls], yv)
        s_full = cv_r2(df[controls + textf], yv)
        out[target_name] = {
            "controls_only_R2_mean": float(np.mean(s_ctl)),
            "controls_only_R2_sd": float(np.std(s_ctl)),
            "controls_plus_text_R2_mean": float(np.mean(s_full)),
            "controls_plus_text_R2_sd": float(np.std(s_full)),
            "delta_R2": float(np.mean(s_full) - np.mean(s_ctl)),
            "per_fold_controls": s_ctl, "per_fold_full": s_full,
        }
        print(target_name, {k: round(v, 4) for k, v in out[target_name].items()
                            if not k.startswith("per_fold")})

    # SHAP for the primary (level) full model
    Xf = df[controls + textf].rename(columns=pretty)
    mdl = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=15,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    ).fit(Xf, df["Q_p1"].values)
    sv = shap.TreeExplainer(mdl).shap_values(Xf)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(np.asarray(sv), Xf, show=False, max_display=15)
    plt.gcf().axes[0].set_xlabel("SHAP value (impact on Tobin's Q at t+1)")
    save_fig(plt.gcf(), outdir, "fig_shap_B_beeswarm")

    out["primary_outcome"] = (
        "TobinQ(t+1) level with TobinQ(t-1) as baseline feature (more stable CV R^2 and "
        "more interpretable SHAP than the delta specification; delta variant reported above)"
    )
    out["mean_abs_shap_top10"] = dict(
        sorted(
            zip(Xf.columns, np.abs(np.asarray(sv)).mean(axis=0).round(5)),
            key=lambda kv: -kv[1],
        )[:10]
    )
    return out


# ============================================================== main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-pkl", default="data/paper1_verifiability_features.pkl")
    ap.add_argument("--event-panel", default="data/paper1_v11_strict_event_panel_with_extra_controls.csv")
    ap.add_argument("--cohort", default=None, help="optional cohort_enriched csv for a consistency check")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    d = pd.read_pickle(args.features_pkl)
    seg, feat = d["segments"], d["features"]

    rules, seg = step0_rules(seg, feat, args.outdir)
    a1 = analysis_a1(seg, args.outdir)
    a2, top_df = analysis_a2(seg, args.outdir)
    b = analysis_b(args.event_panel, seg, feat, args.outdir)

    cohort_check = None
    if args.cohort:
        co = pd.read_csv(args.cohort)
        cohort_check = {
            "n_firms": int(len(co)),
            "first_year_counts": co["first"].value_counts().sort_index().to_dict(),
            "vtype_counts": co["vtype"].value_counts().to_dict(),
        }

    # performance summary CSV
    perf = pd.DataFrame(
        [
            {
                "analysis": "A1 rule cues + length (transparency decomposition)",
                "task": "3-class segment type (verifiable/soft_substantive/symbolic)",
                "model": "LightGBM multiclass", "n": len(seg), "n_features": 6,
                "cv": "5-fold stratified",
                "accuracy": f"{a1['accuracy_mean']:.4f} ± {a1['accuracy_sd']:.4f}",
                "macro_F1": f"{a1['macro_f1_mean']:.4f} ± {a1['macro_f1_sd']:.4f}",
                "AUC_ovr": f"{a1['auc_ovr_mean']:.4f} ± {a1['auc_ovr_sd']:.4f}",
                "R2": "", "delta_R2": "",
            },
            {
                "analysis": "A2 char n-gram TF-IDF only (independent re-discovery)",
                "task": "3-class segment type (verifiable/soft_substantive/symbolic)",
                "model": "LightGBM multiclass", "n": len(seg), "n_features": 3000,
                "cv": "5-fold stratified (fold-wise TF-IDF)",
                "accuracy": f"{a2['accuracy_mean']:.4f} ± {a2['accuracy_sd']:.4f}",
                "macro_F1": f"{a2['macro_f1_mean']:.4f} ± {a2['macro_f1_sd']:.4f}",
                "AUC_ovr": f"{a2['auc_ovr_mean']:.4f} ± {a2['auc_ovr_sd']:.4f}",
                "R2": "", "delta_R2": "",
            },
            {
                "analysis": "B controls only", "task": "TobinQ(t+1) level",
                "model": "LightGBM regression", "n": b["n"], "n_features": 7,
                "cv": "5-fold",
                "accuracy": "", "macro_F1": "", "AUC_ovr": "",
                "R2": f"{b['TobinQ_t+1_level']['controls_only_R2_mean']:.4f} ± {b['TobinQ_t+1_level']['controls_only_R2_sd']:.4f}",
                "delta_R2": "",
            },
            {
                "analysis": "B controls + text verifiability features", "task": "TobinQ(t+1) level",
                "model": "LightGBM regression", "n": b["n"], "n_features": 18,
                "cv": "5-fold",
                "accuracy": "", "macro_F1": "", "AUC_ovr": "",
                "R2": f"{b['TobinQ_t+1_level']['controls_plus_text_R2_mean']:.4f} ± {b['TobinQ_t+1_level']['controls_plus_text_R2_sd']:.4f}",
                "delta_R2": f"{b['TobinQ_t+1_level']['delta_R2']:.4f}",
            },
        ]
    )
    perf.to_csv(f"{args.outdir}/table_model_performance.csv", index=False, encoding="utf-8-sig")

    summary = {
        "step0_measurement_rules": rules,
        "A1_rule_cues": a1,
        "A2_text_only": a2,
        "B_market_reaction": b,
        "cohort_consistency_check": cohort_check,
        "config": {
            "seed": SEED,
            "lightgbm": lgb.__version__, "shap": shap.__version__,
            "A1_model": "LGBMClassifier(multiclass, n_estimators=300, lr=0.1, num_leaves=31)",
            "A2_tfidf": "char ngram (2,4), min_df=20, max_features=3000, fold-wise fitting in CV",
            "A2_model": "LGBMClassifier(multiclass, n_estimators=400, lr=0.1, num_leaves=63, colsample=0.8, subsample=0.8)",
            "B_model": "LGBMRegressor(n_estimators=400, lr=0.03, num_leaves=7, min_child_samples=15, reg_lambda=1)",
        },
    }
    with open(f"{args.outdir}/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_py(summary), f, ensure_ascii=False, indent=2)
    print("\nAll outputs written to", args.outdir)


if __name__ == "__main__":
    main()

"""
72_endogeneity_dynamic_event_study.py

Endogeneity remediation #1 (audit item 1.1 -- pre-trend / parallel trends).

Mirrors the STRICT main-dictionary event pipeline of
19_run_paper1_strict_dictionary_event.py, but:
  * extends the event window from {-2,0,+1} to the FULL {-3,-2,-1,0,+1,+2,+3}
    with t-1 as the omitted (normalized) base period;
  * runs both the firm+year FE main spec and the stricter
    firm + industry-by-year FE spec;
  * computes the formal ACCELERATION test (post-slope minus pre-slope) per
    disclosure type -- a significant t-2 LEVEL is not fatal if the
    trajectory bends at disclosure;
  * traces the VERIFIABLE - SYMBOLIC gap across event time -- the core H3
    contrast, which differences out any pre-trend common to all three
    treated groups;
  * runs the joint pre-trend test (pre-period coefficients = 0).

Output:
  results/paper1_market_pricing_v6/paper1_endogeneity_dynamic_event.xlsx
  + console summary with the numbers to paste into Section 4.2 / Table 6.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import re

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(r"E:\Supply - SHAP")
PANEL_CACHE = ROOT / "results" / "paper1_market_pricing" / "paper1_fullsample_panel.pkl"
V5_CACHE = ROOT / "results" / "paper1_market_pricing_v5" / "paper1_verifiability_features.pkl"
OUT = ROOT / "results" / "paper1_market_pricing_v6"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_XLSX = OUT / "paper1_endogeneity_dynamic_event.xlsx"

CONTROL_TERMS = ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount"]

# Full dynamic window. t-1 ("m1") is the omitted base (NOT in EVENT_TERMS).
EVENT_OFFSETS = [("m3", -3), ("m2", -2), ("0", 0), ("p1", 1), ("p2", 2), ("p3", 3)]
PREFIXES = [("ver", "verifiable"), ("soft", "soft_substantive"), ("sym", "symbolic")]
EVENT_TERMS = [f"{p}_evt_{n}" for p, _ in PREFIXES for n, _ in EVENT_OFFSETS]

# Strict main-dictionary anchors (identical to script 19).
STRICT_ANCHORS = [
    "生成式人工智能", "生成式 AI", "生成式AI", "GENAI", "GENERATIVE AI", "AIGC",
    "大语言模型", "多模态大模型", "CHATGPT", "COPILOT", "CLAUDE", "GEMINI", "PALM",
    "LLAMA", "MISTRAL", "GROK", "SORA", "MIDJOURNEY", "STABLE DIFFUSION", "DALL-E",
    "RUNWAY", "PIKA", "DEEPSEEK", "KIMI", "文心一言", "文心大模型", "ERNIE",
    "通义千问", "通义万相", "QWEN", "混元大模型", "腾讯混元", "盘古大模型", "华为盘古",
    "星火大模型", "讯飞星火", "智谱清言", "CHATGLM", "BAICHUAN", "MINIMAX",
    "ABAB 大模型", "ABAB大模型", "YI 大模型", "YI大模型", "天工大模型", "商量大模型",
    "日日新大模型", "蓝心大模型", "言犀大模型", "VIDU", "生数大模型",
]


def normal_pvalue(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))


def linear_contrast(params, cov, weights: dict) -> tuple[float, float, float]:
    keys = [k for k in weights if k in params.index and k in cov.index and k in cov.columns]
    if not keys:
        return np.nan, np.nan, np.nan
    coef = sum(weights[k] * float(params[k]) for k in keys)
    var = 0.0
    for i in keys:
        for j in keys:
            var += weights[i] * weights[j] * float(cov.loc[i, j])
    if pd.isna(var) or var <= 0:
        return coef, np.nan, np.nan
    se = sqrt(var)
    return coef, se, normal_pvalue(coef / se)


def compile_anchor_pattern(anchors: list[str]) -> re.Pattern:
    pieces = []
    for term in anchors:
        esc = re.escape(term).replace(r"\ ", r"\s*")
        pieces.append(esc)
    pieces = sorted(set(pieces), key=len, reverse=True)
    return re.compile("|".join(pieces), flags=re.IGNORECASE)


def build_strict_features(segments: pd.DataFrame) -> pd.DataFrame:
    pat = compile_anchor_pattern(STRICT_ANCHORS)
    w = segments.copy()
    w["segment_text"] = w["segment_text"].fillna("").astype(str)
    w["hit"] = w["segment_text"].apply(lambda x: int(bool(pat.search(x))))
    s = w.loc[w["hit"] == 1].copy()
    g = s.groupby(["Firm_ID", "Year"], as_index=False).agg(
        N_Strict_Segments=("segment_text", "size"),
        N_Strict_Substantive=("label", lambda x: int((x == "substantive").sum())),
        N_Strict_Verifiable=("verifiable_segment", "sum"),
    )
    g["Has_V"] = (g["N_Strict_Verifiable"] > 0).astype(int)
    g["Has_Soft"] = ((g["N_Strict_Substantive"] > 0) & (g["Has_V"] == 0)).astype(int)
    g["Type_V6"] = np.where(
        g["Has_V"] == 1, "verifiable",
        np.where(g["Has_Soft"] == 1, "soft_substantive", "symbolic"),
    )
    return g


def build_event_panel(panel: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = panel.sort_values(["Firm_ID", "Year"]).copy()
    ay = feats[["Firm_ID", "Year", "Type_V6"]].drop_duplicates()
    first = ay.sort_values(["Firm_ID", "Year"]).groupby("Firm_ID").head(1).copy()
    first = first.rename(columns={"Year": "first_ai_year", "Type_V6": "first_type_v6"})
    first = first.loc[first["first_ai_year"].between(2017, 2023)].copy()

    ev = ordered.merge(first, on="Firm_ID", how="left")
    ev["never_adopter"] = ev["first_ai_year"].isna().astype(int)
    ev["event_time"] = ev["Year"] - ev["first_ai_year"]
    # Full window: never-adopters OR event_time in [-3, +3]
    ev = ev.loc[(ev["never_adopter"] == 1) | (ev["event_time"].between(-3, 3))].copy()

    for prefix, cohort in PREFIXES:
        is_c = (ev["first_type_v6"] == cohort).astype(int)
        for name, val in EVENT_OFFSETS:
            ev[f"{prefix}_evt_{name}"] = ((is_c == 1) & (ev["event_time"] == val)).astype(float)
    return ev, first


def add_industry_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IndYear"] = df["Industry"].astype(str) + "_" + df["Year"].astype(str)
    return df


def fit_model(df: pd.DataFrame, controls: bool, ind_year_fe: bool, label: str):
    keep = ["Firm_ID", "Year", "TobinQ"] + EVENT_TERMS + CONTROL_TERMS
    if ind_year_fe:
        keep = keep + ["IndYear"]
    work = df[keep].dropna(subset=["TobinQ"]).copy()
    exog = EVENT_TERMS.copy()
    if controls:
        work = work.dropna(subset=CONTROL_TERMS).copy()
        exog = exog + CONTROL_TERMS

    if ind_year_fe:
        sample = work.set_index(["Firm_ID", "Year"]).sort_index()
        model = PanelOLS(
            sample["TobinQ"], sample[exog],
            entity_effects=True, other_effects=sample["IndYear"],
            drop_absorbed=True,
        )
    else:
        sample = work.set_index(["Firm_ID", "Year"]).sort_index()
        model = PanelOLS(
            sample["TobinQ"], sample[exog],
            entity_effects=True, time_effects=True, drop_absorbed=True,
        )
    res = model.fit(cov_type="clustered", cluster_entity=True)
    rows = [{
        "model": label, "term": t,
        "coef": res.params.get(t, np.nan),
        "std_err": res.std_errors.get(t, np.nan),
        "p_value": res.pvalues.get(t, np.nan),
        "n": int(res.nobs),
        "firms": int(sample.index.get_level_values(0).nunique()),
    } for t in exog]
    return pd.DataFrame(rows), res


def diagnostics(res, label: str) -> pd.DataFrame:
    rows = []
    # 1. Per-type dynamic path (already in coef table) + acceleration test
    for pfx, cohort in PREFIXES:
        # Pre-trend joint-ish: report t-3 and t-2 individually vs base t-1=0
        for name, _ in EVENT_OFFSETS:
            term = f"{pfx}_evt_{name}"
            c, se, p = linear_contrast(res.params, res.cov, {term: 1.0})
            rows.append({"model": label, "stat": f"{cohort} @ {name}", "coef": c, "se": se, "p": p})
        # Acceleration = (post-slope) - (pre-slope)
        #   post-slope ~ beta_p1 - beta_0 ; pre-slope ~ beta_m2 - beta_m3
        c, se, p = linear_contrast(
            res.params, res.cov,
            {f"{pfx}_evt_p1": 1.0, f"{pfx}_evt_0": -1.0,
             f"{pfx}_evt_m2": -1.0, f"{pfx}_evt_m3": 1.0},
        )
        rows.append({"model": label, "stat": f"{cohort}: ACCELERATION (post-slope - pre-slope)",
                     "coef": c, "se": se, "p": p})
    # 2. Verifiable - Symbolic gap path (the core H3 contrast)
    for name, _ in EVENT_OFFSETS:
        c, se, p = linear_contrast(
            res.params, res.cov,
            {f"ver_evt_{name}": 1.0, f"sym_evt_{name}": -1.0},
        )
        rows.append({"model": label, "stat": f"GAP ver-sym @ {name}", "coef": c, "se": se, "p": p})
    # 3. Verifiable - Soft gap at +1
    c, se, p = linear_contrast(
        res.params, res.cov, {"ver_evt_p1": 1.0, "soft_evt_p1": -1.0})
    rows.append({"model": label, "stat": "GAP ver-soft @ p1", "coef": c, "se": se, "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_pickle(PANEL_CACHE)
    obj = pd.read_pickle(V5_CACHE)
    segments = obj["segments"].copy()

    feats = build_strict_features(segments)
    ev, first = build_event_panel(panel, feats)
    ev = add_industry_year(ev)

    print("Strict first-adopter types (2017-2023):",
          first["first_type_v6"].value_counts(dropna=False).to_dict(), flush=True)

    m_main, r_main = fit_model(ev, controls=False, ind_year_fe=False, label="main_firm_year")
    m_ctrl, r_ctrl = fit_model(ev, controls=True, ind_year_fe=False, label="ctrl_firm_year")
    m_iy, r_iy = fit_model(ev, controls=True, ind_year_fe=True, label="ctrl_firm_indYear")

    d_main = diagnostics(r_main, "main_firm_year")
    d_ctrl = diagnostics(r_ctrl, "ctrl_firm_year")
    d_iy = diagnostics(r_iy, "ctrl_firm_indYear")

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as w:
        m_main.to_excel(w, sheet_name="coef_main", index=False)
        m_ctrl.to_excel(w, sheet_name="coef_ctrl", index=False)
        m_iy.to_excel(w, sheet_name="coef_indYear", index=False)
        d_main.to_excel(w, sheet_name="diag_main", index=False)
        d_ctrl.to_excel(w, sheet_name="diag_ctrl", index=False)
        d_iy.to_excel(w, sheet_name="diag_indYear", index=False)

    def show(tag, dd):
        print(f"\n================ {tag} ================", flush=True)
        for _, r in dd.iterrows():
            if any(s in r["stat"] for s in ["ACCELERATION", "GAP ver-sym", "GAP ver-soft"]) \
               or "@ m3" in r["stat"] or "@ m2" in r["stat"] or "@ p1" in r["stat"]:
                cf = r["coef"]; se = r["se"]; p = r["p"]
                star = "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""
                print(f"  {r['stat']:<52} b={cf:+.4f}  se={se:.4f}  p={p:.4f}{star}", flush=True)

    show("MAIN: firm+year FE (no controls)", d_main)
    show("CTRL: firm+year FE (+controls)", d_ctrl)
    show("STRICT: firm + industry*year FE (+controls)", d_iy)
    print(f"\nSaved -> {RESULTS_XLSX}", flush=True)


if __name__ == "__main__":
    main()

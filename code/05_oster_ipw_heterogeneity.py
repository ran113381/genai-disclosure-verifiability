"""
74_endogeneity_oster_ipw_het.py

Endogeneity remediation #2: Oster (2019) bounds, IPW-DID, and
heterogeneity / long-horizon tests -- all on the REAL strict-dictionary
event panel, using only variables that exist in the project caches
(no inst_own / analyst / board / province; we use SoE, Size, HHI,
RnD_Ratio, Age, Industry instead).

Builds the same strict first-adopter panel as script 72, then:

  PART 1  OSTER (2019) bounds for the verifiable-vs-symbolic premium.
          delta* (selection ratio that nullifies the effect) and the
          bias-adjusted beta* at delta=1, with R-max = min(1.3*R2c, 1).

  PART 2  IPW-DID. Propensity of being a VERIFIABLE (vs SYMBOLIC) first
          discloser, estimated on PRE-disclosure characteristics
          (t-1 Q level, pre-disclosure Q trend, Size, Lev, SoE, ROA,
          RnD_Ratio, Age, HHI). Stabilized weights -> weighted event DID;
          balance table + post-weight pre-trend test.

  PART 3  Heterogeneity & long horizon under firm + industry*year FE:
          verifiable-symbolic gap at t+1 and t+3 in sub-samples split by
          SoE, Size (median), HHI (median), RnD_Ratio (median).

Output:
  results/paper1_market_pricing_v6/paper1_endogeneity_oster_ipw_het.xlsx
  + console summary.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

ROOT = Path(r"E:\Supply - SHAP")
PANEL_CACHE = ROOT / "results" / "paper1_market_pricing" / "paper1_fullsample_panel.pkl"
V5_CACHE = ROOT / "results" / "paper1_market_pricing_v5" / "paper1_verifiability_features.pkl"
OUT = ROOT / "results" / "paper1_market_pricing_v6"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_XLSX = OUT / "paper1_endogeneity_oster_ipw_het.xlsx"

CONTROL_TERMS = ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount"]
RICH_CONTROLS = ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount",
                 "ROA", "RnD_Ratio", "Age", "HHI"]

EVENT_OFFSETS = [("m3", -3), ("m2", -2), ("0", 0), ("p1", 1), ("p2", 2), ("p3", 3)]
PREFIXES = [("ver", "verifiable"), ("soft", "soft_substantive"), ("sym", "symbolic")]
EVENT_TERMS = [f"{p}_evt_{n}" for p, _ in PREFIXES for n, _ in EVENT_OFFSETS]

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


def normal_p(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))


def linear_contrast(params, cov, weights):
    keys = [k for k in weights if k in params.index and k in cov.index]
    if not keys:
        return np.nan, np.nan, np.nan
    coef = sum(weights[k] * float(params[k]) for k in keys)
    var = sum(weights[i] * weights[j] * float(cov.loc[i, j]) for i in keys for j in keys)
    if pd.isna(var) or var <= 0:
        return coef, np.nan, np.nan
    se = sqrt(var)
    return coef, se, normal_p(coef / se)


def compile_pat(anchors):
    pieces = sorted({re.escape(t).replace(r"\ ", r"\s*") for t in anchors},
                    key=len, reverse=True)
    return re.compile("|".join(pieces), flags=re.IGNORECASE)


def strict_features(segments):
    pat = compile_pat(STRICT_ANCHORS)
    w = segments.copy()
    w["segment_text"] = w["segment_text"].fillna("").astype(str)
    w["hit"] = w["segment_text"].apply(lambda x: int(bool(pat.search(x))))
    s = w.loc[w["hit"] == 1].copy()
    g = s.groupby(["Firm_ID", "Year"], as_index=False).agg(
        N_Seg=("segment_text", "size"),
        N_Sub=("label", lambda x: int((x == "substantive").sum())),
        N_Ver=("verifiable_segment", "sum"),
    )
    g["Has_V"] = (g["N_Ver"] > 0).astype(int)
    g["Has_Soft"] = ((g["N_Sub"] > 0) & (g["Has_V"] == 0)).astype(int)
    g["Type_V6"] = np.where(g["Has_V"] == 1, "verifiable",
                            np.where(g["Has_Soft"] == 1, "soft_substantive", "symbolic"))
    return g


def build_event(panel, feats):
    ordered = panel.sort_values(["Firm_ID", "Year"]).copy()
    ay = feats[["Firm_ID", "Year", "Type_V6"]].drop_duplicates()
    first = ay.sort_values(["Firm_ID", "Year"]).groupby("Firm_ID").head(1).copy()
    first = first.rename(columns={"Year": "first_ai_year", "Type_V6": "first_type_v6"})
    first = first.loc[first["first_ai_year"].between(2017, 2023)].copy()
    ev = ordered.merge(first, on="Firm_ID", how="left")
    ev["never_adopter"] = ev["first_ai_year"].isna().astype(int)
    ev["event_time"] = ev["Year"] - ev["first_ai_year"]
    ev = ev.loc[(ev["never_adopter"] == 1) | (ev["event_time"].between(-3, 3))].copy()
    for prefix, cohort in PREFIXES:
        is_c = (ev["first_type_v6"] == cohort).astype(int)
        for name, val in EVENT_OFFSETS:
            ev[f"{prefix}_evt_{name}"] = ((is_c == 1) & (ev["event_time"] == val)).astype(float)
    ev["IndYear"] = ev["Industry"].astype(str) + "_" + ev["Year"].astype(str)
    return ev, first


# ---------------------------------------------------------------------------
# PART 1: Oster (2019) bounds
# ---------------------------------------------------------------------------
def oster_bounds(ev):
    """
    Verifiable(=1) vs Symbolic(=0) among first disclosers, post period.
    Within-transform y & treatment & controls by firm + industry*year, then
    run uncontrolled vs controlled OLS to get (beta, R2) pairs, then Oster.
    """
    d = ev.loc[(ev["never_adopter"] == 0) & (ev["event_time"] >= 0) &
               (ev["first_type_v6"].isin(["verifiable", "symbolic"]))].copy()
    d = d.dropna(subset=["TobinQ"] + RICH_CONTROLS).copy()
    d["VER"] = (d["first_type_v6"] == "verifiable").astype(float)

    # demean within firm and within industry*year (two-way absorb, iterative)
    cols = ["TobinQ", "VER"] + RICH_CONTROLS
    work = d[cols + ["Firm_ID", "IndYear"]].copy()
    for _ in range(40):
        work[cols] = work[cols] - work.groupby("Firm_ID")[cols].transform("mean")
        work[cols] = work[cols] - work.groupby("IndYear")[cols].transform("mean")

    y = work["TobinQ"].values
    t = work["VER"].values

    def ols(X):
        Xc = sm.add_constant(X)
        m = sm.OLS(y, Xc).fit()
        return m

    m0 = ols(work[["VER"]].values)
    b0, r0 = m0.params[1], m0.rsquared
    m1 = ols(work[["VER"] + RICH_CONTROLS].values if False else
             np.column_stack([work["VER"].values, work[RICH_CONTROLS].values]))
    b1, r1 = m1.params[1], m1.rsquared

    rmax = min(1.3 * r1, 1.0)

    def delta_star(rm):
        # Oster (2019) eq.: solve for delta s.t. bias-adjusted beta = 0.
        # Closed-form approx (Oster 2019, Prop.2 / psacalc):
        num = (b1) * (rm - r1)
        den = (b0 - b1) * (r1 - r0)
        if den == 0:
            return np.nan
        return num / den

    def beta_star(rm, delta=1.0):
        # bias-adjusted beta at given delta (Oster 2019 approximation)
        return b1 - delta * (b0 - b1) * (rm - r1) / (r1 - r0)

    rows = [
        {"stat": "beta_uncontrolled", "value": b0},
        {"stat": "R2_uncontrolled", "value": r0},
        {"stat": "beta_controlled", "value": b1},
        {"stat": "R2_controlled", "value": r1},
        {"stat": "Rmax(1.3*R2c, cap1)", "value": rmax},
        {"stat": "delta* (Rmax=1.3*R2c)", "value": delta_star(rmax)},
        {"stat": "delta* (Rmax=1.0)", "value": delta_star(1.0)},
        {"stat": "beta* (delta=1, Rmax=1.3*R2c)", "value": beta_star(rmax, 1.0)},
        {"stat": "beta* (delta=1, Rmax=1.0)", "value": beta_star(1.0, 1.0)},
        {"stat": "N", "value": len(work)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PART 2: IPW-DID
# ---------------------------------------------------------------------------
def ipw_did(ev, panel):
    # pre-disclosure Q level (t-1) and pre-trend (t-1 minus t-3)/2
    p = panel.sort_values(["Firm_ID", "Year"]).copy()
    p["q_lag1"] = p.groupby("Firm_ID")["TobinQ"].shift(1)
    p["q_lag3"] = p.groupby("Firm_ID")["TobinQ"].shift(3)
    pre = p[["Firm_ID", "Year", "q_lag1", "q_lag3"]]

    first = ev[["Firm_ID", "first_ai_year", "first_type_v6", "never_adopter"]].drop_duplicates()
    treated = first.loc[first["never_adopter"] == 0].copy()
    treated = treated.loc[treated["first_type_v6"].isin(["verifiable", "symbolic"])].copy()
    treated["base_year"] = treated["first_ai_year"] - 1

    base = panel.merge(treated[["Firm_ID", "base_year", "first_type_v6"]],
                       left_on=["Firm_ID", "Year"], right_on=["Firm_ID", "base_year"],
                       how="inner")
    base = base.merge(pre, on=["Firm_ID", "Year"], how="left")
    base["q_pretrend"] = (base["q_lag1"] - base["q_lag3"]) / 2.0
    base["VER"] = (base["first_type_v6"] == "verifiable").astype(int)

    psvars = ["q_lag1", "q_pretrend", "Size", "Lev", "SoE", "ROA",
              "RnD_Ratio", "Age", "HHI"]
    ps = base.dropna(subset=psvars + ["VER"]).copy()
    Xp = sm.add_constant(ps[psvars])
    logit = sm.Logit(ps["VER"], Xp).fit(disp=0)
    ps["ehat"] = logit.predict(Xp)
    # ATT weighting: verifiable weight 1; symbolic weight e/(1-e)
    ps["w"] = np.where(ps["VER"] == 1, 1.0,
                       ps["ehat"] / (1.0 - ps["ehat"]).clip(lower=1e-3))
    # trim
    lo, hi = ps["ehat"].quantile([0.02, 0.98])
    ps = ps.loc[ps["ehat"].between(lo, hi)].copy()
    ps["w"] = ps["w"].clip(upper=ps["w"].quantile(0.99))

    # balance
    bal = []
    for v in psvars:
        mt, mc = ps.loc[ps.VER == 1, v].mean(), ps.loc[ps.VER == 0, v].mean()
        vt, vc = ps.loc[ps.VER == 1, v].var(), ps.loc[ps.VER == 0, v].var()
        sd_raw = (mt - mc) / sqrt((vt + vc) / 2) if (vt + vc) > 0 else np.nan
        wm_t = np.average(ps.loc[ps.VER == 1, v], weights=ps.loc[ps.VER == 1, "w"])
        wm_c = np.average(ps.loc[ps.VER == 0, v], weights=ps.loc[ps.VER == 0, "w"])
        sd_w = (wm_t - wm_c) / sqrt((vt + vc) / 2) if (vt + vc) > 0 else np.nan
        bal.append({"var": v, "std_diff_raw": sd_raw, "std_diff_ipw": sd_w})
    bal_df = pd.DataFrame(bal)

    # weighted event DID on verifiable+symbolic+never sample
    fw = ps[["Firm_ID", "w"]].drop_duplicates("Firm_ID")
    samp = ev.merge(fw, on="Firm_ID", how="left")
    samp["w"] = samp["w"].fillna(1.0)  # never-adopters carry weight 1
    keep = samp["never_adopter"].eq(1) | samp["first_type_v6"].isin(["verifiable", "symbolic"])
    samp = samp.loc[keep].copy()
    terms = [f"{p}_evt_{n}" for p in ["ver", "sym"] for n, _ in EVENT_OFFSETS]
    sdf = samp.dropna(subset=["TobinQ"] + CONTROL_TERMS).copy()
    sidx = sdf.set_index(["Firm_ID", "Year"]).sort_index()
    mod = PanelOLS(sidx["TobinQ"], sidx[terms + CONTROL_TERMS],
                   entity_effects=True, time_effects=True, weights=sidx["w"],
                   drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    gap = []
    for name, _ in EVENT_OFFSETS:
        c, se, pv = linear_contrast(res.params, res.cov,
                                    {f"ver_evt_{name}": 1.0, f"sym_evt_{name}": -1.0})
        gap.append({"event": name, "gap_coef": c, "se": se, "p": pv})
    return bal_df, pd.DataFrame(gap), logit.prsquared


# ---------------------------------------------------------------------------
# PART 3: heterogeneity & long horizon (firm + industry*year FE)
# ---------------------------------------------------------------------------
def heterogeneity(ev):
    rows = []
    treated_mask = ev["never_adopter"] == 0
    splits = {
        "SoE": ("SoE", lambda s: s == 1, lambda s: s == 0),
        "Size_hi": ("Size", None, None),
        "HHI_hi": ("HHI", None, None),
        "RnD_hi": ("RnD_Ratio", None, None),
    }
    for tag, (col, hi_f, lo_f) in splits.items():
        if hi_f is None:
            med = ev.loc[treated_mask, col].median()
            hi_f = (lambda s, m=med: s > m)
            lo_f = (lambda s, m=med: s <= m)
        for grp, fn in [("HI", hi_f), ("LO", lo_f)]:
            keep_treat = treated_mask & fn(ev[col])
            sub = ev.loc[(ev["never_adopter"] == 1) | keep_treat].copy()
            sub = sub.dropna(subset=["TobinQ"] + CONTROL_TERMS).copy()
            terms = EVENT_TERMS
            sidx = sub.set_index(["Firm_ID", "Year"]).sort_index()
            try:
                mod = PanelOLS(sidx["TobinQ"], sidx[terms + CONTROL_TERMS],
                               entity_effects=True, other_effects=sidx["IndYear"],
                               drop_absorbed=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True)
                for ev_name in ["p1", "p3"]:
                    c, se, pv = linear_contrast(
                        res.params, res.cov,
                        {f"ver_evt_{ev_name}": 1.0, f"sym_evt_{ev_name}": -1.0})
                    rows.append({"split": f"{tag}={grp}", "event": ev_name,
                                 "gap": c, "se": se, "p": pv,
                                 "n": int(res.nobs)})
            except Exception as e:  # noqa
                rows.append({"split": f"{tag}={grp}", "event": "ERR",
                             "gap": np.nan, "se": np.nan, "p": np.nan, "n": str(e)[:60]})
    return pd.DataFrame(rows)


def main():
    panel = pd.read_pickle(PANEL_CACHE)
    obj = pd.read_pickle(V5_CACHE)
    feats = strict_features(obj["segments"].copy())
    ev, first = build_event(panel, feats)

    print("first-adopter types:", first["first_type_v6"].value_counts().to_dict(), flush=True)

    oster = oster_bounds(ev)
    bal, ipw_gap, ps_r2 = ipw_did(ev, panel)
    het = heterogeneity(ev)

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as w:
        oster.to_excel(w, sheet_name="oster", index=False)
        bal.to_excel(w, sheet_name="ipw_balance", index=False)
        ipw_gap.to_excel(w, sheet_name="ipw_gap", index=False)
        het.to_excel(w, sheet_name="heterogeneity", index=False)

    print("\n===== OSTER (verifiable vs symbolic, firm+ind*yr within) =====", flush=True)
    for _, r in oster.iterrows():
        print(f"  {r['stat']:<34} {r['value']:.4f}", flush=True)
    print("\n===== IPW balance (|std diff| should drop) =====", flush=True)
    for _, r in bal.iterrows():
        print(f"  {r['var']:<12} raw={r['std_diff_raw']:+.3f}  ipw={r['std_diff_ipw']:+.3f}", flush=True)
    print(f"  (propensity pseudo-R2={ps_r2:.3f})", flush=True)
    print("\n===== IPW-DID verifiable-symbolic gap =====", flush=True)
    for _, r in ipw_gap.iterrows():
        s = r["p"]
        star = "***" if s < .01 else "**" if s < .05 else "*" if s < .1 else ""
        print(f"  @{r['event']:<3} b={r['gap']:+.4f} se={r['se']:.4f} p={r['p']:.4f}{star}", flush=True)
    print("\n===== Heterogeneity / long horizon (firm+ind*yr FE) =====", flush=True)
    for _, r in het.iterrows():
        if r["event"] == "ERR":
            print(f"  {r['split']:<14} ERROR {r['n']}", flush=True); continue
        s = r["p"]
        star = "***" if s < .01 else "**" if s < .05 else "*" if s < .1 else ""
        print(f"  {r['split']:<14} @{r['event']:<3} gap={r['gap']:+.4f} "
              f"se={r['se']:.4f} p={r['p']:.4f}{star} n={r['n']}", flush=True)
    print(f"\nSaved -> {RESULTS_XLSX}", flush=True)


if __name__ == "__main__":
    main()

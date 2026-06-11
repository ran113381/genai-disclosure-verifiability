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

RESULTS_XLSX = OUT / "paper1_market_pricing_v6_results.xlsx"

CONTROL_TERMS = ["Size", "Lev", "SoE", "IndepDir", "MDA_CharCount"]
EVENT_TERMS = [
    "ver_evt_m2", "ver_evt_0", "ver_evt_p1",
    "soft_evt_m2", "soft_evt_0", "soft_evt_p1",
    "sym_evt_m2", "sym_evt_0", "sym_evt_p1",
]

# Strict main-dictionary design for the paper:
# keep only high-confidence generative-AI anchors as the main measure,
# and move the wider 103-term universe to robustness/appendix.
STRICT_ANCHORS = [
    "生成式人工智能",
    "生成式 AI",
    "生成式AI",
    "GENAI",
    "GENERATIVE AI",
    "AIGC",
    "大语言模型",
    "多模态大模型",
    "CHATGPT",
    "COPILOT",
    "CLAUDE",
    "GEMINI",
    "PALM",
    "LLAMA",
    "MISTRAL",
    "GROK",
    "SORA",
    "MIDJOURNEY",
    "STABLE DIFFUSION",
    "DALL-E",
    "RUNWAY",
    "PIKA",
    "DEEPSEEK",
    "KIMI",
    "文心一言",
    "文心大模型",
    "ERNIE",
    "通义千问",
    "通义万相",
    "QWEN",
    "混元大模型",
    "腾讯混元",
    "盘古大模型",
    "华为盘古",
    "星火大模型",
    "讯飞星火",
    "智谱清言",
    "CHATGLM",
    "BAICHUAN",
    "MINIMAX",
    "ABAB 大模型",
    "ABAB大模型",
    "YI 大模型",
    "YI大模型",
    "天工大模型",
    "商量大模型",
    "日日新大模型",
    "蓝心大模型",
    "言犀大模型",
    "VIDU",
    "生数大模型",
]

EXCLUDED_FROM_MAIN = [
    "大模型",
    "GPT",
    "OPENAI",
    "ANTHROPIC",
    "HUGGING FACE",
    "智能体",
    "AI AGENT",
    "TRANSFORMER",
    "GAN",
    "生成对抗网络",
    "VAE",
    "变分自编码器",
    "扩散模型",
    "DIFFUSION",
    "预训练模型",
    "PRETRAINED MODEL",
    "FOUNDATION MODEL",
    "PROMPT",
    "提示词",
    "COT",
    "思维链",
    "RLHF",
    "RAG",
    "检索增强生成",
    "SFT",
    "指令微调",
    "LORA",
    "MOE",
    "混合专家模型",
    "火山引擎",
    "火山方舟",
    "MODELARTS",
    "AMAZON BEDROCK",
    "VERTEX AI",
    "NVIDIA AI",
    "千帆大模型",
    "百炼大模型",
    "海螺",
    "可灵",
    "豆包",
    "智谱AI",
    "百川智能",
    "零一万物",
    "阶跃星辰",
    "面壁智能",
    "幻方量化",
    "月之暗面",
    "深度求索",
]


def load_panel() -> pd.DataFrame:
    if not PANEL_CACHE.exists():
        raise FileNotFoundError(f"Missing base panel cache: {PANEL_CACHE}")
    return pd.read_pickle(PANEL_CACHE)


def load_v5_cache() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not V5_CACHE.exists():
        raise FileNotFoundError(f"Missing v5 cache: {V5_CACHE}")
    obj = pd.read_pickle(V5_CACHE)
    return obj["features"].copy(), obj["segments"].copy()


def compile_anchor_pattern(anchors: list[str]) -> re.Pattern:
    pieces = []
    for term in anchors:
        escaped = re.escape(term)
        escaped = escaped.replace(r"\ ", r"\s*")
        pieces.append(escaped)
    pieces = sorted(set(pieces), key=len, reverse=True)
    return re.compile("|".join(pieces), flags=re.IGNORECASE)


def normal_pvalue(z_value: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z_value) / sqrt(2.0))))


def linear_contrast(params: pd.Series, cov: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float]:
    keys = [k for k in weights if k in params.index and k in cov.index and k in cov.columns]
    if not keys:
        return np.nan, np.nan
    coef = sum(weights[k] * float(params[k]) for k in keys)
    variance = 0.0
    for i in keys:
        for j in keys:
            variance += weights[i] * weights[j] * float(cov.loc[i, j])
    if pd.isna(variance) or variance <= 0:
        return coef, np.nan
    z_value = coef / sqrt(variance)
    return coef, normal_pvalue(z_value)


def build_strict_segments(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = compile_anchor_pattern(STRICT_ANCHORS)
    work = segments.copy()
    work["segment_text"] = work["segment_text"].fillna("").astype(str)
    work["strict_matches"] = work["segment_text"].apply(lambda x: pattern.findall(x))
    work["strict_hit"] = work["strict_matches"].apply(lambda x: int(len(x) > 0))
    strict = work.loc[work["strict_hit"] == 1].copy()
    strict["strict_anchor_first"] = strict["strict_matches"].apply(lambda x: x[0] if x else "")
    strict["strict_anchor_count"] = strict["strict_matches"].apply(len)

    term_rows = []
    for _, row in strict.iterrows():
        seen = []
        for term in row["strict_matches"]:
            key = str(term).strip()
            upper = re.sub(r"\s+", "", key.upper())
            if upper in seen:
                continue
            seen.append(upper)
            term_rows.append(
                {
                    "Firm_ID": row["Firm_ID"],
                    "Year": row["Year"],
                    "term_raw": key,
                    "term_norm": upper,
                }
            )
    term_df = pd.DataFrame(term_rows)
    return strict, term_df


def aggregate_strict_features(strict_segments: pd.DataFrame) -> pd.DataFrame:
    if strict_segments.empty:
        return pd.DataFrame(
            columns=[
                "Firm_ID",
                "Year",
                "N_Strict_Segments",
                "N_Strict_Substantive",
                "N_Strict_Verifiable",
                "Strict_Verifiable_Share",
                "Strict_Mean_Verifiable_Score",
                "Has_Strict_Verifiable",
                "Has_Strict_Soft_Substantive",
                "Has_Strict_Symbolic_Only",
                "Type_V6",
            ]
        )

    grouped = strict_segments.groupby(["Firm_ID", "Year"], as_index=False).agg(
        N_Strict_Segments=("segment_text", "size"),
        N_Strict_Substantive=("label", lambda s: int((s == "substantive").sum())),
        N_Strict_Verifiable=("verifiable_segment", "sum"),
        Strict_Mean_Verifiable_Score=("verifiable_score", "mean"),
    )
    grouped["Strict_Verifiable_Share"] = grouped["N_Strict_Verifiable"] / grouped["N_Strict_Segments"]
    grouped["Has_Strict_Verifiable"] = (grouped["N_Strict_Verifiable"] > 0).astype(int)
    grouped["Has_Strict_Soft_Substantive"] = (
        (grouped["N_Strict_Substantive"] > 0) & (grouped["Has_Strict_Verifiable"] == 0)
    ).astype(int)
    grouped["Has_Strict_Symbolic_Only"] = (
        (grouped["N_Strict_Substantive"] == 0) & (grouped["Has_Strict_Verifiable"] == 0)
    ).astype(int)
    grouped["Type_V6"] = np.where(
        grouped["Has_Strict_Verifiable"] == 1,
        "verifiable",
        np.where(grouped["Has_Strict_Soft_Substantive"] == 1, "soft_substantive", "symbolic"),
    )
    return grouped


def build_event_panel(panel: pd.DataFrame, strict_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = panel.sort_values(["Firm_ID", "Year"]).copy()
    adopter_years = strict_features[["Firm_ID", "Year", "Type_V6"]].drop_duplicates().copy()

    first_ai = adopter_years.sort_values(["Firm_ID", "Year"]).groupby("Firm_ID").head(1).copy()
    first_ai = first_ai.rename(columns={"Year": "first_ai_year", "Type_V6": "first_type_v6"})
    first_ai = first_ai.loc[first_ai["first_ai_year"].between(2017, 2023)].copy()

    event_df = ordered.merge(first_ai, on="Firm_ID", how="left")
    event_df["never_adopter"] = event_df["first_ai_year"].isna().astype(int)
    event_df["event_time"] = event_df["Year"] - event_df["first_ai_year"]
    event_df = event_df.loc[(event_df["never_adopter"] == 1) | (event_df["event_time"].between(-2, 1))].copy()

    for prefix, cohort in [("ver", "verifiable"), ("soft", "soft_substantive"), ("sym", "symbolic")]:
        event_df[f"{prefix}_cohort"] = (event_df["first_type_v6"] == cohort).astype(int)
        for evt_name, evt_value in [("m2", -2), ("0", 0), ("p1", 1)]:
            event_df[f"{prefix}_evt_{evt_name}"] = (
                (event_df[f"{prefix}_cohort"] == 1) & (event_df["event_time"] == evt_value)
            ).astype(float)
    return event_df, first_ai


def fit_event_model(df: pd.DataFrame, with_controls: bool, label: str) -> tuple[pd.DataFrame, object]:
    keep = ["Firm_ID", "Year", "TobinQ"] + EVENT_TERMS + CONTROL_TERMS
    work = df[keep].dropna(subset=["TobinQ"]).copy()
    exog_cols = EVENT_TERMS.copy()
    if with_controls:
        work = work.dropna(subset=CONTROL_TERMS).copy()
        exog_cols += CONTROL_TERMS

    sample = work.set_index(["Firm_ID", "Year"]).sort_index()
    model = PanelOLS(
        sample["TobinQ"],
        sample[exog_cols],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for term in exog_cols:
        rows.append(
            {
                "model": label,
                "term": term,
                "coef": result.params.get(term, np.nan),
                "std_err": result.std_errors.get(term, np.nan),
                "p_value": result.pvalues.get(term, np.nan),
                "n": int(result.nobs),
                "firms": int(sample.index.get_level_values(0).nunique()),
                "r2_within": result.rsquared_within,
            }
        )
    return pd.DataFrame(rows), result


def build_key_results(panel: pd.DataFrame, first_ai: pd.DataFrame, fit_main, fit_ctrl) -> pd.DataFrame:
    rows = []
    for slot, desc, weights in [
        ("E1", "Verifiable vs none at t+1", {"ver_evt_p1": 1.0}),
        ("E2", "Soft substantive vs none at t+1", {"soft_evt_p1": 1.0}),
        ("E3", "Symbolic vs none at t+1", {"sym_evt_p1": 1.0}),
        ("D1_pre", "Verifiable minus symbolic at t-2", {"ver_evt_m2": 1.0, "sym_evt_m2": -1.0}),
        ("D1_t0", "Verifiable minus symbolic at t", {"ver_evt_0": 1.0, "sym_evt_0": -1.0}),
        ("D1_t1", "Verifiable minus symbolic at t+1", {"ver_evt_p1": 1.0, "sym_evt_p1": -1.0}),
        ("D2_t1", "Verifiable minus soft substantive at t+1", {"ver_evt_p1": 1.0, "soft_evt_p1": -1.0}),
        ("D3_t1", "Soft substantive minus symbolic at t+1", {"soft_evt_p1": 1.0, "sym_evt_p1": -1.0}),
    ]:
        coef, p_value = linear_contrast(fit_main.params, fit_main.cov, weights)
        rows.append({"slot": slot, "description": desc, "coef": coef, "p_value": p_value, "n": int(fit_main.nobs)})

    for slot, desc, weights in [
        ("C1_t1", "Controlled: verifiable minus symbolic at t+1", {"ver_evt_p1": 1.0, "sym_evt_p1": -1.0}),
        ("C2_t1", "Controlled: verifiable minus soft substantive at t+1", {"ver_evt_p1": 1.0, "soft_evt_p1": -1.0}),
        ("C3_t1", "Controlled: soft substantive minus symbolic at t+1", {"soft_evt_p1": 1.0, "sym_evt_p1": -1.0}),
    ]:
        coef, p_value = linear_contrast(fit_ctrl.params, fit_ctrl.cov, weights)
        rows.append({"slot": slot, "description": desc, "coef": coef, "p_value": p_value, "n": int(fit_ctrl.nobs)})

    out = pd.DataFrame(rows)
    out["panel_rows"] = len(panel)
    out["panel_firms"] = panel["Firm_ID"].nunique()
    out["eligible_first_adopters"] = first_ai["Firm_ID"].nunique()
    return out


def summary_table(panel: pd.DataFrame, strict_segments: pd.DataFrame, strict_features: pd.DataFrame, first_ai: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "panel_rows": [len(panel)],
            "panel_firms": [panel["Firm_ID"].nunique()],
            "broad_adopter_obs": [int(panel["GenAI_Dummy"].sum())],
            "strict_segment_rows": [len(strict_segments)],
            "strict_firm_year_obs": [len(strict_features)],
            "strict_firms": [strict_features["Firm_ID"].nunique()],
            "eligible_first_adopters": [first_ai["Firm_ID"].nunique()],
            "first_verifiable": [int((first_ai["first_type_v6"] == "verifiable").sum())],
            "first_soft_substantive": [int((first_ai["first_type_v6"] == "soft_substantive").sum())],
            "first_symbolic": [int((first_ai["first_type_v6"] == "symbolic").sum())],
        }
    )


def build_examples(first_ai: pd.DataFrame, strict_segments: pd.DataFrame) -> pd.DataFrame:
    tagged_first = first_ai.rename(columns={"first_ai_year": "Year"})
    merged = strict_segments.merge(tagged_first, on=["Firm_ID", "Year"], how="inner")
    examples = merged.sort_values(
        ["first_type_v6", "verifiable_segment", "verifiable_score", "Firm_ID", "Year"],
        ascending=[True, False, False, True, True],
    )
    return examples[
        ["Firm_ID", "Year", "first_type_v6", "label", "strict_anchor_first", "verifiable_segment", "verifiable_score", "segment_text"]
    ].groupby("first_type_v6").head(15).reset_index(drop=True)


def strict_term_count_table(term_df: pd.DataFrame) -> pd.DataFrame:
    if term_df.empty:
        return pd.DataFrame(columns=["term_norm", "hits", "firm_years", "firms"])
    grouped = term_df.groupby("term_norm").agg(
        hits=("term_norm", "size"),
        firm_years=("Firm_ID", lambda s: int(len(pd.DataFrame({"Firm_ID": s, "Year": term_df.loc[s.index, "Year"]}).drop_duplicates()))),
        firms=("Firm_ID", "nunique"),
    ).reset_index()
    return grouped.sort_values(["hits", "firms"], ascending=[False, False]).reset_index(drop=True)


def dictionary_sheet() -> pd.DataFrame:
    rows = [{"bucket": "strict_main_keep", "term": term} for term in STRICT_ANCHORS]
    rows += [{"bucket": "appendix_only_drop_from_main", "term": term} for term in EXCLUDED_FROM_MAIN]
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    _, segments = load_v5_cache()

    strict_segments, term_df = build_strict_segments(segments)
    strict_features = aggregate_strict_features(strict_segments)
    event_df, first_ai = build_event_panel(panel, strict_features)

    print(
        "Strict first adopter types:",
        first_ai["first_type_v6"].value_counts(dropna=False).to_dict(),
        flush=True,
    )

    model_main, fit_main = fit_event_model(event_df, with_controls=False, label="V6_event_main")
    model_ctrl, fit_ctrl = fit_event_model(event_df, with_controls=True, label="V6_event_ctrl")

    summary = summary_table(panel, strict_segments, strict_features, first_ai)
    key_results = build_key_results(panel, first_ai, fit_main, fit_ctrl)
    examples = build_examples(first_ai, strict_segments)
    first_counts = first_ai["first_type_v6"].value_counts(dropna=False).rename_axis("first_type_v6").reset_index(name="firms")
    term_counts = strict_term_count_table(term_df)
    dictionary_df = dictionary_sheet()

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        key_results.to_excel(writer, sheet_name="key_results", index=False)
        first_counts.to_excel(writer, sheet_name="first_type_counts", index=False)
        term_counts.to_excel(writer, sheet_name="strict_term_counts", index=False)
        model_main.to_excel(writer, sheet_name="event_main", index=False)
        model_ctrl.to_excel(writer, sheet_name="event_ctrl", index=False)
        strict_features.to_excel(writer, sheet_name="firm_year_features", index=False)
        examples.to_excel(writer, sheet_name="examples", index=False)
        dictionary_df.to_excel(writer, sheet_name="dictionary_design", index=False)

    print(f"Saved v6 results to: {RESULTS_XLSX}", flush=True)


if __name__ == "__main__":
    main()

# Measuring Verifiability in Corporate Generative AI Disclosures

Dictionaries, gold-standard annotations, derived data, and code for the paper:

> **From topic salience to verifiability: constructing and validating a measurement system for corporate generative artificial intelligence disclosure texts**
> （从话题显著性到可验证性：企业生成式人工智能披露文本的测度体系构建与验证）

The paper builds a verifiability measurement system for corporate technology-disclosure text: a three-tier GenAI dictionary with time-validity constraints, segment-level scoring on five implementation cues (implementation verbs, time anchoring, named products/tools, identifiable partners, measurable rollout), and a three-way classification of first annual-report GenAI disclosures into *verifiable*, *soft-substantive*, and *symbolic*. The measure is cross-validated by human double-blind coding (Cohen's κ = 0.949), blind classification by two independent large language models, and SHAP-based interpretable machine learning, and is externally validated against Chinese A-share market reactions (2015–2024).

## Repository structure

```
dictionaries/
  genai_dictionaries_full_terms.csv   Three dictionary tiers with full term lists,
                                      construction boundaries, and functional roles
                                      (broad narrative / strict main / conservative checks)
gold_standard/
  02_gold_standard.xlsx               180 stratified gold-standard segments (60 per class)
                                      with five-cue flags, scores, and consensus labels
  01_blind_coder_A_COMPLETED.xlsx     Independent blind-coding sheet, coder A
  01_blind_coder_B_COMPLETED.xlsx     Independent blind-coding sheet, coder B
  04_kappa_results.xlsx               Inter-coder reliability (Cohen's κ = 0.949)
  coding_codebook.md                  Coding rules given to the blind coders
data/
  firm_year_text_features.csv         4,242 firm-year aggregated text features
                                      (N_Segments, N_Verifiable, Verifiable_Share,
                                      Mean_Verifiable_Score, Type_V5, ...)
  segment_features_no_text.csv        20,703 segment-level records: five-cue flags,
                                      verifiability score, segment classification
                                      (raw MD&A text omitted for database licensing)
  first_disclosure_cohort_1123.csv    1,123 first-disclosure firm-years (fid, first
                                      disclosure year, verifiability type, segment count)
code/
  01_extract_genai_segments.py        Dictionary matching + segment extraction +
                                      five-cue scoring + classification rules
  02_bert_zero_shot_classification.py Semantic (substantive/strategic/generic) labels
  03_strict_dictionary_event_study.py Strict-dictionary first-disclosure event study
  04_dynamic_event_acceleration.py    Dynamic event study + trajectory acceleration
  05_oster_ipw_heterogeneity.py       Oster bounds, IPW, size-interaction checks
  06_llm_investor_persona_elicitation.py  LLM validation & investor-persona prompts
                                      (T1 classification, T2 institutional, T3 retail;
                                      API keys are read from a local file, not included)
  07_shap_interpretability.py         LightGBM + SHAP transparency analyses (A1/A2/B)
supplementary/
  online_appendix.pdf                 Supplementary materials (PSM diagnostics, LLM
                                      prompt details, additional descriptives)
```

## Classification rules (as implemented)

Segment verifiability score: `S = has_completion + has_quant + has_artifact + has_partner + has_current`.

A segment is **verifiable** when the cues form one of the following combinations:
`(completion AND (quant OR artifact OR partner))` or `(quant AND artifact AND (partner OR current))` or `(partner AND artifact AND completion)`.

Firm-year type: **verifiable** if the firm-year has at least one verifiable segment; **soft-substantive** if it has none but has at least one semantically substantive segment; **symbolic** otherwise.

## Data licensing boundary

Raw annual-report MD&A texts and firm financial data are licensed from commercial databases (CSMAR) and cannot be redistributed. This repository releases all author-constructed artifacts: dictionaries, coding materials, derived segment/firm-year features, and analysis code. The 180 gold-standard segments quote short excerpts from publicly available annual reports for replication of the reliability analysis. Researchers with CSMAR access can rebuild the full pipeline with `code/01` – `code/03`.

## Requirements

Python ≥ 3.10; pandas, numpy, scikit-learn, lightgbm, shap, linearmodels, transformers (for `02`).

## Citation

A citation entry will be added upon publication. Until then, please cite this repository URL.

## 中文说明

本仓库公开论文《从话题显著性到可验证性：企业生成式人工智能披露文本的测度体系构建与验证》的三套分层词典全词表、180 段人工金标准与双盲编码材料、企业—年度及片段级衍生数据、全部分析代码与在线补充材料。受 CSMAR 等商业数据库授权约束，原始年报全文与财务数据不在公开范围内；具备数据库访问权限的研究者可用 `code/01`–`code/03` 复现完整管道。

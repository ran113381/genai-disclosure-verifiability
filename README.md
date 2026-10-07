# Measuring Verifiability in Corporate Generative AI Disclosures

Dictionaries, gold-standard annotations, derived data, and code for the paper:

> **From topic salience to verifiability: constructing and validating a measurement system for corporate generative artificial intelligence disclosure texts**
> （从话题显著性到可验证性：企业生成式人工智能披露文本的测度体系构建与验证）

The paper builds a verifiability measurement system for corporate technology-disclosure text: a three-tier GenAI dictionary with time-validity constraints, segment-level scoring on five implementation cues (implementation verbs, time anchoring, named products/tools, identifiable partners, measurable rollout), and a three-way classification of first annual-report GenAI disclosures into *verifiable*, *soft-substantive*, and *symbolic*. The measure is validated in two stages against human gold standards: a **600-passage binary identification** gold standard (dictionary vs. adjudicated human labels, Cohen's κ = 0.720, recall 1.00) for the dictionary-matching stage, and a **180-segment three-way verifiability** gold standard (two-coder reliability κ = 0.949) for the scoring stage. It is further cross-checked by blind classification from two independent large language models and SHAP-based interpretable machine learning, and externally validated against Chinese A-share market reactions (2015–2024).

## Repository structure

```
dictionaries/
  genai_dictionaries_full_terms.csv   Three dictionary tiers with full term lists,
                                      construction boundaries, and functional roles
                                      (broad narrative / strict main / conservative checks)
gold_standard/
  # Stage 2 — verifiability scoring (180 segments, three-way)
  02_gold_standard.xlsx               180 stratified gold-standard segments (60 per class)
                                      with five-cue flags, scores, and consensus labels
  01_blind_coder_A_COMPLETED.xlsx     Independent blind-coding sheet, coder A
  01_blind_coder_B_COMPLETED.xlsx     Independent blind-coding sheet, coder B
  04_kappa_results.xlsx               Inter-coder reliability (Cohen's κ = 0.949)
  coding_codebook.md                  Coding rules given to the blind coders
  # Stage 1 — dictionary identification (600 passages, binary)
  identification_gold_standard_600.xlsx                600 passages, two coders, all 122
                                      disagreements adjudicated (478 agreed + 122 adjudicated);
                                      machine vs gold
  identification_annotation_coder1_2_adjudicated.xlsx  Blind annotation + adjudication +
                                      sampling metadata + codebook (4 sheets)
  identification_codebook.md          Binary coding rules (GenAI disclosure vs. not)
  identification_metrics.json         Reproduced metrics (κ = 0.720, acc 0.860, recall 1.00)
  # Stage 3 — third-party LLM blind validation (Section 3.5)
  llm_blind_validation_180.csv        Per-segment blind predictions of both LLMs (Claude
                                      Sonnet 4.5, GPT-5.5) alongside the human consensus
                                      (all 180 segments; human = NC for the six without
                                      two-coder consensus)
  llm_blind_metrics.json              Reproduced LLM blind-validation metrics
                                      (κ = 0.70 / 0.54 vs consensus; inter-model κ = 0.68)
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
  08_gold_standard_evaluation.py      Two-stage human-gold-standard evaluation; reproduces
                                      κ = 0.720 (identification) and κ = 0.231 (verifiability)
  09_llm_blind_validation.py          Section 3.5 — re-scores the two LLMs' blind predictions
                                      against the human consensus (κ = 0.70 Claude / 0.54 GPT)
  10_benchmark_learned_classifiers.py Table 5 — cue rule vs learned classifiers on the human
                                      gold standard (rule 0.231 → cues ~0.23 → text ~0.45/0.55)
  11_confusion_matrix_figure.py       Figure 6 — confusion matrices of the two blind LLMs
                                      vs the human consensus
figures/
  figure6_llm_confusion.{png,pdf}     Generated by code/11
supplementary/
  online_appendix.pdf                 Supplementary materials (PSM diagnostics, LLM
                                      prompt details, additional descriptives)
```

## Classification rules (as implemented)

Segment verifiability score: `S = has_completion + has_quant + has_artifact + has_partner + has_current`.

A segment is **verifiable** when the cues form one of the following combinations:
`(completion AND (quant OR artifact OR partner))` or `(quant AND artifact AND (partner OR current))` or `(partner AND artifact AND completion)`.

Firm-year type: **verifiable** if the firm-year has at least one verifiable segment; **soft-substantive** if it has none but has at least one semantically substantive segment; **symbolic** otherwise.

## Identification gold standard: adjudication

The 600-passage identification gold standard was blind double-coded (coders unaware of dictionary-hit status). The two coders initially agreed on 478 of the 600 passages; the 122 initial disagreements were adjudicated by the authors, who reviewed each case individually on the basis of AI-assisted draft rationales. The resulting dictionary-vs-human metrics are TP/FP/FN/TN = 216/84/0/300, precision 0.720, recall 1.00, accuracy 0.860, and Cohen's κ = 0.720 (`gold_standard/identification_metrics.json`; reproduced by `code/08_gold_standard_evaluation.py`).

This is the final adjudication round. The previous version (211 GenAI-related passages, κ = 0.703) is preserved in the git history. The same 600-passage gold standard is shared with a companion study by the same authors.

## Data licensing boundary

Raw annual-report MD&A texts and firm financial data are licensed from commercial databases (CSMAR) and cannot be redistributed. This repository releases all author-constructed artifacts: dictionaries, coding materials, derived segment/firm-year features, and analysis code. The 180 verifiability and 600 identification gold-standard passages quote short excerpts from publicly available annual reports for replication of the validation analyses; a small number of long identification passages are truncated to 1,500 characters. Researchers with CSMAR access can rebuild the full pipeline with `code/01` – `code/03`.

## Requirements

Python ≥ 3.10; pandas, numpy, scikit-learn, lightgbm, shap, linearmodels, scipy, matplotlib,
transformers (for `02`). `sentence-transformers` + `torch` are optional and only needed to
reproduce the frozen-transformer benchmark row (κ ≈ 0.55) in `10`; the script runs without them.

## Citation

A citation entry will be added upon publication. Until then, please cite this repository URL.

## 中文说明

本仓库公开论文《从话题显著性到可验证性：企业生成式人工智能披露文本的测度体系构建与验证》的三套分层词典全词表、180 段人工金标准与双盲编码材料、企业—年度及片段级衍生数据、全部分析代码与在线补充材料。受 CSMAR 等商业数据库授权约束，原始年报全文与财务数据不在公开范围内；具备数据库访问权限的研究者可用 `code/01`–`code/03` 复现完整管道。

"""
Step 04: Extract GenAI-Related Paragraphs from MDA Text
========================================================
For every firm-year with GenAI_Dummy=1 (4,242 obs),
extract the surrounding paragraph context of each GenAI keyword hit.

Then classify each paragraph as:
  - Substantive (实质性): concrete implementation, metrics, outcomes
  - Strategic (策略性): vague plans, aspirational, future-oriented
  - Generic (背景性): industry background, trend description

Output:
  E:/Supply/data/04_genai_paragraphs.xlsx   (all extracted paragraphs)
  E:/Supply/data/04_genai_firm_scores.xlsx  (firm-year level scores)
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from collections import Counter

CNRDS_ROOT = Path(r"E:\Supply_Chain_Project\data\raw_data\cnrds_MDA")
DICT_PATH = Path(r"E:\Supply_Chain_Project\data\processed_data\04_genai_dictionary.xlsx")
GENAI_PANEL = Path(r"E:/Supply/data/01_genai_full_panel.xlsx")
OUT = Path(r"E:/Supply/data")

# ── Load GenAI dictionary ──
dict_df = pd.read_excel(DICT_PATH)
keywords = dict_df['term'].dropna().str.strip().tolist()
keywords = [k for k in keywords if len(k) >= 2]
keywords_sorted = sorted(keywords, key=len, reverse=True)
kw_pattern = re.compile('|'.join(re.escape(k) for k in keywords_sorted))

# ── Load GenAI panel (only firms with GenAI=1) ──
genai = pd.read_excel(GENAI_PANEL)
adopters = genai[genai['GenAI_Dummy'] == 1][['Firm_ID', 'Year']].copy()
adopters['Firm_ID'] = adopters['Firm_ID'].astype(str).str.zfill(6)
print(f"GenAI adopters to process: {len(adopters):,}")

# ──────────────────────────────────────────────
# CLASSIFICATION RULES (Dictionary-based)
# ──────────────────────────────────────────────

# 实质性标志词 (Substantive markers)
SUBSTANTIVE_MARKERS = [
    # 已完成/已实施
    r'已部署', r'已上线', r'已投入', r'已应用', r'已落地', r'已实现',
    r'已建成', r'已搭建', r'已开发', r'已推出', r'已完成',
    r'投入使用', r'投入运营', r'正式上线', r'正式运行',
    r'成功应用', r'成功部署', r'成功实施', r'成功研发',
    r'落地应用', r'规模化应用', r'商业化应用',
    # 具体数字/效果
    r'提升了?\d', r'提高了?\d', r'降低了?\d', r'减少了?\d',
    r'增长了?\d', r'节约了?\d', r'缩短了?\d',
    r'\d+%', r'\d+个', r'\d+项', r'\d+套', r'\d+台',
    r'效率提升', r'成本降低', r'准确率', r'良品率',
    # 具体产品/系统
    r'自研', r'自主研发', r'自主开发',
    r'获得.*专利', r'申请.*专利', r'授权专利',
    r'平台已', r'系统已', r'模型已',
]

# 策略性标志词 (Strategic/aspirational markers)
STRATEGIC_MARKERS = [
    r'将.*探索', r'将.*推进', r'将.*加强', r'将.*推动',
    r'计划.*开展', r'计划.*推进', r'计划.*建设',
    r'拟.*开展', r'拟.*推进', r'拟.*建设',
    r'积极探索', r'积极推进', r'积极布局', r'积极拥抱',
    r'持续关注', r'持续探索', r'持续推进',
    r'深入研究', r'深入探索',
    r'未来将', r'下一步将', r'力争', r'争取',
    r'战略.*布局', r'战略.*规划', r'战略.*部署',
    r'加大.*投入', r'加快.*建设', r'加强.*研发',
    r'努力.*打造', r'致力于',
]

# 背景性标志词 (Generic/background markers)
GENERIC_MARKERS = [
    r'随着.*发展', r'随着.*进步', r'随着.*普及',
    r'行业趋势', r'发展趋势', r'技术趋势',
    r'行业背景', r'宏观环境', r'政策背景',
    r'国家.*战略', r'国家.*政策', r'政府.*支持',
    r'数字经济时代', r'智能化时代', r'新一轮.*革命',
    r'蓬勃发展', r'方兴未艾', r'日新月异',
    r'广泛应用于', r'被广泛', r'引起.*关注',
    r'业界.*认为', r'普遍认为', r'众所周知',
]

sub_patterns = [re.compile(p) for p in SUBSTANTIVE_MARKERS]
str_patterns = [re.compile(p) for p in STRATEGIC_MARKERS]
gen_patterns = [re.compile(p) for p in GENERIC_MARKERS]

def classify_paragraph(text):
    """Classify a paragraph as substantive/strategic/generic."""
    sub_score = sum(1 for p in sub_patterns if p.search(text))
    str_score = sum(1 for p in str_patterns if p.search(text))
    gen_score = sum(1 for p in gen_patterns if p.search(text))

    # Priority: substantive > strategic > generic
    if sub_score > 0 and sub_score >= str_score:
        return 'substantive', sub_score, str_score, gen_score
    elif str_score > 0 and str_score > gen_score:
        return 'strategic', sub_score, str_score, gen_score
    elif gen_score > 0:
        return 'generic', sub_score, str_score, gen_score
    else:
        # Default: if no markers matched, classify by sentence structure
        if any(w in text for w in ['已', '完成', '实现', '达到']):
            return 'substantive', sub_score, str_score, gen_score
        elif any(w in text for w in ['将', '计划', '拟', '探索']):
            return 'strategic', sub_score, str_score, gen_score
        else:
            return 'generic', sub_score, str_score, gen_score

def extract_context(text, match_start, match_end, window=200):
    """Extract surrounding context around a keyword match."""
    # Find paragraph boundaries (。or \n)
    # Look backward for sentence start
    start = max(0, match_start - window)
    # Find nearest sentence boundary
    for i in range(match_start, start, -1):
        if text[i] in '。\n':
            start = i + 1
            break

    end = min(len(text), match_end + window)
    for i in range(match_end, end):
        if text[i] in '。\n':
            end = i + 1
            break

    return text[start:end].strip()

# ── Process all adopter MDA files ──
print("\nExtracting GenAI paragraphs...")
all_paragraphs = []
firm_scores = []

for idx, row in adopters.iterrows():
    firm_id = str(row['Firm_ID']).zfill(6)
    year = int(row['Year'])

    # Find MDA file
    txt_dir = CNRDS_ROOT / str(year) / "文本"
    if not txt_dir.exists():
        continue

    # Match filename
    target = None
    for f in os.listdir(txt_dir):
        if f.startswith(firm_id) and '12-31' in f and f.endswith('.txt'):
            target = txt_dir / f
            break

    if target is None:
        continue

    # Read text
    try:
        with open(target, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except:
        try:
            with open(target, 'r', encoding='gbk', errors='ignore') as f:
                text = f.read()
        except:
            continue

    if len(text) < 100:
        continue

    # Find all keyword matches and extract context
    matches = list(kw_pattern.finditer(text))
    if not matches:
        continue

    # Deduplicate paragraphs (merge overlapping contexts)
    paragraphs = []
    seen_ranges = []
    for m in matches:
        ctx = extract_context(text, m.start(), m.end())
        # Check overlap with previous
        overlap = False
        for sr, er in seen_ranges:
            if abs(m.start() - sr) < 100:
                overlap = True
                break
        if not overlap:
            cat, sub_s, str_s, gen_s = classify_paragraph(ctx)
            paragraphs.append({
                'Firm_ID': firm_id,
                'Year': year,
                'Keyword': m.group(),
                'Context': ctx[:500],  # cap at 500 chars
                'Category': cat,
                'Sub_Score': sub_s,
                'Str_Score': str_s,
                'Gen_Score': gen_s,
            })
            seen_ranges.append((m.start(), m.end()))

    all_paragraphs.extend(paragraphs)

    # Firm-year level aggregation
    cats = [p['Category'] for p in paragraphs]
    n_sub = cats.count('substantive')
    n_str = cats.count('strategic')
    n_gen = cats.count('generic')
    n_total = len(cats)

    firm_scores.append({
        'Firm_ID': firm_id,
        'Year': year,
        'N_Paragraphs': n_total,
        'N_Substantive': n_sub,
        'N_Strategic': n_str,
        'N_Generic': n_gen,
        'Pct_Substantive': n_sub / n_total if n_total > 0 else 0,
        'Pct_Strategic': n_str / n_total if n_total > 0 else 0,
        'Pct_Generic': n_gen / n_total if n_total > 0 else 0,
        # Binary indicators
        'Has_Substantive': 1 if n_sub > 0 else 0,
        'Is_Mostly_Substantive': 1 if n_sub > n_str + n_gen else 0,
        'Is_Washing': 1 if n_sub == 0 and (n_str + n_gen) > 0 else 0,
        # Continuous scores
        'AI_Substance_Score': n_sub / n_total if n_total > 0 else 0,
        'AI_Washing_Score': (n_str + n_gen) / n_total if n_total > 0 else 0,
    })

    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx+1}/{len(adopters)}...")

# ── Build DataFrames ──
para_df = pd.DataFrame(all_paragraphs)
score_df = pd.DataFrame(firm_scores)

# ── Summary ──
print(f"\n{'='*60}")
print(f"PARAGRAPH EXTRACTION & CLASSIFICATION")
print(f"{'='*60}")
print(f"  Total paragraphs extracted: {len(para_df):,}")
print(f"  Firm-years processed:       {len(score_df):,}")
print(f"\n  Classification distribution:")
if len(para_df) > 0:
    dist = para_df['Category'].value_counts()
    for cat, n in dist.items():
        print(f"    {cat:15s}: {n:>5,} ({n/len(para_df):.1%})")

print(f"\n  Firm-year level:")
print(f"    Has_Substantive=1:      {score_df['Has_Substantive'].sum():,} ({score_df['Has_Substantive'].mean():.1%})")
print(f"    Is_Mostly_Substantive=1:{score_df['Is_Mostly_Substantive'].sum():,} ({score_df['Is_Mostly_Substantive'].mean():.1%})")
print(f"    Is_Washing=1:           {score_df['Is_Washing'].sum():,} ({score_df['Is_Washing'].mean():.1%})")
print(f"    Avg AI_Substance_Score: {score_df['AI_Substance_Score'].mean():.3f}")
print(f"    Avg AI_Washing_Score:   {score_df['AI_Washing_Score'].mean():.3f}")

# By year
print(f"\n  By year:")
yr = score_df.groupby('Year').agg(
    N=('Firm_ID', 'count'),
    Pct_Substantive=('Has_Substantive', 'mean'),
    Pct_Washing=('Is_Washing', 'mean'),
    Avg_Sub_Score=('AI_Substance_Score', 'mean'),
)
print(yr.to_string())

# Sample paragraphs
print(f"\n  Sample SUBSTANTIVE paragraphs:")
sub_sample = para_df[para_df['Category'] == 'substantive'].head(3)
for _, r in sub_sample.iterrows():
    print(f"    [{r['Firm_ID']} {r['Year']}] {r['Context'][:120]}...")

print(f"\n  Sample STRATEGIC paragraphs:")
str_sample = para_df[para_df['Category'] == 'strategic'].head(3)
for _, r in str_sample.iterrows():
    print(f"    [{r['Firm_ID']} {r['Year']}] {r['Context'][:120]}...")

print(f"\n  Sample GENERIC paragraphs:")
gen_sample = para_df[para_df['Category'] == 'generic'].head(3)
for _, r in gen_sample.iterrows():
    print(f"    [{r['Firm_ID']} {r['Year']}] {r['Context'][:120]}...")

# ── Save ──
# Clean illegal characters for Excel
import openpyxl.utils.exceptions
ILLEGAL_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
for col in para_df.select_dtypes(include='object').columns:
    para_df[col] = para_df[col].apply(lambda x: ILLEGAL_CHARS_RE.sub('', str(x)) if pd.notna(x) else x)
para_df.to_excel(OUT / "04_genai_paragraphs.xlsx", index=False)
score_df.to_excel(OUT / "04_genai_firm_scores.xlsx", index=False)
print(f"\nSaved:")
print(f"  {OUT / '04_genai_paragraphs.xlsx'} ({len(para_df):,} rows)")
print(f"  {OUT / '04_genai_firm_scores.xlsx'} ({len(score_df):,} rows)")

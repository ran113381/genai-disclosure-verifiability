"""
Paper 2: BERT Zero-Shot Classification (Alternative to Fine-tuning)
===================================================================
Uses pre-trained Chinese BERT model for zero-shot text classification.
No manual labeling required - uses prompt-based classification.

Model: hfl/chinese-roberta-wwm-ext (or similar)
Labels: ['substantive', 'strategic', 'generic']

Input:  E:/Supply/data/04_genai_paragraphs.xlsx (20,703 paragraphs)
Output: E:/Supply/data/09_bert_classified_paragraphs.xlsx
"""

import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA = Path(r"E:/Supply/data")
OUT = Path(r"E:/Supply/data")

# ── Load data ──
print("Loading paragraphs...")
para = pd.read_excel(DATA / "04_genai_paragraphs.xlsx")
print(f"  Total: {len(para):,} paragraphs")

# Sample for testing (first 100)
test_sample = para.head(100).copy()
print(f"  Testing on first 100 paragraphs...")

# ── Load Zero-Shot Classification Pipeline ──
print("\nLoading Chinese BERT model for zero-shot classification...")
print("  Model: hfl/chinese-roberta-wwm-ext")

# Define candidate labels for each class
label_mapping = {
    'substantive': [
        "已部署", "已上线", "已投入", "已应用", "已实现", "成功", "完成",
        "自主研发", "效率提升", "成本降低", "准确率", "平台", "系统", "产品"
    ],
    'strategic': [
        "将探索", "计划", "拟", "积极推进", "持续关注", "未来将", "战略",
        "加大投入", "加强研发", "致力于", "力争"
    ],
    'generic': [
        "随着发展", "行业趋势", "背景", "普遍认为", "广泛", "蓬勃发展",
        "时代", "革命", "方兴未艾"
    ]
}

# Zero-shot classification using pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="hfl/chinese-roberta-wwm-ext",
    device=-1,  # CPU
    batch_size=16
)

# Candidate labels (simplified for zero-shot)
candidate_labels = ["实质性实施", "战略性规划", "背景性描述"]

def classify_text(text):
    """Classify a single text using zero-shot."""
    try:
        result = classifier(text[:512], candidate_labels, multi_label=False)
        label_idx = result['labels'].index(max(result['scores']))
        pred_label = result['labels'][label_idx]
        confidence = max(result['scores'])

        # Map back to English
        if pred_label == "实质性实施":
            return 'substantive', confidence
        elif pred_label == "战略性规划":
            return 'strategic', confidence
        else:
            return 'generic', confidence
    except Exception as e:
        return 'error', 0.0

# ── Run classification on test sample ──
print(f"\nClassifying {len(test_sample)} paragraphs...")
results = []
for i, row in test_sample.iterrows():
    bert_label, bert_conf = classify_text(row['Context'])
    orig_label = row['Category']

    results.append({
        'Firm_ID': row['Firm_ID'],
        'Year': row['Year'],
        'Keyword': row['Keyword'],
        'Context': row['Context'][:200],
        'Original_Label': orig_label,
        'BERT_Label': bert_label,
        'BERT_Confidence': bert_conf,
        'Match': 1 if bert_label == orig_label else 0,
    })

    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(test_sample)}...")

results_df = pd.DataFrame(results)

# ── Calculate agreement ──
print(f"\n{'='*60}")
print("BERT ZERO-SHOT vs DICTIONARY COMPARISON")
print(f"{'='*60}")

agreement = (results_df['Match'] == 1).mean()
print(f"\nOverall Agreement: {agreement:.1%}")

# Confusion matrix
print(f"\nConfusion Matrix (Rows=Original, Cols=BERT):")
confusion = pd.crosstab(results_df['Original_Label'], results_df['BERT_Label'])
print(confusion.to_string())

# By class
print(f"\nPer-Class Accuracy:")
for cls in ['substantive', 'strategic', 'generic']:
    sub = results_df[results_df['Original_Label'] == cls]
    if len(sub) > 0:
        acc = (sub['Match'] == 1).mean()
        print(f"  {cls}: {acc:.1%} ({len(sub)} samples)")

# Sample disagreements
disagree = results_df[results_df['Match'] == 0]
if len(disagree) > 0:
    print(f"\nSample Disagreements ({len(disagree)} total):")
    for i, r in disagree.head(5).iterrows():
        print(f"  Orig={r['Original_Label']}, BERT={r['BERT_Label']} (conf={r['BERT_Confidence']:.2f})")
        print(f"    Context: {r['Context'][:80]}...")

# Save results
results_df.to_excel(OUT / "09_bert_classification_sample.xlsx", index=False)
print(f"\nSaved: {OUT / '09_bert_classification_sample.xlsx'}")

print(f"\n{'='*60}")
print("NEXT STEP:")
print(f"{'='*60}")
if agreement >= 0.75:
    print("Agreement >= 75%: Dictionary method is validated.")
    print("  → Can proceed with dictionary-based results for paper submission.")
else:
    print("Agreement < 75%: Consider manual review or BERT fine-tuning.")

# -*- coding: utf-8 -*-
"""
VCC step 2/2 — elicitation on the FULL verified 735 first-disclosure
cohort (results/paper1_vcc/vcc_735_cohort.csv from script 134, hard-
guarded == manuscript 735/180/162/393).

Identical pre-registered design & prompts as script 132 (T1 verifiability
0-3, T2 sophisticated-investor adoption prob, T3 retail-investor adoption;
temperature=0; relay model labelled gpt5.5, identity as-labelled). The
verifiability axis here is the paper's STRICT-DICTIONARY class
(first_type_v6) — dictionary-derived, NOT two-human gold (that exists
only for the 180-segment validity subsample handled by 132/133).

2205 calls (735*3). Per-firm checkpoint stores RAW responses (auditable);
resumable; preflight 1 call -> abort on total failure (NO fabrication).

Out: results/paper1_vcc/vcc_735_elicitation.xlsx + _vcc735_ckpt.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
import requests

KEYF = Path(r"C:\Users\asus\.mc3_keys.txt")
OUT = Path(r"E:\Supply - SHAP\results\paper1_vcc")
COHORT = OUT / "vcc_735_cohort.csv"
CKPT = OUT / "_vcc735_ckpt.csv"
XLSX = OUT / "vcc_735_elicitation.xlsx"

cfg = {}
for ln in KEYF.read_text(encoding="utf-8").splitlines():
    if "=" in ln:
        k, v = ln.split("=", 1)
        cfg[k.strip()] = v.strip()
TOKEN = cfg["API_KEY"]
MODEL = cfg.get("MODEL", "gpt5.5")
BASE = cfg["BASE_URL"].rstrip("/")

SYS_T1 = (
    "你是独立文本标注员。仅依据片段文本本身、按下面定义做三分类，不使用关键词清单。\n"
    "可验证(verifiable)=含可被外部观察的实施证据：已部署/已上线/明确应用场景/"
    "可识别合作伙伴/可衡量推广/明确时间锚定/命名具体工具或产品。\n"
    "软实质性(soft_substantive)=有应用导向语言但缺乏上述可验证实施证据。\n"
    "象征性(symbolic)=仅宽泛、愿景式或概念性表述，或与GenAI无关的财报样板。\n"
    "score：0=symbolic,1=soft_substantive,2=verifiable但弱,3=verifiable且充分。\n"
    '只输出JSON：{"category":"verifiable|soft_substantive|symbolic","score":0-3}'
)
SYS_T2 = (
    "你是资深机构投资者/卖方资深分析师，擅长穿透年报措辞，对缺乏可被外部"
    "核实的实施证据的表述系统性打折扣。仅依据下面给定的中文年报披露片段"
    "本身，估计该公司已经真实、实质性落地采用生成式人工智能的概率。\n"
    '只输出JSON：{"p": 0到1之间的一个小数}，不要任何解释。'
)
SYS_T3 = (
    "你是普通个人投资者（散户），信息处理时间与专业能力有限，主要凭披露"
    "读起来给你的总体印象快速判断，不做深入尽职调查。仅依据下面给定的"
    "中文年报披露片段本身，给出你认为该公司已经真实采用生成式人工智能"
    "的程度。\n"
    '只输出JSON：{"p": 0到1之间的一个小数}，不要任何解释。'
)


def _extract_text(j):
    # Robust: relay may emit anthropic content=[{thinking},{text}],
    # a bare string, OpenAI choices[].message.content, or bury the JSON
    # in a non-"text" block. Never index content[0] blindly.
    if isinstance(j, str):
        return j.strip()
    try:
        ch = j.get("choices")
        if ch:
            c = (ch[0].get("message", {}) or {}).get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
    except Exception:
        pass
    blks = j.get("content", [])
    if isinstance(blks, str):
        return blks.strip()
    if not isinstance(blks, list):
        return ""
    txt = "".join(b.get("text", "") for b in blks
                  if isinstance(b, dict) and b.get("type") == "text")
    if not txt.strip():
        for b in blks:
            if isinstance(b, dict):
                for v in b.values():
                    if isinstance(v, str) and "{" in v and "}" in v:
                        return v.strip()
    return txt.strip()


def _anthropic(sys_p, seg, hdr):
    h = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    h.update(hdr)
    b = {"model": MODEL, "max_tokens": 1024, "temperature": 0, "system": sys_p,
         "messages": [{"role": "user", "content": "披露片段：\n" + seg[:1800]}]}
    r = requests.post(f"{BASE}/v1/messages", headers=h, json=b, timeout=(10, 45))
    if r.status_code == 200:
        return _extract_text(r.json()), 200
    return None, f"{r.status_code}:{r.text[:120]}"


def _openai(sys_p, seg, hdr):
    h = {"content-type": "application/json"}
    h.update(hdr)
    b = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
         "messages": [{"role": "system", "content": sys_p},
                      {"role": "user", "content": "披露片段：\n" + seg[:1800]}]}
    r = requests.post(f"{BASE}/v1/chat/completions", headers=h, json=b,
                      timeout=(10, 45))
    if r.status_code == 200:
        return _extract_text(r.json()), 200
    return None, f"{r.status_code}:{r.text[:120]}"


COMBOS = [
    ("anthropic+xapikey", _anthropic, {"x-api-key": TOKEN}),
    ("anthropic+bearer", _anthropic, {"authorization": "Bearer " + TOKEN}),
    ("openai+bearer", _openai, {"authorization": "Bearer " + TOKEN}),
]
WORK = {"fn": None, "hdr": None, "name": None}


def raw_call(sys_p, seg, parser, ok_fn, retries=6):
    # Retry until the response actually PARSES (not merely non-empty):
    # empty/thinking-only 200s and unparseable bodies are retried, so a
    # transient relay hiccup never leaves a permanent hole.
    fn, hdr = WORK["fn"], WORK["hdr"]
    last = "?"
    for k in range(retries):
        try:
            txt, st = fn(sys_p, seg, hdr)
            if txt:
                if ok_fn(parser(txt)):
                    return txt, "ok"
                last = "unparseable"
                time.sleep(1.5 * (k + 1))
                continue
            last = (st[:60] if isinstance(st, str) else f"empty:{st}")
            time.sleep(2 ** k)
        except Exception as e:
            last = str(e)[:110]
            time.sleep(2 ** k)
    return None, last


def parse_t1(txt):
    try:
        d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        c = str(d.get("category", "")).strip()
        s = int(d.get("score", -1))
        if c in ("verifiable", "soft_substantive", "symbolic"):
            return c, s
    except Exception:
        pass
    return None, None


def parse_p(txt):
    try:
        d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        return min(1.0, max(0.0, float(d.get("p"))))
    except Exception:
        return None


def main():
    if not COHORT.exists():
        print("cohort csv missing -> run 134 first. ABORT.", flush=True)
        sys.exit(1)
    g = pd.read_csv(COHORT)
    g["disclosure_text"] = g["disclosure_text"].fillna("").astype(str)
    g["FY"] = g["Firm_ID"].astype(str) + "_" + g["first_ai_year"].astype(str)
    probe = g["disclosure_text"].iloc[0]

    print(f"PREFLIGHT base={BASE} model={MODEL} n={len(g)}", flush=True)
    for name, fn, hdr in COMBOS:
        txt, st, ok = None, "?", False
        for _ in range(3):  # tolerate a transient empty/garbled probe
            try:
                txt, st = fn(SYS_T1, probe, hdr)
            except Exception as e:
                txt, st = None, f"EXC:{str(e)[:80]}"
            ok = bool(txt) and parse_t1(txt)[0] is not None
            if ok:
                break
            time.sleep(2)
        print(f"  {name:<20} -> {'OK' if ok else 'fail ' + str(st)[:100]}",
              flush=True)
        if ok:
            WORK.update(fn=fn, hdr=hdr, name=name)
            break
    if not WORK["fn"]:
        print("PREFLIGHT FAILED -> abort (no fabrication).", flush=True)
        sys.exit(1)
    print(f"using: {WORK['name']}", flush=True)

    done = {}
    if CKPT.exists():
        prev = pd.read_csv(CKPT)
        for r in prev.itertuples():
            done[r.FY] = r._asdict()
        print(f"resuming: {len(done)} cached", flush=True)

    rows = []
    for i, r in g.iterrows():
        fy = r["FY"]
        c0 = done.get(fy)
        if c0 and str(c0.get("t1_cat")) in (
                "verifiable", "soft_substantive", "symbolic") and \
                pd.notna(c0.get("t2_p")) and pd.notna(c0.get("t3_p")):
            rows.append({k: c0[k] for k in
                         ["FY", "Firm_ID", "first_ai_year", "strict_type",
                          "t1_cat", "t1_score", "t2_p", "t3_p",
                          "t1_raw", "t2_raw", "t3_raw", "note"]})
        else:
            seg = r["disclosure_text"]
            t1_raw, n1 = raw_call(SYS_T1, seg, parse_t1,
                                  lambda v: v is not None and v[0] is not None)
            t2_raw, n2 = raw_call(SYS_T2, seg, parse_p,
                                  lambda v: v is not None)
            t3_raw, n3 = raw_call(SYS_T3, seg, parse_p,
                                  lambda v: v is not None)
            c, s = parse_t1(t1_raw) if t1_raw else (None, None)
            rows.append({
                "FY": fy, "Firm_ID": r["Firm_ID"],
                "first_ai_year": r["first_ai_year"],
                "strict_type": r["first_type_v6"],
                "t1_cat": c, "t1_score": s,
                "t2_p": parse_p(t2_raw) if t2_raw else None,
                "t3_p": parse_p(t3_raw) if t3_raw else None,
                "t1_raw": (t1_raw or "")[:200],
                "t2_raw": (t2_raw or "")[:200],
                "t3_raw": (t3_raw or "")[:200],
                "note": f"{n1}|{n2}|{n3}"})
            time.sleep(0.25)
        if (i + 1) % 15 == 0 or i == len(g) - 1:
            pd.DataFrame(rows).to_csv(CKPT, index=False, encoding="utf-8-sig")
            print(f"  {i+1}/{len(g)} last={rows[-1]['note']} "
                  f"t2={rows[-1]['t2_p']} t3={rows[-1]['t3_p']}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(CKPT, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        d.to_excel(w, "per_firm", index=False)
    n_ok = d[["t2_p", "t3_p"]].notna().all(axis=1).sum()
    print(f"\nDONE: {len(d)} firms, {int(n_ok)} with T2&T3 parsed.")
    print(f"Saved -> {XLSX}\nRun 136 for 735 statistics.", flush=True)


if __name__ == "__main__":
    main()

# app/ingest.py
import os, glob, json, re, argparse, random
from typing import List, Dict, Any
import pandas as pd

# —— 可调参数（不确定就先用默认）——
MIN_LEN = 15          # 小段最小长度（字符数），太短多半是噪声或截断
SAMPLE_SIZE = 50      # 随机抽样输出多少行到 sample_units.jsonl

def normalize_text(s: str) -> str:
    # """轻量规范化：去首尾空白，压缩多空白为一个空格；保留标点与大小写"""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def as_list(x) -> List[str]:
    """把字段安全地转成 list[str]（None/str/list 都能处理）"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []

def process_record(rec: Dict[str, Any], source_file: str, stats: Dict[str, int], drops: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """把一条专利记录拆成多个小段（claims/desc），并做清洗"""
    title  = rec.get("title", "") or ""
    docnum = rec.get("doc_number", "") or ""
    abstr  = rec.get("abstract", "") or ""
    cpc    = rec.get("classification", "") or ""

    claims = as_list(rec.get("claims"))
    descs  = as_list(rec.get("detailed_description"))

    rows = []

    # 可选：同一文档内去重，避免完全重复段落反复出现
    seen_claims = set()
    seen_descs  = set()

    # 处理 claims
    for i, raw in enumerate(claims):
        t = normalize_text(raw)
        if not t:
            stats["drop_empty"] += 1
            if len(drops["empty"]) < 5: drops["empty"].append(f"[claim] {t!r}")
            continue
        if len(t) < MIN_LEN:
            stats["drop_short"] += 1
            if len(drops["short"]) < 5: drops["short"].append(f"[claim] {t!r}")
            continue
        if t in seen_claims:
            stats["drop_dup"] += 1
            continue
        seen_claims.add(t)
        unit_id = f"{docnum}::claim::{i}"
        rows.append({
            "unit_id": unit_id,
            "doc_number": docnum,
            "title": title,
            "classification": cpc,
            "section": "claim",
            "idx": i,
            "text": t,
            "abstract": abstr,
            "source_file": os.path.basename(source_file)
        })
        stats["keep_claim"] += 1

    # 处理 detailed_description
    for j, raw in enumerate(descs):
        t = normalize_text(raw)
        if not t:
            stats["drop_empty"] += 1
            if len(drops["empty"]) < 5: drops["empty"].append(f"[desc] {t!r}")
            continue
        if len(t) < MIN_LEN:
            stats["drop_short"] += 1
            if len(drops["short"]) < 5: drops["short"].append(f"[desc] {t!r}")
            continue
        if t in seen_descs:
            stats["drop_dup"] += 1
            continue
        seen_descs.add(t)
        unit_id = f"{docnum}::desc::{j}"
        rows.append({
            "unit_id": unit_id,
            "doc_number": docnum,
            "title": title,
            "classification": cpc,
            "section": "desc",
            "idx": j,
            "text": t,
            "abstract": abstr,
            "source_file": os.path.basename(source_file)
        })
        stats["keep_desc"] += 1

    return rows

def run(data_dir: str, out_parquet: str, out_sample: str, out_report: str):
    files = sorted(glob.glob(os.path.join(data_dir, "patents_ipa*.json")))
    if not files:
        raise FileNotFoundError(f"No files matched {data_dir}/patents_ipa*.json")

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    os.makedirs(os.path.dirname(out_sample), exist_ok=True)
    os.makedirs(os.path.dirname(out_report), exist_ok=True)

    stats = {
        "records": 0,
        "keep_claim": 0,
        "keep_desc": 0,
        "drop_empty": 0,
        "drop_short": 0,
        "drop_dup": 0
    }
    drops = {"empty": [], "short": []}
    rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skip {fp}: cannot read json ({e})")
            continue

        if isinstance(data, dict):
            # 兼容万一顶层是对象而不是数组
            items = data.get("items") or data.get("data") or []
        else:
            items = data

        for rec in items:
            stats["records"] += 1
            rows.extend(process_record(rec, fp, stats, drops))

    # 生成 DataFrame
    df = pd.DataFrame(rows, columns=[
        "unit_id","doc_number","title","classification",
        "section","idx","text","abstract","source_file"
    ])

    # 保存全量表（Parquet）
    df.to_parquet(out_parquet, index=False)

    # 抽样保存 JSONL（便于人工检查）
    pool = list(range(len(df)))
    random.shuffle(pool)
    sample_idx = pool[:min(SAMPLE_SIZE, len(df))]
    with open(out_sample, "w", encoding="utf-8") as w:
        for k in sample_idx:
            w.write(json.dumps(df.iloc[k].to_dict(), ensure_ascii=False) + "\n")

    # 统计信息与报告
    total_units = len(df)
    keep_total = stats["keep_claim"] + stats["keep_desc"]
    with open(out_report, "w", encoding="utf-8") as w:
        w.write("# Ingest Report (Step A)\n\n")
        w.write(f"- Source files: {len(files)}\n")
        w.write(f"- Patent records: {stats['records']}\n")
        w.write(f"- Units kept (total): {keep_total}\n")
        w.write(f"  - claims kept: {stats['keep_claim']}\n")
        w.write(f"  - desc kept:   {stats['keep_desc']}\n")
        w.write(f"- Dropped (empty): {stats['drop_empty']}\n")
        w.write(f"- Dropped (too short < {MIN_LEN} chars): {stats['drop_short']}\n")
        w.write(f"- Dropped (duplicate within doc): {stats['drop_dup']}\n\n")
        if drops["empty"]:
            w.write("## Examples: empty after normalization\n")
            for x in drops["empty"]:
                w.write(f"- {x}\n")
            w.write("\n")
        if drops["short"]:
            w.write(f"## Examples: too short (< {MIN_LEN})\n")
            for x in drops["short"]:
                w.write(f"- {x}\n")
            w.write("\n")
        w.write("## Schema\n")
        w.write("unit_id | doc_number | title | classification | section | idx | text | abstract | source_file\n")

    print(f"[OK] Saved units: {total_units} → {out_parquet}")
    print(f"[OK] Sample: {out_sample}")
    print(f"[OK] Report: {out_report}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing patents_ipa*.json")
    ap.add_argument("--out", default="./cache/units.parquet", help="output parquet path")
    ap.add_argument("--sample", default="./cache/sample_units.jsonl", help="sample jsonl path")
    ap.add_argument("--report", default="./notes/ingest_report.md", help="report markdown path")
    args = ap.parse_args()
    run(args.data, args.out, args.sample, args.report)

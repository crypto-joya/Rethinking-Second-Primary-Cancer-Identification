"""
Rule-Based SPC Identification Baseline (Non-LLM) v2
====================================================

Fully aligned with Hybrid:：
  - Input: The same LLM_input_annotated_200.csv (timeline_text)
  - Judgment: Completely consistent with the 6 clinical heuristic rules in the Hybrid reasoning prompt
  - The only difference: The extraction method (regex vs LLM) 

Privacy: The script runs locally, and the timeline_text does not leave the local machine. 

Input：LLM_input_timeline_annotated_anonymized.csv（patient_id, timeline_text）
Output：output_rule_baseline.csv

"""

import pandas as pd
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ================================================================
# 配置
# ================================================================

INPUT_FILE = "LLM_input_timeline_annotated_anonymized.csv"
OUTPUT_FILE = "output_rule_baseline.csv"

# ================================================================
# 一、癌症诊断语句识别（核心改进）
# ================================================================

# 癌症关键词（必须出现才认为是诊断语句）
CANCER_KEYWORDS = [
    r"癌", r"恶性肿瘤", r"CA\b", r"carcinoma", r"cancer",
    r"肿瘤", r"瘤", r"neoplasm", r"tumor", r"tumour",
    r"淋巴瘤", r"白血病", r"骨髓瘤", r"肉瘤",
    r"melanoma", r"sarcoma", r"lymphoma", r"leukemia",
]

# 排除模式（出现这些则不是独立原发诊断）
EXCLUDE_IN_CONTEXT = [
    r"转移", r"继发", r"复发", r"metastas", r"recurren",
    r"良性", r"benign", r"息肉", r"囊肿", r"纤维瘤",
    r"脂肪瘤", r"腺瘤", r"血管瘤", r"错构瘤", r"畸胎瘤",
    r"Warthin", r"腺淋巴瘤",  # Warthin瘤是良性的
    r"未见", r"未提示", r"排除", r"不考虑", r"不支持",
    r"否认", r"无.*(?:证据|发现|病灶)", r"不除外",
    r"可疑", r"疑似", r"待排", r"待查",
    r"炎症", r"结核", r"结节病",
    r"放射性", r"化疗后改变", r"放疗后",
    r"个人史", r"既往史",  # 单纯的个人史提及，无具体诊断
]

# 确认关键词（出现这些则提升可信度）
CONFIRM_KEYWORDS = [
    r"病理", r"活检", r"查见", r"确诊", r"免疫组化",
    r"IHC", r"穿刺", r"手术", r"切除", r"ESD",
    r"术后病理", r"活检报告", r"石蜡切片",
]

# 影像关键词（中等可信度）
IMAGING_KEYWORDS = [
    r"CT", r"MRI", r"PET", r"超声", r"影像",
    r"内镜", r"胃镜", r"肠镜", r"支气管镜",
    r"增强", r"占位", r"肿块", r"新生物",
]

# ================================================================
# 二、器官映射
# ================================================================

ORGAN_MAP = {
    # 消化系统
    "食管": "食管", "食道": "食管", "esophag": "食管",
    "贲门": "贲门", "胃底贲门": "贲门", "胃贲门": "贲门",
    "胃": "胃", "gastric": "胃", "stomach": "胃",
    "胃底": "胃", "胃体": "胃", "胃窦": "胃", "胃角": "胃",
    "残胃": "胃", "胃小弯": "胃",
    "十二指肠": "十二指肠", "duodenum": "十二指肠",
    "结肠": "结肠", "colon": "结肠",
    "直肠": "直肠", "rectum": "直肠",
    "乙状结肠": "结肠", "升结肠": "结肠", "横结肠": "结肠",
    "降结肠": "结肠", "盲肠": "结肠",
    "肝": "肝", "肝脏": "肝", "liver": "肝", "hepatic": "肝",
    "胆囊": "胆囊", "胆管": "胆管", "胆道": "胆管",
    "胰腺": "胰腺", "胰": "胰腺", "pancrea": "胰腺",
    # 呼吸系统
    "肺": "肺", "lung": "肺", "pulmonary": "肺",
    "支气管": "肺", "气管": "气管", "trachea": "气管",
    "左肺": "肺", "右肺": "肺", "左上肺": "肺", "左下肺": "肺",
    "右上肺": "肺", "右下肺": "肺", "肺门": "肺",
    # 头颈
    "喉": "喉", "larynx": "喉", "laryngeal": "喉",
    "下咽": "下咽", "hypopharynx": "下咽",
    "口咽": "口咽", "oropharynx": "口咽",
    "鼻咽": "鼻咽", "nasopharynx": "鼻咽",
    "口腔": "口腔", "舌": "舌", "唇": "唇",
    "腮腺": "腮腺", "甲状腺": "甲状腺", "thyroid": "甲状腺",
    # 泌尿
    "肾": "肾", "肾脏": "肾", "kidney": "肾", "renal": "肾",
    "膀胱": "膀胱", "bladder": "膀胱",
    "前列腺": "前列腺", "prostate": "前列腺",
    # 妇科
    "宫颈": "宫颈", "cervix": "宫颈", "cervical": "宫颈",
    "子宫": "子宫", "uterus": "子宫",
    "卵巢": "卵巢", "ovary": "卵巢",
    "子宫内膜": "子宫内膜", "endometr": "子宫内膜",
    "乳腺": "乳腺", "乳房": "乳腺", "breast": "乳腺",
    "外阴": "外阴", "阴道": "阴道",
    # 其他
    "脑": "脑", "brain": "脑", "cerebral": "脑", "颅内": "脑",
    "骨": "骨", "bone": "骨", "骨骼": "骨",
    "皮肤": "皮肤", "skin": "皮肤", "cutaneous": "皮肤",
    "胸膜": "胸膜", "pleura": "胸膜",
    "腹膜": "腹膜", "peritoneum": "腹膜",
    "睾丸": "睾丸", "testis": "睾丸",
    "食管胃交界": "贲门", "食管胃连接": "贲门",
    "食管胃连接处": "贲门",
}

# 按长度降序排列，优先匹配更长的词
ORGAN_KEYS_SORTED = sorted(ORGAN_MAP.keys(), key=len, reverse=True)

# ================================================================
# 三、组织学识别
# ================================================================

HISTOLOGY_PATTERNS = [
    (r"鳞状细胞癌", "鳞状细胞癌"), (r"鳞癌", "鳞状细胞癌"),
    (r"角化.*?鳞", "鳞状细胞癌"),
    (r"腺癌", "腺癌"), (r"adenocarcinoma", "腺癌"),
    (r"黏液腺癌", "黏液腺癌"), (r"印戒细胞癌", "印戒细胞癌"),
    (r"神经内分泌", "神经内分泌肿瘤"), (r"neuroendocrine", "神经内分泌肿瘤"),
    (r"小细胞癌", "小细胞癌"), (r"大细胞神经内分泌", "大细胞神经内分泌癌"),
    (r"淋巴瘤", "淋巴瘤"), (r"白血病", "白血病"),
    (r"肝细胞癌", "肝细胞癌"), (r"胆管细胞癌", "胆管细胞癌"),
    (r"cholangiocarcinoma", "胆管细胞癌"),
    (r"肉瘤", "肉瘤"), (r"sarcoma", "肉瘤"),
    (r"恶性黑色素瘤", "恶性黑色素瘤"), (r"melanoma", "恶性黑色素瘤"),
    (r"间质瘤", "胃肠道间质瘤"), (r"GIST", "胃肠道间质瘤"),
    (r"移行细胞癌", "移行细胞癌"),
    (r"基底细胞癌", "基底细胞癌"),
    (r"胶质母细胞瘤", "胶质母细胞瘤"),
    (r"高级别上皮内瘤变", "高级别上皮内瘤变"),
    (r"低级别上皮内瘤变", "低级别上皮内瘤变"),
    (r"原位癌", "原位癌"), (r"上皮内瘤变", "上皮内瘤变"),
    (r"异型增生", "异型增生"),
]

# ================================================================
# 四、日期提取
# ================================================================

DATE_PATTERNS = [
    r"(\d{4})[年\-\/\.](\d{1,2})[月\-\/\.](\d{1,2})[日号]?",
    r"(\d{4})[年\-\/\.](\d{1,2})[月]?",
    r"(\d{4})年",
    r"(\d{4})-(\d{2})-(\d{2})",
    r"(\d{4})/(\d{2})/(\d{2})",
]

# ================================================================
# 五、解剖连续性（对齐 Hybrid RULE 3）
# ================================================================

CONTIGUOUS_PAIRS = {
    ("食管", "贲门"), ("食管", "胃"), ("食管", "气管"),
    ("食管", "支气管"), ("食管", "下咽"), ("食管", "喉"),
    ("食管", "纵隔"), ("贲门", "胃"),
    ("下咽", "喉"), ("下咽", "食管"), ("喉", "气管"),
    ("口腔", "口咽"), ("口咽", "下咽"),
    ("结肠", "直肠"), ("胆囊", "肝"), ("胆管", "肝"),
}

# ================================================================
# 核心函数
# ================================================================

def identify_organ(text: str) -> Optional[str]:
    """从文本中识别器官"""
    for key in ORGAN_KEYS_SORTED:
        if key in text or re.search(re.escape(key), text, re.IGNORECASE):
            return ORGAN_MAP[key]
    return None


def identify_histology(text: str) -> Optional[str]:
    """从文本中识别组织学"""
    for pattern, label in HISTOLOGY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return None


def extract_date_from_text(text: str) -> Optional[str]:
    """从文本中提取日期"""
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            groups = m.groups()
            try:
                if len(groups) >= 3:
                    return f"{groups[0]}-{int(groups[1]):02d}-{int(groups[2]):02d}"
                elif len(groups) >= 2:
                    return f"{groups[0]}-{int(groups[1]):02d}"
                elif len(groups) >= 1:
                    return f"{groups[0]}"
            except (ValueError, IndexError):
                continue
    return None


def parse_date(date_str: str) -> Optional[datetime]:
    """解析日期"""
    if not date_str:
        return None
    for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def months_between(d1: Optional[datetime], d2: Optional[datetime]) -> Optional[int]:
    if d1 is None or d2 is None:
        return None
    return abs((d1.year - d2.year) * 12 + d1.month - d2.month)


def has_cancer_keyword(text: str) -> bool:
    """检查是否包含癌症关键词"""
    for kw in CANCER_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


def is_excluded(text: str) -> bool:
    """检查是否应排除"""
    for pattern in EXCLUDE_IN_CONTEXT:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def has_confirmation(text: str) -> bool:
    """检查是否有病理确认"""
    for kw in CONFIRM_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


def has_imaging(text: str) -> bool:
    """检查是否有影像支持"""
    for kw in IMAGING_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


def are_contiguous(site1: str, site2: str) -> bool:
    if not site1 or not site2:
        return False
    for a, b in CONTIGUOUS_PAIRS:
        if (a in site1 and b in site2) or (b in site1 and a in site2):
            return True
        if (a in site1 and a in site2) or (b in site1 and b in site2):
            return True
    return False


def same_histology_type(h1: str, h2: str) -> bool:
    if not h1 or not h2:
        return False
    h1, h2 = h1.lower(), h2.lower()
    scc = {"鳞", "squamous"}
    adc = {"腺癌", "adenocarcinoma"}
    nec = {"神经内分泌", "小细胞", "neuroendocrine"}
    if any(s in h1 for s in scc) and any(s in h2 for s in scc):
        return True
    if any(s in h1 for s in adc) and any(s in h2 for s in adc):
        return True
    if any(s in h1 for s in nec) and any(s in h2 for s in nec):
        return True
    if h1 == h2:
        return True
    return False


def is_preinvasive(hist: str) -> bool:
    if not hist:
        return False
    h = hist.lower()
    return any(kw in h for kw in [
        "原位", "in situ", "上皮内瘤变", "异型增生",
        "hgin", "lgin", "dysplasia"
    ])


# ================================================================
# Step 1: 从 timeline_text 提取肿瘤事件（v2：找诊断语句）
# ================================================================

def extract_tumors_from_text(timeline_text: str) -> List[Dict]:
    """
    v2 核心改进：不是"找器官"，而是"找癌症诊断语句"
    
    策略：
    1. 按句子分割
    2. 只保留包含癌症关键词的句子
    3. 排除包含否定/转移/良性等关键词的句子
    4. 从保留的句子中提取器官+组织学+日期
    5. 全文去重（同器官+同组织学 → 合并，保留最早日期）
    """
    if not timeline_text or pd.isna(timeline_text):
        return []

    text = str(timeline_text)
    
    # 按句子分割
    segments = re.split(r'[。\n；;！!？?]', text)
    
    # 收集所有候选诊断语句
    candidates = []
    
    for seg in segments:
        seg = seg.strip()
        if len(seg) < 6:
            continue
        
        # 必须包含癌症关键词
        if not has_cancer_keyword(seg):
            continue
        
        # 排除否定/转移/良性
        if is_excluded(seg):
            continue
        
        # 识别器官
        organ = identify_organ(seg)
        if not organ:
            continue
        
        # 识别组织学
        histology = identify_histology(seg)
        
        # 提取日期
        date = extract_date_from_text(seg)
        
        # 判断证据等级
        if has_confirmation(seg):
            ev_level = "A"
        elif has_imaging(seg):
            ev_level = "B"
        else:
            ev_level = "C"
        
        # 截取证据文本
        evidence = seg[:300] if len(seg) > 300 else seg
        
        candidates.append({
            "site": organ,
            "histology": histology or "",
            "date": date or "",
            "evidence": evidence,
            "evidence_level": ev_level,
        })
    
    if not candidates:
        return []
    
    # 全文去重：同器官+同组织学 → 合并，保留最早日期和最高证据等级
    merged = {}
    for c in candidates:
        key = (c["site"], c["histology"])
        if key not in merged:
            merged[key] = c
        else:
            # 保留更早的日期
            existing_date = parse_date(merged[key]["date"])
            new_date = parse_date(c["date"])
            if new_date and (not existing_date or new_date < existing_date):
                merged[key]["date"] = c["date"]
            # 保留更高的证据等级
            levels = {"A": 3, "B": 2, "C": 1}
            if levels.get(c["evidence_level"], 0) > levels.get(merged[key]["evidence_level"], 0):
                merged[key]["evidence_level"] = c["evidence_level"]
    
    return list(merged.values())


# ================================================================
# Step 2: 规则判断（与 Hybrid 6条规则完全对齐）
# ================================================================

def judge_spc(tumors: List[Dict]) -> Dict:
    if len(tumors) == 0:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": "No tumor events extracted.",
            "rule_applied": "none"
        }

    if len(tumors) == 1:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": f"Only one tumor ({tumors[0]['site']}).",
            "rule_applied": "none"
        }

    # 过滤癌前病变
    invasive = [t for t in tumors if not is_preinvasive(t.get("histology", ""))]
    if len(invasive) == 0:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": "All lesions are pre-invasive.",
            "rule_applied": "rule2_preinvasive"
        }

    # RULE 5: 排除淋巴结（除非独立淋巴瘤）
    non_lymph = [t for t in invasive if "淋巴" not in t["site"]]
    if len(non_lymph) < 2:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": "After excluding lymph nodes, only one primary remains.",
            "rule_applied": "rule5_lymph"
        }

    # RULE 1: 同一器官
    organs = set(t["site"] for t in non_lymph)
    if len(organs) == 1:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": f"All tumors in same organ ({list(organs)[0]}).",
            "rule_applied": "rule1_same_organ"
        }

    # 检查不同器官的肿瘤对
    spc_found = False
    spc_type = 0
    best_evidence = "C"
    best_reason = ""

    for i in range(len(non_lymph)):
        for j in range(i + 1, len(non_lymph)):
            t1, t2 = non_lymph[i], non_lymph[j]

            # RULE 3: 连续结构 + 同组织学 = 转移
            if are_contiguous(t1["site"], t2["site"]) and same_histology_type(
                t1.get("histology", ""), t2.get("histology", "")
            ):
                best_reason = (f"{t1['site']} and {t2['site']} are contiguous "
                              f"with same histology → metastasis.")
                continue

            # RULE 4: 吻合口
            if "吻合口" in t1.get("evidence", "") or "吻合口" in t2.get("evidence", ""):
                continue

            # 同器官不同组织学
            if t1["site"] == t2["site"] and not same_histology_type(
                t1.get("histology", ""), t2.get("histology", "")
            ):
                if t1.get("histology") and t2.get("histology"):
                    spc_found = True
                    best_reason = f"Different histology in {t1['site']}: {t1['histology']} vs {t2['histology']}."
                    break

            # 不同器官
            if t1["site"] != t2["site"]:
                spc_found = True
                best_reason = (f"Different sites: {t1['site']} ({t1.get('histology','?')}) "
                              f"and {t2['site']} ({t2.get('histology','?')}).")
                break

    if spc_found:
        # 时间类型
        dated = [(t, parse_date(t["date"])) for t in non_lymph]
        dated.sort(key=lambda x: x[1] or datetime.min)
        first = last = None
        for t, d in dated:
            if d:
                if first is None:
                    first = (t, d)
                last = (t, d)

        if first and last and first[1] != last[1]:
            interval = months_between(first[1], last[1])
            if interval is not None:
                spc_type = 2 if interval <= 6 else 1
            else:
                spc_type = 3
        else:
            spc_type = 3

        levels = [t.get("evidence_level", "C") for t in non_lymph]
        best_evidence = "A" if "A" in levels else ("B" if "B" in levels else "C")

        return {
            "is_spc": True, "SPC_type": spc_type,
            "Evidence_level": best_evidence,
            "reason": best_reason,
            "rule_applied": "confirmed"
        }

    if best_reason:
        return {
            "is_spc": False, "SPC_type": 0, "Evidence_level": "A",
            "reason": best_reason,
            "rule_applied": "rule3_contiguous"
        }

    return {
        "is_spc": False, "SPC_type": 0, "Evidence_level": "B",
        "reason": "Multiple tumors but none meet SPC criteria.",
        "rule_applied": "none"
    }


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 60)
    print("  Rule-Based SPC Baseline v2 (Non-LLM)")
    print("  策略: 找癌症诊断语句 → 提取器官/组织学 → 去重")
    print("  输入: timeline_text (与 Hybrid 相同)")
    print("=" * 60)

    print(f"\n[1/3] 读取数据...")
    if not os.path.exists(INPUT_FILE):
        print(f"  ❌ 文件不存在: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"  患者数: {len(df)}")

    if "timeline_text" not in df.columns:
        print(f"  ❌ 缺少 timeline_text 列")
        return

    print(f"\n[2/3] 逐例提取+判断...")

    results = []
    stats = {"spc": 0, "non_spc": 0, "no_tumor": 0}

    for _, row in df.iterrows():
        pid = row["patient_id"]
        timeline = row.get("timeline_text", "")

        tumors = extract_tumors_from_text(timeline)
        result = judge_spc(tumors)

        if result["is_spc"]:
            stats["spc"] += 1
        else:
            stats["non_spc"] += 1
        if len(tumors) == 0:
            stats["no_tumor"] += 1

        results.append({
            "patient_id": pid,
            "is_spc": result["is_spc"],
            "SPC_type": result["SPC_type"],
            "Evidence_level": result["Evidence_level"],
            "reason": result["reason"],
            "rule_applied": result["rule_applied"],
            "extracted_tumors": json.dumps(tumors, ensure_ascii=False),
            "n_tumors_extracted": len(tumors),
        })

    df_out = pd.DataFrame(results)

    print(f"\n[3/3] 保存结果...")
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"  结果统计")
    print(f"{'=' * 60}")
    print(f"  总数: {len(results)}")
    print(f"  判定 SPC: {stats['spc']}")
    print(f"  判定 Non-SPC: {stats['non_spc']}")
    print(f"  未提取到肿瘤: {stats['no_tumor']}")

    print(f"\n  规则分布:")
    for rule, count in df_out["rule_applied"].value_counts().items():
        print(f"    {rule}: {count}")

    print(f"\n  提取肿瘤数分布:")
    for n, count in df_out["n_tumors_extracted"].value_counts().sort_index().items():
        print(f"    {n}个肿瘤: {count}例")

    print(f"\n  ✅ 已保存: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

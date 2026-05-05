"""
script_5_hybrid_iterative_v5.py
================================
Method 5: Iterative Rule-Guided Reflection Agent

Architecture:
  extraction → reasoning → rule_check → [violation?] → feedback + re-reason → rule_check → ...
  
Key difference from Methods 1-4:
  - Rules are NOT in the reasoning prompt (clean baseline)
  - Rules are applied as POST-HOC checks
  - Violations trigger feedback to LLM for re-evaluation
  - LLM makes the FINAL decision (rules advise, not override)
  - Full reasoning trace is preserved for interpretability

Based on v5 script architecture by TA.
"""

import pandas as pd
import requests
import json
import time
import os
import re
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Dict, List, Any, Optional

# =========================
# Config
# =========================
API_URL = "http://localhost:11434/api/generate"
MODEL_EXTRACT = "deepseek-r1:32b"  
MODEL_REASON = "deepseek-r1:32b"

INPUT_FILE = "LLM_input_timeline_annotated_anonymized.csv"
OUTPUT_FILE = "output_hybrid_iterative_v5.csv"
TRACE_FILE = "trace_hybrid_iterative_v5.jsonl"  # full reasoning traces
ERROR_LOG = "error_log_hybrid_iterative_v5.txt"

MAX_RETRIES = 3
TIMEOUT = 720
MAX_ITERATIONS = 3  # max reflection rounds


# =========================
# LLM
# =========================
def call_llm(model: str, prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "seed": 42
                    }
                },
                timeout=TIMEOUT
            )
            if r.status_code == 200:
                return r.json()["response"]
            else:
                log_error(f"[{attempt}/{MAX_RETRIES}] HTTP {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            log_error(f"[{attempt}/{MAX_RETRIES}] Timeout after {TIMEOUT}s")
        except requests.exceptions.ConnectionError:
            log_error(f"[{attempt}/{MAX_RETRIES}] Connection failed")
        except Exception as e:
            log_error(f"[{attempt}/{MAX_RETRIES}] {str(e)}")

        if attempt < MAX_RETRIES:
            time.sleep(5)
    return None


# =========================
# Utils
# =========================
def log_error(msg: str):
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def clean_json(raw: str) -> str:
    if raw is None:
        return ""
    raw = re.sub(r"<think[\s\S]*?</think\s*>", "", raw, flags=re.DOTALL)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return m.group(0)
    stripped = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    m = re.search(r"\{.*\}", stripped, re.DOTALL)
    return m.group(0) if m else raw


def safe_json(raw: str) -> dict:
    try:
        return json.loads(clean_json(raw))
    except:
        return None


SPC_TYPE_MAP = {
    "0": 0, "non-spc": 0, "none": 0,
    "1": 1, "metachronous": 1, "metachronous spc": 1,
    "2": 2, "synchronous": 2, "synchronous spc": 2,
    "3": 3, "indeterminate": 3, "indeterminate spc": 3
}


def parse_llm_result(raw: str) -> Tuple[Dict[str, Any], str]:
    """Parse LLM output into structured result."""
    d = safe_json(raw)
    if not d:
        return {"is_spc": None, "SPC_type": None, "Evidence_level": None, "reason": "parse_error"}, raw

    is_spc = d.get("is_spc")
    if isinstance(is_spc, bool):
        pass
    elif isinstance(is_spc, int):
        is_spc = bool(is_spc)
    elif isinstance(is_spc, str):
        is_spc = is_spc.lower().strip() in ["true", "yes", "1"]
    else:
        is_spc = None

    spc_type = d.get("SPC_type")
    if isinstance(spc_type, int):
        pass
    elif isinstance(spc_type, str):
        mapped = SPC_TYPE_MAP.get(str(spc_type).lower().strip())
        if mapped is not None:
            spc_type = mapped
        else:
            try:
                spc_type = int(spc_type)
            except:
                spc_type = None
    else:
        spc_type = None
    if spc_type not in [0, 1, 2, 3]:
        spc_type = None

    evidence = d.get("Evidence_level", "")
    if isinstance(evidence, str):
        evidence = evidence.upper().strip()
    if evidence not in ["A", "B", "C"]:
        evidence = None

    return {
        "is_spc": is_spc,
        "SPC_type": spc_type,
        "Evidence_level": evidence,
        "reason": d.get("reason", "")
    }, raw


# =========================
# Step 1: Extraction (same as v5)
# =========================
def extract_prompt(text: str) -> str:
    return f"""You are a clinical oncology information extraction assistant. Your task is to extract DISTINCT TUMOR EVENTS from longitudinal clinical records.

================================ CRITICAL GOAL ================================
Extract each INDEPENDENT tumor as ONE event. Do NOT create duplicate events for the same tumor.

================================ OUTPUT FORMAT (JSON ONLY) ================================
{{
  "tumors": [
    {{
      "site": "",
      "histology": "",
      "date": "",
      "evidence": ""
    }}
  ]
}}

================================ EXTRACTION RULES ================================

1. INCLUDE ALL MALIGNANCIES
You MUST extract:
- All solid tumors (e.g., esophageal cancer, lung cancer, gastric cancer)
- All hematologic malignancies (leukemia, lymphoma, myeloma)
- Tumors mentioned in past medical history, prior diagnoses, discharge summaries
⚠️ Even if not the main disease in this admission, you MUST include it.

--------------------------------
2. DO NOT MISS PRIOR CANCERS
If the text mentions "history of cancer", "既往…癌", "previously diagnosed with…" → extract as a tumor event.

--------------------------------
3. STRICTLY EXCLUDE NON-CONFIRMED LESIONS
DO NOT extract if only described as:
- "考虑肿瘤", "不除外恶性", "待排", "可能是", "可疑"
UNLESS there is a confirmed diagnosis elsewhere in the record.

--------------------------------
4. EVIDENCE MUST BE VERBATIM
Copy the ORIGINAL diagnostic sentence(s). DO NOT summarize or paraphrase.

--------------------------------
5. DATE
Use exact date if available. Otherwise estimate (YYYY or YYYY-MM). If unknown → "".

--------------------------------
6. NO CLINICAL REASONING IN EXTRACTION
DO NOT decide metastasis, recurrence, or SPC. Just extract what is documented.

--------------------------------
7. ⚠️ CONSOLIDATION RULES (VERY IMPORTANT — prevents duplicate events)

DO NOT create separate tumor events for:
(a) Multiple biopsies/procedures of the SAME lesion
    → Example: endoscopic biopsy + ESD + post-op pathology of the same esophageal lesion = 1 event
(b) Disease progression of the SAME tumor at the SAME site
    → Example: HGIN → carcinoma in situ → invasive SCC at the same esophageal location = 1 event
    → Use the MOST ADVANCED diagnosis as the histology
(c) Same organ + same histology diagnosed within 3 months
    → Example: "esophageal SCC" on 2023-01-15 and "esophageal SCC" on 2023-02-01 = 1 event

KEEP SEPARATE only when:
- Clearly DIFFERENT organs (e.g., esophagus + lung)
- Clearly DIFFERENT histological types (e.g., SCC + adenocarcinoma)
- Explicitly stated as independent (e.g., "双原发", "多原发", "second primary")

--------------------------------
8. LANGUAGE
Output must be valid JSON only. No explanation outside JSON.

================================ INPUT ================================
{text}
"""


def extract(text: str) -> Tuple[list, str]:
    raw = call_llm(MODEL_EXTRACT, extract_prompt(text))
    d = safe_json(raw)
    tumors = d.get("tumors", []) if d else []
    return tumors, raw or ""


# =========================
# Step 2: Initial Reasoning (NO clinical rules in prompt)
# =========================
def initial_reasoning_prompt(tumors: list) -> str:
    """
    Clean reasoning prompt — NO clinical rules embedded.
    Rules will be applied externally via rule_check + feedback loop.
    This is the key design difference from Methods 2-4.
    """
    return f"""You are an experienced oncology clinician specializing in esophageal cancer and multiple primary cancers.

Based on the extracted tumor events below, determine whether the patient has a second primary cancer (SPC).

================================ DEFINITIONS ================================
SPC (Second Primary Cancer): A new, independent primary malignancy that is NOT:
- Metastasis from an existing primary tumor
- Local recurrence of the same primary tumor
- Direct extension/invasion from an adjacent tumor
- Regional lymph node involvement from the same primary

SPC_type definitions:
- 0: non-SPC (no second primary cancer)
- 1: metachronous SPC (diagnosed >6 months after the first primary)
- 2: synchronous SPC (diagnosed ≤6 months from the first primary)
- 3: SPC with indeterminate timing (SPC is present but time interval cannot be determined)

Evidence_level definitions:
- A: high confidence (supported by pathology/biopsy/histology, including immunohistochemistry)
- B: moderate confidence (supported by clinical findings/imaging only, no pathology)
- C: low confidence (only personal history or uncertain/conflicting evidence)

================================ OUTPUT ================================
Output valid JSON only, in English only. No explanation outside JSON.
{{"is_spc": true/false, "SPC_type": 0/1/2/3, "Evidence_level": "A"/"B"/"C", "reason": ""}}

================================ EXTRACTED TUMOR EVENTS ================================
{json.dumps(tumors, ensure_ascii=False)}
"""


# =========================
# Step 3: Rule Check System
# =========================
# Organ grouping (same as v4)
ORGAN_SYNONYMS = {
    "食管": ["食管", "食道", "esophagus", "esophageal", "颈段食管", "胸段食管",
             "胸上段", "胸中段", "胸下段", "食管上段", "食管中段", "食管下段",
             "食管胃交界", "esophagogastric"],
    "胃": ["胃", "gastric", "stomach", "胃底", "胃体", "胃窦", "胃角", "胃贲门",
           "cardia", "gastric cardia"],
    "肺": ["肺", "lung", "pulmonary", "支气管", "左肺", "右肺", "左上肺", "左下肺",
           "右上肺", "右下肺", "肺门", "支气管"],
    "结直肠": ["结肠", "直肠", "colon", "rectum", "colorectal", "乙状结肠", "升结肠",
               "横结肠", "降结肠", "盲肠", "sigmoid"],
    "肝胆": ["肝", "肝脏", "liver", "hepatic", "胆管", "胆囊", "biliary", "bile duct"],
    "头颈": ["下咽", "喉", "咽", "hypopharynx", "larynx", "pharynx", "口腔", "舌",
             "head and neck", "neck"],
    "泌尿": ["膀胱", "肾", "肾脏", "前列腺", "bladder", "kidney", "prostate", "renal",
             "ureter", "输尿管"],
    "妇科": ["卵巢", "子宫", "宫颈", "ovary", "ovarian", "uterine", "cervix", "uterus",
             "endometrium", "子宫内膜"],
    "乳腺": ["乳腺", "乳房", "breast"],
    "皮肤": ["皮肤", "skin", "cutaneous"],
    "血液": ["淋巴瘤", "白血病", "lymphoma", "leukemia", "myeloma", "骨髓瘤",
             "multiple myeloma"],
    "甲状腺": ["甲状腺", "thyroid"],
    "胰腺": ["胰腺", "pancreas", "pancreatic"],
}


def get_organ(site: str) -> str:
    if not site:
        return ""
    s = site.strip().lower()
    for organ, synonyms in ORGAN_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in s or s in syn.lower():
                return organ
    return s


def parse_date(date_str: str) -> Optional[str]:
    """Extract year from date string for interval calculation."""
    if not date_str:
        return None
    m = re.search(r"(\d{4})", str(date_str))
    return m.group(1) if m else None


def calc_month_interval(d1: str, d2: str) -> Optional[int]:
    """Calculate month interval between two date strings. Returns None if unparseable."""
    def to_ym(s):
        s = str(s).strip()
        m = re.match(r"(\d{4})-(\d{1,2})", s)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        m = re.match(r"(\d{4})", s)
        if m:
            return int(m.group(1)) * 12
        return None
    ym1, ym2 = to_ym(d1), to_ym(d2)
    if ym1 is None or ym2 is None:
        return None
    return abs(ym1 - ym2)


def rule_check(tumors: list, llm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Post-hoc rule checker. Returns a list of violations.
    Each violation is a dict with:
      - rule_id: which rule
      - severity: "hard" (strong violation) or "soft" (advisory)
      - message: human-readable explanation
      - evidence: what triggered the violation
    
    IMPORTANT: This does NOT override the LLM judgment.
    It only IDENTIFIES potential issues for the feedback loop.
    """
    violations = []
    
    if llm_result.get("is_spc") is None:
        return violations  # can't check if no judgment
    
    is_spc = llm_result["is_spc"]
    reason_lower = (llm_result.get("reason") or "").lower()
    
    # ---- Structural analysis ----
    sites = []
    histologies = []
    organs = []
    dates = []
    for t in tumors:
        site = (t.get("site") or "").strip()
        histo = (t.get("histology") or "").strip()
        date = (t.get("date") or "").strip()
        sites.append(site.lower() if site else "")
        histologies.append(histo.lower() if histo else "")
        organs.append(get_organ(site))
        dates.append(date)
    
    unique_sites = set(s for s in sites if s)
    unique_histologies = set(h for h in histologies if h)
    unique_organs = set(o for o in organs if o)
    
    # ---- Rule 1: Single tumor → cannot be SPC ----
    if is_spc and len(tumors) < 2:
        violations.append({
            "rule_id": "R1_single_tumor",
            "severity": "hard",
            "message": "Only one tumor event was extracted. A second primary cancer requires at least two independent primary tumors.",
            "evidence": f"Number of tumors: {len(tumors)}"
        })
    
    # ---- Rule 2: Same site + same histology → likely same primary ----
    if is_spc and len(unique_sites) == 1 and len(unique_histologies) <= 1:
        violations.append({
            "rule_id": "R2_same_site_histology",
            "severity": "hard",
            "message": "All tumors share the same site and histology. This pattern suggests the same primary tumor (possibly multifocal or progressive), not a second primary cancer.",
            "evidence": f"Sites: {unique_sites}, Histologies: {unique_histologies}"
        })
    
    # ---- Rule 3: Same organ + same histology → field cancerization ----
    if is_spc and len(unique_organs) == 1 and len(unique_histologies) <= 1 and len(unique_sites) > 1:
        violations.append({
            "rule_id": "R3_field_cancerization",
            "severity": "hard",
            "message": "All tumors are in the same organ with the same histology but different sub-sites. This pattern is consistent with field cancerization (e.g., multifocal esophageal SCC), not a second primary cancer. SPC requires either a different organ or a different histology.",
            "evidence": f"Organ: {unique_organs}, Sub-sites: {unique_sites}, Histology: {unique_histologies}"
        })
    
    # ---- Rule 4: Metastasis does NOT exclude SPC (reverse check) ----
    # If LLM says non-SPC and reason mentions metastasis, check if there's evidence
    # of an independent second tumor that the LLM might have overlooked
    if not is_spc and ("转移" in reason_lower or "metastas" in reason_lower):
        # Check if there are tumors at different organs with different histologies
        if len(unique_organs) > 1 and len(unique_histologies) > 1:
            violations.append({
                "rule_id": "R4_metastasis_exclusion",
                "severity": "soft",
                "message": "You concluded non-SPC citing metastasis, but there are tumors at different organs with different histologies. A patient CAN have both metastasis AND a second primary cancer. Please re-evaluate whether any of these tumors represent an independent second primary.",
                "evidence": f"Organs: {unique_organs}, Histologies: {unique_histologies}"
            })
    
    # ---- Rule 5: Lymph node only → not SPC ----
    if is_spc and len(tumors) >= 2:
        # Check if the "second tumor" is only a lymph node finding
        non_ln_sites = [s for s in sites if s and "淋巴" not in s and "lymph" not in s and "node" not in s]
        if len(non_ln_sites) <= 1:
            violations.append({
                "rule_id": "R5_lymph_node",
                "severity": "hard",
                "message": "The additional tumor(s) appear to be lymph node involvement only. Regional or distant lymph node metastasis from the same primary is not a second primary cancer.",
                "evidence": f"All sites: {sites}"
            })
    
    # ---- Rule 6: Imaging without pathology ----
    if is_spc and llm_result.get("Evidence_level") == "A":
        # Check if any tumor evidence mentions imaging-only findings
        has_imaging_only = False
        for t in tumors:
            ev = (t.get("evidence") or "").lower()
            if any(kw in ev for kw in ["ct", "mri", "pet", "超声", "影像", "imaging", "scan"]):
                has_imaging_only = True
                break
        if has_imaging_only:
            violations.append({
                "rule_id": "R6_imaging_only",
                "severity": "soft",
                "message": "You assigned Evidence_level A (pathology-confirmed), but some tumor events appear to be based on imaging findings only. Evidence_level A requires pathology/biopsy confirmation. Please verify the evidence level.",
                "evidence": "Imaging keywords found in tumor evidence"
            })
    
    # ---- Rule 7: Carcinoma in situ / HGIN at same organ ----
    if is_spc and len(tumors) >= 2:
        has_invasive = False
        has_cis_only = False
        for t in tumors:
            h = (t.get("histology") or "").lower()
            if any(kw in h for kw in ["in situ", "原位", "hgin", "high-grade intraepithelial", "重度不典型"]):
                has_cis_only = True
            if any(kw in h for kw in ["carcinoma", "癌", "scc", "adenocarcinoma", "鳞癌", "腺癌"]):
                if "in situ" not in h and "原位" not in h:
                    has_invasive = True
        
        if has_cis_only and not has_invasive and len(unique_organs) == 1:
            violations.append({
                "rule_id": "R7_cis_same_organ",
                "severity": "soft",
                "message": "The additional finding appears to be carcinoma in situ or HGIN at the same organ as the index tumor. Pre-invasive lesions at the same organ are generally not considered second primary cancers.",
                "evidence": f"Organs: {unique_organs}, CIS/HGIN detected"
            })
    
    # ---- Rule 8: Anastomotic recurrence ----
    if is_spc and len(tumors) >= 2:
        for t in tumors:
            s = (t.get("site") or "").lower()
            if any(kw in s for kw in ["吻合口", "anastomos", "残胃", "gastric tube", "代胃"]):
                violations.append({
                    "rule_id": "R8_anastomotic",
                    "severity": "soft",
                    "message": "One of the tumors is at or near the anastomotic site. Tumor at the anastomosis with the same histology typically represents local recurrence, not a second primary cancer.",
                    "evidence": f"Site: {t.get('site')}"
                })
                break
    
    return violations


# =========================
# Step 4: Feedback Prompt (for re-evaluation)
# =========================
def reflection_prompt(tumors: list, previous_result: Dict, violations: List[Dict], round_num: int) -> str:
    """
    Construct feedback prompt for re-evaluation.
    
    Key design principles:
    1. Present violations as "clinical reviewer concerns" (not commands)
    2. Give the LLM the option to DISAGREE with the rule (if it has good reasons)
    3. Require the LLM to explicitly address each violation
    4. Keep the original reasoning visible for context
    """
    
    violation_text = ""
    for i, v in enumerate(violations, 1):
        severity_label = "⚠️ IMPORTANT" if v["severity"] == "hard" else "💡 Please consider"
        violation_text += f"""
{severity_label} — Clinical Reviewer Concern #{i} ({v['rule_id']}):
{v['message']}
Evidence: {v['evidence']}
"""
    
    return f"""You are an experienced oncology clinician. A clinical reviewer has raised concerns about your previous SPC assessment.

================================ YOUR PREVIOUS ASSESSMENT ================================
Judgment: {"SPC (Second Primary Cancer)" if previous_result.get("is_spc") else "Non-SPC"}
SPC_type: {previous_result.get("SPC_type")}
Evidence_level: {previous_result.get("Evidence_level")}
Your reasoning: {previous_result.get("reason", "")}

================================ CLINICAL REVIEWER CONCERNS ================================
The following concerns were identified during a structured rule-based review:
{violation_text}

================================ YOUR TASK ================================
Please RE-EVALUATE your assessment considering the concerns above.

IMPORTANT INSTRUCTIONS:
1. For each concern, explicitly state whether you AGREE or DISAGREE and why.
2. If you agree with a concern, revise your judgment accordingly.
3. If you DISAGREE with a concern, explain why (e.g., specific clinical evidence that overrides the rule).
4. You are the final decision-maker. The rules are advisory — your clinical judgment takes precedence if you have strong evidence.

This is reflection round {round_num} of {MAX_ITERATIONS}. Please be thorough in your reasoning.

================================ EXTRACTED TUMOR EVENTS ================================
{json.dumps(tumors, ensure_ascii=False)}

================================ OUTPUT ================================
Output valid JSON only, in English only. No explanation outside JSON.
{{"is_spc": true/false, "SPC_type": 0/1/2/3, "Evidence_level": "A"/"B"/"C", "reason": ""}}
"""


# =========================
# Step 5: Iterative Agent Loop
# =========================
def iterative_agent(tumors: list, patient_id: str) -> Tuple[Dict[str, Any], Dict]:
    """
    Main iterative agent loop.
    
    Flow:
      1. Initial reasoning (no rules in prompt)
      2. Rule check → violations?
         - No violations → return result
         - Violations → construct feedback → re-reason
      3. Repeat until no violations or MAX_ITERATIONS
    
    Returns:
      - final_result: the final LLM judgment
      - trace: full reasoning trace for interpretability
    """
    trace = {
        "patient_id": patient_id,
        "num_tumors": len(tumors),
        "iterations": [],
        "total_rounds": 0,
        "converged": False,
        "final_judgment_changed": False
    }
    
    # ---- Round 1: Initial reasoning ----
    raw_1 = call_llm(MODEL_REASON, initial_reasoning_prompt(tumors))
    result_1, _ = parse_llm_result(raw_1)
    
    trace["iterations"].append({
        "round": 1,
        "prompt_type": "initial",
        "is_spc": result_1["is_spc"],
        "SPC_type": result_1["SPC_type"],
        "Evidence_level": result_1["Evidence_level"],
        "reason": result_1["reason"],
        "raw_output": raw_1 or "",
        "violations": [],
        "violation_count": 0
    })
    
    # ---- Rule check ----
    violations = rule_check(tumors, result_1)
    trace["iterations"][0]["violations"] = [v["rule_id"] for v in violations]
    trace["iterations"][0]["violation_count"] = len(violations)
    
    if not violations or result_1.get("is_spc") is None:
        trace["total_rounds"] = 1
        trace["converged"] = True
        return result_1, trace
    
    # ---- Iterative reflection ----
    current_result = result_1
    
    for round_num in range(2, MAX_ITERATIONS + 1):
        # Construct feedback
        feedback_prompt = reflection_prompt(tumors, current_result, violations, round_num)
        
        # Re-reason
        raw_n = call_llm(MODEL_REASON, feedback_prompt)
        new_result, _ = parse_llm_result(raw_n)
        
        # Check if judgment changed
        judgment_changed = (new_result.get("is_spc") != current_result.get("is_spc"))
        
        # Rule check on new result
        new_violations = rule_check(tumors, new_result)
        
        trace["iterations"].append({
            "round": round_num,
            "prompt_type": "reflection",
            "is_spc": new_result["is_spc"],
            "SPC_type": new_result["SPC_type"],
            "Evidence_level": new_result["Evidence_level"],
            "reason": new_result["reason"],
            "raw_output": raw_n or "",
            "violations": [v["rule_id"] for v in new_violations],
            "violation_count": len(new_violations),
            "judgment_changed": judgment_changed,
            "feedback_violations": [v["rule_id"] for v in violations]
        })
        
        # Update state
        current_result = new_result
        violations = new_violations
        
        # Check convergence: no violations → done
        if not violations:
            trace["total_rounds"] = round_num
            trace["converged"] = True
            trace["final_judgment_changed"] = (current_result["is_spc"] != result_1["is_spc"])
            return current_result, trace
        
        # Check convergence: same judgment as previous round → done (avoid oscillation)
        if not judgment_changed and round_num >= 2:
            trace["total_rounds"] = round_num
            trace["converged"] = True  # converged to stable judgment
            trace["final_judgment_changed"] = (current_result["is_spc"] != result_1["is_spc"])
            return current_result, trace
    
    # Max iterations reached
    trace["total_rounds"] = MAX_ITERATIONS
    trace["converged"] = False
    trace["final_judgment_changed"] = (current_result["is_spc"] != result_1["is_spc"])
    return current_result, trace


# =========================
# Resume
# =========================
def load_existing() -> Tuple[pd.DataFrame, set]:
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            if "patient_id" in df.columns:
                return df, set(df["patient_id"])
            else:
                return pd.DataFrame(), set()
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame(), set()
    return pd.DataFrame(), set()


# =========================
# Main
# =========================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    required_cols = {"patient_id", "timeline_text"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input CSV missing required columns: {missing}")

    existing, done = load_existing()
    results = existing.to_dict("records")
    
    # Open trace file in append mode
    trace_fh = open(TRACE_FILE, "a")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row["patient_id"]
        if pid in done:
            continue

        start = time.time()

        # Step 1: Extract
        tumors, extract_raw = extract(row["timeline_text"])

        # Step 2-4: Iterative agent
        final, trace = iterative_agent(tumors, pid)

        # Save trace
        trace_fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
        trace_fh.flush()

        results.append({
            "patient_id": pid,
            "is_spc": final["is_spc"],
            "SPC_type": final["SPC_type"],
            "Evidence_level": final["Evidence_level"],
            "reason": final["reason"],
            "extracted_tumors": json.dumps(tumors, ensure_ascii=False),
            "extract_raw_output": extract_raw,
            "raw_output": trace["iterations"][-1]["raw_output"],
            "agent_rounds": trace["total_rounds"],
            "agent_converged": trace["converged"],
            "judgment_changed": trace["final_judgment_changed"],
            "time_sec": round(time.time() - start, 2)
        })

        # Atomic write
        tmp = OUTPUT_FILE + ".tmp"
        pd.DataFrame(results).to_csv(tmp, index=False)
        os.replace(tmp, OUTPUT_FILE)

    trace_fh.close()

    print(f"\n{'=' * 50}")
    print(f"  Done: {len(results)} patients")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Traces: {TRACE_FILE}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

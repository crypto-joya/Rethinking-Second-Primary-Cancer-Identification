import pandas as pd
import requests
import json
import time
import os
import re
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Dict, List, Any

# =========================
# Config
# =========================
API_URL = "http://localhost:11434/api/generate"
MODEL_EXTRACT = "deepseek-r1:32b"
MODEL_REASON = "deepseek-r1:32b"
INPUT_FILE = "LLM_input_timeline_annotated_anonymized.csv"
OUTPUT_FILE = "output_hybrid_no_rule_v5.csv"
ERROR_LOG = "error_log_hybrid_no_rule_v5.txt"
MAX_RETRIES = 3
TIMEOUT = 720

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
    # Remove thinking process tags like <think...</think >
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

# =========================
# Step 1: Extraction (v4 — with deduplication)
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
    """Extract tumor events. Returns (parsed_tumors_list, raw_llm_output)."""
    raw = call_llm(MODEL_EXTRACT, extract_prompt(text))
    d = safe_json(raw)
    tumors = d.get("tumors", []) if d else []
    return tumors, raw or ""

# =========================
# Step 2: Reasoning (v4 — with clinical decision rules)
# =========================
def reasoning_prompt(tumors: list) -> str:
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

================================ CRITICAL CLINICAL RULES ================================

RULE 1: SAME-ORGAN MULTIFOCAL ≠ SPC
Multiple lesions in the same organ (e.g., esophageal multisegment, multifocal HGIN progressing to invasive cancer) represent field cancerization or disease progression, NOT separate primary cancers.
→ Classify as non-SPC unless histologies are clearly different types (e.g., squamous cell carcinoma vs adenocarcinoma in the same organ).

RULE 2: CARCINOMA IN SITU / HGIN HANDLING
- Pre-invasive lesions (carcinoma in situ, high-grade intraepithelial neoplasia/HGIN) at the SAME organ as the index tumor → NOT SPC.
- If the ONLY additional finding is in situ/HGIN at the same organ → non-SPC.
- If there is a confirmed invasive malignancy at a DIFFERENT organ → evaluate that invasive cancer as the potential SPC.

RULE 3: DISTINGUISHING METASTASIS FROM SPC (MOST IMPORTANT)
- Different organ + different histology + independent pathology → likely SPC
- Same histology + anatomically contiguous structures (e.g., esophagus→trachea, esophagus→gastric cardia) → likely metastasis/direct extension, NOT SPC
- Immunohistochemistry (IHC) is the gold standard: different IHC profiles support SPC
- ⚠️ CRITICAL: A patient CAN have BOTH metastasis AND a second primary cancer simultaneously.
  The presence of metastatic disease does NOT rule out SPC. Evaluate each tumor independently.

RULE 4: ANASTOMOTIC RECURRENCE ≠ SPC
- Tumor at or near the surgical anastomosis with the same histology = recurrence, NOT SPC
- Tumor in a reconstructed organ (e.g., gastric tube after esophagectomy) with same histology = recurrence

RULE 5: LYMPH NODES ≠ SPC
- Regional or distant lymph node metastasis from the same primary ≠ SPC
- Exception: if a lymph node harbors a completely different cancer type (e.g., esophageal SCC patient with lymphoma in a node), that could be SPC

RULE 6: IMAGING WITHOUT PATHOLOGY
- A suspicious lesion on imaging WITHOUT pathology/biopsy confirmation is NOT a confirmed malignancy
- Such cases should receive Evidence_level B (imaging-supported) or C (uncertain), NEVER A

================================ OUTPUT ================================
Output valid JSON only, in English only. No explanation outside JSON.
{{"is_spc": true/false, "SPC_type": 0/1/2/3, "Evidence_level": "A"/"B"/"C", "reason": ""}}

================================ EXTRACTED TUMOR EVENTS ================================
{json.dumps(tumors, ensure_ascii=False)}
"""

# Mapping for SPC_type in case LLM returns text instead of numbers
SPC_TYPE_MAP = {
    "0": 0, "non-spc": 0, "none": 0,
    "1": 1, "metachronous": 1, "metachronous spc": 1,
    "2": 2, "synchronous": 2, "synchronous spc": 2,
    "3": 3, "indeterminate": 3, "indeterminate spc": 3
}

def reason(tumors: list) -> Tuple[Dict[str, Any], str]:
    raw = call_llm(MODEL_REASON, reasoning_prompt(tumors))
    d = safe_json(raw)

    if not d:
        return {"is_spc": None, "SPC_type": None, "Evidence_level": None, "reason": "parse_error"}, raw

    # Handle is_spc bool, int, and string properly
    is_spc = d.get("is_spc")
    if isinstance(is_spc, bool):
        pass
    elif isinstance(is_spc, int):
        is_spc = bool(is_spc)
    elif isinstance(is_spc, str):
        is_spc = is_spc.lower().strip() in ["true", "yes", "1"]
    else:
        is_spc = None

    # Handle SPC_type string mapping and validation
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

    # Handle Evidence_level case-insensitively
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

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row["patient_id"]
        if pid in done:
            continue

        start = time.time()
        tumors, extract_raw = extract(row["timeline_text"])
        final, reason_raw = reason(tumors)

        results.append({
            "patient_id": pid,
            "is_spc": final["is_spc"],
            "SPC_type": final["SPC_type"],
            "Evidence_level": final["Evidence_level"],
            "reason": final["reason"],
            "extracted_tumors": json.dumps(tumors, ensure_ascii=False),
            "extract_raw_output": extract_raw,
            "raw_output": reason_raw,
            "time_sec": round(time.time() - start, 2)
        })

        # Atomic write
        tmp = OUTPUT_FILE + ".tmp"
        pd.DataFrame(results).to_csv(tmp, index=False)
        os.replace(tmp, OUTPUT_FILE)

    print(f"\n{'=' * 50}")
    print(f" Done: {len(results)} patients")
    print(f" Output: {OUTPUT_FILE}")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    main()

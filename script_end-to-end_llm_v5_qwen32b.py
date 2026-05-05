import pandas as pd
import requests
import json
import time
import os
import re
from datetime import datetime
from tqdm import tqdm

# =========================
# Config
# =========================
API_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:32b"

INPUT_FILE = "LLM_input_timeline_annotated_anonymized.csv"
OUTPUT_FILE = "output_llm_only_v5_32b.csv"
ERROR_LOG = "error_log_llm_only_v5_32b.txt"

MAX_RETRIES = 3
TIMEOUT = 720


# =========================
# Prompt — Pure Reasoning (baseline, no clinical rules)
# =========================
def build_prompt(text):
    return f"""You are an experienced oncology clinician.

Based on the following longitudinal clinical records, determine whether the patient has a second primary cancer (SPC).

Use clinical reasoning to distinguish:
- Metastasis: NOT SPC
- Recurrence: NOT SPC
- Second primary cancer: IS SPC

SPC_type:
0 = non-SPC
1 = metachronous (>6 months)
2 = synchronous (≤6 months)
3 = SPC with indeterminate timing

Evidence_level:
A = pathology
B = clinical/imaging
C = uncertain

Return ONLY valid JSON. No explanation outside JSON.
{{"is_spc": true/false, "SPC_type": 0/1/2/3, "Evidence_level": "A"/"B"/"C", "reason": ""}}

Clinical records:
{text}
"""


# =========================
# LLM
# =========================
def call_llm(prompt):

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL,
                json={
                    "model": MODEL,
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

        except Exception as e:
            log_error(f"[{attempt}] {str(e)}")

        time.sleep(3)

    return None


# =========================
# JSON 清洗
# =========================
def extract_valid_json(raw):

    if raw is None:
        return None

    # 1. 去掉 <think
    raw = re.sub(r"<think[\s\S]*?</think\s*>", "", raw, flags=re.DOTALL)

    # 2. 修复 True/False
    raw = raw.replace("True", "true").replace("False", "false")

    # 3. 找所有 JSON 块（非贪婪）
    matches = re.findall(r"\{[\s\S]*?\}", raw)

    # 4. 返回第一个合法 JSON
    for m in matches:
        try:
            json.loads(m)
            return m
        except:
            continue

    return None


# =========================
# 解析
# =========================
def parse_output(raw):

    json_str = extract_valid_json(raw)

    if not json_str:
        return {
            "is_spc": None,
            "SPC_type": None,
            "Evidence_level": None,
            "reason": "parse_error"
        }

    try:
        data = json.loads(json_str)
    except:
        return {
            "is_spc": None,
            "SPC_type": None,
            "Evidence_level": None,
            "reason": "json_error"
        }

    # 标准化
    is_spc = data.get("is_spc")
    if isinstance(is_spc, str):
        is_spc = is_spc.lower() in ["true", "yes", "1"]

    spc_type = data.get("SPC_type")
    if isinstance(spc_type, str):
        try:
            spc_type = int(spc_type)
        except:
            spc_type = None

    evidence = data.get("Evidence_level")
    if isinstance(evidence, str):
        evidence = evidence.upper().strip()
    if evidence not in ["A", "B", "C"]:
        evidence = None

    return {
        "is_spc": is_spc if isinstance(is_spc, bool) else None,
        "SPC_type": spc_type if spc_type in [0, 1, 2, 3] else None,
        "Evidence_level": evidence,
        "reason": data.get("reason", "")
    }


# =========================
# Utils
# =========================
def log_error(msg):
    with open(ERROR_LOG, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")


def load_existing():
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            if "patient_id" in df.columns:
                return df, set(df["patient_id"])
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass
    return pd.DataFrame(), set()


# =========================
# Main
# =========================
def main():

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    existing, done = load_existing()
    results = existing.to_dict("records")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        pid = row["patient_id"]
        if pid in done:
            continue

        start = time.time()

        raw = call_llm(build_prompt(row["timeline_text"]))

        if raw is None:
            parsed = {
                "is_spc": None,
                "SPC_type": None,
                "Evidence_level": None,
                "reason": "LLM_FAIL"
            }
        else:
            parsed = parse_output(raw)

        results.append({
            "patient_id": pid,
            "is_spc": parsed["is_spc"],
            "SPC_type": parsed["SPC_type"],
            "Evidence_level": parsed["Evidence_level"],
            "reason": parsed["reason"],
            "raw_output": raw if raw else "",
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

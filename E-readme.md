# git_SPC_LLM

Batch processing of longitudinal clinical texts for identifying Second Primary Cancer (SPC) using large language models.

## Project Structure

This project includes three groups of scripts to evaluate SPC identification from different architectural and model perspectives.

### Main Scripts (Three Architectures)

| Script | Architecture |
|--------|-------------|
| `script_rule-based_spc_v5.py` | Rule-based architecture |
| `script_hybrid_v5.py` | Hybrid architecture (Rule + LLM) |
| `script_end-to-end_llm_v5.py` | End-to-end LLM architecture |

All three architectures process the same dataset to compare SPC identification performance.

### Ablation Study

Testing whether adding rule constraints to the reasoning process improves SPC identification accuracy:

- `script_2_hybrid_base_v5.py`
- `script_3_hybrid_hard_rule_v5.py`
- `script_4_hybrid_soft_rule_v5.py`
- `script_5_hybrid_iterative_v5.py`

### Cross-Model Comparison

Testing whether performance improvements from different model sizes are consistent across architectures:

- `script_end-to-end_llm_v5_qwen14b.py`
- `script_end-to-end_llm_v5_qwen32b.py`
- `script_hybrid_v5_qwen14b.py`
- `script_hybrid_v5_qwen32b.py`

## Requirements

- Python 3
- [Ollama](https://ollama.com/) (for running LLMs locally)
- RAM **≥ 32GB**

## Installation

```bash
pip install pandas openpyxl requests tqdm
```
## Usage
1. Prepare Input File
Input must be an Excel file (.xlsx) with the following headers:

| Column | Description |
|--------|-------------|
| `patient_id` | Unique patient identifier |
| `timeline_text` | Longitudinal clinical text records of the patient |

2. Start Ollama
```bash
ollama serve
```

Ensure the required models are downloaded, for example:

```bash
ollama pull deepseek-r1:32b
ollama pull qwen2.5:32b
```

3. Configure the Script
Open the script you want to run and modify:

Input file path: Replace with the path to your Excel file
Model name: Update according to the model name in your local Ollama (e.g., qwen2.5:14b)

4. Run the Script
```bash
python script_name.py
```
Example:

```bash
python script_hybrid_v5.py
```


## Notes
Ensure Ollama is running and models are downloaded before executing scripts

When processing large datasets, close other memory-intensive applications to ensure sufficient RAM

Different scripts may use different model names; check and update each script before running

The Excel file must contain both patient_id and timeline_text columns, otherwise an error will occur

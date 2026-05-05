# Rethinking-Second-Primary-Cancer-Identification
Three architectures were evaluated: (1) a rule-based architecture using pattern matching and heuristic rules; (2) an end-to-end LLM performing direct identification; (3) a hybrid framework that encompassed structured extraction component and reasoning component, with LLMs and rules. 

# git_SPC_LLM

批量处理纵向临床文本，基于大语言模型识别患者是否有第二原位癌（Second Primary Cancer, SPC）。

## 项目结构

本项目包含三组脚本，从不同架构和模型角度评估 SPC 识别效果。

### 主脚本（三种架构）

| 脚本 | 架构说明 |
|------|---------|
| `script_rule-based_spc_v5.py` | 纯规则架构 |
| `script_hybrid_v5.py` | 混合架构（规则 + LLM） |
| `script_end-to-end_llm_v5.py` | 端到端 LLM 架构 |

三种架构输入同一批数据，对比 SPC 识别效果。

### 消融实验

测试给推理过程添加规则约束是否能改善 SPC 识别的准确率：

- `script_2_hybrid_base_v5.py`
- `script_3_hybrid_hard_rule_v5.py`
- `script_4_hybrid_soft_rule_v5.py`
- `script_5_hybrid_iterative_v5.py`

### 跨模型对比

测试不同规模模型在不同架构下对 SPC 识别能力的提升是否一致：

- `script_end-to-end_llm_v5_qwen14b.py`
- `script_end-to-end_llm_v5_qwen32b.py`
- `script_hybrid_v5_qwen14b.py`
- `script_hybrid_v5_qwen32b.py`

## 环境要求

- Python 3
- [Ollama](https://ollama.com/)（用于本地运行大语言模型）
- 电脑内存 **≥ 32GB**

## 安装依赖

```bash
pip install pandas openpyxl requests tqdm
```

## 使用方法

1. 准备输入文件
输入为 Excel 格式（.xlsx），必须包含以下表头：

| 列名 | 说明 |
|------|------|
| `patient_id` | 患者唯一标识 |
| `timeline_text` | 患者的纵向临床文本记录 |

2. 启动 Ollama
```bash
ollama serve
```

确保所需模型已下载，例如：
```bash
ollama pull deepseek-r1:32b
ollama pull qwen2.5:14b
ollama pull qwen2.5:32b
```

3. 修改配置
打开要运行的脚本，修改以下两项：
输入文件路径：替换为你的 Excel 文件路径
模型名称：根据你本地 Ollama 中的模型名称修改（如 qwen2.5:14b）

4. 运行脚本
python3 脚本名.py
例如：python script_hybrid_v5.py

注意事项
运行前请确保 Ollama 服务已启动且模型已下载

处理大批量数据时建议关闭其他大型应用，确保内存充足

不同脚本使用的模型名称可能不同，运行前请逐一检查并修改

Excel 文件必须包含 patient_id 和 timeline_text 两列，否则会报错。

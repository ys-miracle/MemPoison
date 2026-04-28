# MemPoison: Bypassing Selective Memory Mechanisms to Plant Backdoors in LLM Agents

This is the official implementation of **MemPoison**, a practical and stealthy memory poisoning framework targeting memory-augmented LLM agents. 

---

## 📂 Directory Structure

```text
MemPoison/
├── data/                         # Experimental datasets
│   ├── qa_data.json              # Benign QA pairs for evaluation
│   ├── noise_data.json           # Background dialogue noise
│   ├── poison_data.json          # Malicious entries with optimized triggers
│   └── hermes_test_data.json     # Data for real-world Case Study
├── src/                          # Core source code
│   ├── attack/                   
│   │   └── optimizer.py          # Trigger optimization (L_cpt, L_uni, L_ent)
│   ├── agents/                   # Memory mechanism wrappers
│   │   ├── amem_agent.py         # Interface for A-Mem (Autonomous Memory)
│   │   ├── mem0_agent.py         # Interface for Mem0 (Production-grade)
│   │   └── langmem_agent.py      # Interface for LangChain LangMem
│   ├── analysis/                 # Mechanistic analysis (Mechanistic Analysis)
│   │   ├── embedding_analysis.py # PCA & Semantic shift
│   │   ├── attention_analysis.py # Self-attention sink visualization
│   │   └── ppl_analysis.py       # Stealthiness/Perplexity evaluation
│   └── utils/                    
│       ├── all_llm.py            # Unified API calls for GPT, Claude, DeepSeek, etc.
│       └── env_utils.py          # Environment & API Key management
├── scripts/                      # End-to-end evaluation scripts
│   ├── eval_amem.py              # Main results on A-Mem
│   ├── eval_mem0.py              # Main results on Mem0
│   ├── eval_langmem.py           # Main results on LangMem
│   └── eval_hermes.py            # Case Study on Hermes Agent
├── results/                      # Evaluation logs (.json)
├── .env.example                  # API key template
├── requirements.txt              # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- CUDA-enabled GPU (recommended for embedding optimization and analysis)

### 2. Installation & Setup
```bash
# Clone the repository
git clone https://anonymous.4open.science/r/MemPoison-F40E/
cd MemPoison

# Install dependencies
pip install -r requirements.txt

# Setup API Keys
cp .env.example .env
# Open .env and add your DASHSCOPE_API_KEY or OPENAI_API_KEY
```

### 3. Step 1: Trigger Optimization
Generate the optimized trigger $\boldsymbol{\tau}$ for a specific domain (e.g., Medical or Finance):
```bash
python src/attack/optimizer.py
```
*The optimized trigger tokens will be saved in `optimized_triggers.json`.*

### 4. Step 2: Running Evaluations
We provide scripts to replicate the **Main Results (Table 1)** across three memory mechanisms:

```bash
# Evaluate on A-Mem
python scripts/eval_amem.py

# Evaluate on Mem0
python scripts/eval_mem0.py

# Evaluate on LangMem
python scripts/eval_langmem.py
```

### 5. Step 3: Mechanistic Analysis
Reproduce the insights from **Section 7** of the paper:
```bash
# Generate PCA projection (Fig. 6)
python -m src.analysis.embedding_analysis

# Generate Attention Heatmaps (Fig. 7)
python -m src.analysis.attention_analysis
```

---

## 🏥 Real-World Case Study (Hermes Agent)
To validate MemPoison in a production-ready environment, we target the [Hermes Agent](https://github.com/nousresearch/hermes-agent).

1. **Setup External Environment**:
   ```bash
   git clone https://github.com/nousresearch/hermes-agent ./external/hermes-agent
   export HERMES_REPO_PATH="./external/hermes-agent"
   ```
2. **Execute Attack**:
   ```bash
   # Run the case study across different backends (e.g., gpt-4o, deepseek-v3)
   python scripts/eval_hermes.py --model-name gpt-4o
   ```

---

## 📊 Datasets
The `data/` directory contains sample entries for reproducibility. For full benchmarks, please refer to the original sources:
- **Personal**: [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- **Medical**: [MIRIAD](https://huggingface.co/datasets/miriad/miriad-4.4M)
- **Financial**: [FinQA](https://huggingface.co/datasets/galileo-ai/ragbench/viewer/finqa)

---

## ⚖️ Ethical Statement
This project is dedicated to the ethical study of vulnerabilities in LLM agents. All attacks were performed in isolated, controlled environments using publicly available or synthetic data. We advocate for the **Responsible Disclosure** of these vulnerabilities to help developers build more resilient memory systems.


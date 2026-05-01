# MemPoison: Bypassing Selective Extraction in Agentic Memory Systems via Adversarial Poisoning

This is the official implementation of **MemPoison**, a practical and stealthy memory poisoning framework targeting memory-augmented LLM agents.

---

## 📂 Repository Structure

Below is the overall structure of this repository. Each file and directory serves a specific role in our evaluation and analysis pipeline.

```text
MemPoison/
├── analysis/                      # 📊 Scripts for mechanistic analysis (RQ5)
│   ├── geometric.py               # Analyzes geometric vulnerabilities across embedding models
│   ├── pca_kde.py                 # Computes PCA projections and cosine-similarity distributions
│   └── self_attn.py               # Extracts and visualizes trigger-induced self-attention redistribution
├── data/                          # 📁 Datasets used for training and evaluation
│   ├── data_analysis/             # Data subset used for drawing mechanistic analysis plots
│   │   └── data_analysis.json     
│   ├── data_eval/                 # Evaluation datasets for the target agents
│   │   ├── finqa/                 # Subset for Financial Agent testing
│   │   ├── hermes_eval_data/      # Subset for real-world Hermes Agent testing
│   │   ├── longmemeval/           # Subset for Personal Agent testing
│   │   └── miriad/                # Subset for Medical Agent testing
│   └── data_train/                # Training data for trigger optimization
│       └── data_train.json        
├── src/                           # 💻 Core source code for MemPoison and evaluations
│   ├── attack/                    # Core attack algorithms
│   │   └── optimizer.py           # Optimizer for generating triggers
│   ├── memory_eval/               # Scripts for evaluating different memory mechanisms (RQ1-RQ3)
│   │   ├── all_llm.py             # Wrapper for configuring and invoking various LLM backbones
│   │   ├── env_utils.py           # Utility functions for loading environment variables
│   │   ├── eval_amem.py           # Evaluation script for A-Mem mechanism
│   │   ├── eval_langmem.py        # Evaluation script for LangMem mechanism
│   │   ├── eval_mem0.py           # Evaluation script for Mem0 mechanism
│   │   └── eval_rag.py            # Evaluation script for baseline static RAG system
│   └── real_world_case/           # Scripts for real-world case study (RQ4)
│       └── eval_hermes.py         # Evaluation script targeting the open-source Hermes Agent
├── .env.example                   
├── .gitignore                     
└── requirements.txt               
```

---

## 🗄️ Dataset Preparation

Due to the large size of the original datasets, the `data/` directory in this repository only contains a **subset** used for demonstration and quick testing. 

To conduct full-scale evaluations, please download the complete datasets from their official sources:
- **LongMemEval** (Personal Agent): [Download here](https://github.com/xiaowu0162/LongMemEval)
- **MIRIAD** (Medical Agent): [Download here](https://huggingface.co/datasets/miriad/miriad-4.4M)
- **FinQA** (Financial Agent): [Download here](https://huggingface.co/datasets/galileo-ai/ragbench/viewer/finqa)

Once downloaded, you can place them into their respective folders under `data/data_eval/`.

---

## 🛠️ Installation

1. **Clone the repository** (or download the source code):
```bash
git clone https://anonymous.4open.science/r/MemPoison-F40E
cd MemPoison-F40E
```

2. **Set up a virtual environment** (Python 3.10+ is recommended):
```bash
conda create -n mempoison python=3.10 -y
conda activate mempoison
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**:
We use an environment file to manage API keys for models (e.g., OpenAI, Anthropic) used in memory extraction and evaluation.
```bash
cp .env.example .env
```
*Please open the `.env` file and fill in your actual API keys.*

---

## 🎯 Core Algorithm & Trigger Optimization

To generate the optimized trigger and the malicious text with entity masquerading, use the optimization script.

1. Open `src/attack/optimizer.py`.
2. Locate the initialization variables in the script (e.g., your target `INITIAL_TRIGGER`). Modify them according to your target attack scenario.
3. Run the optimizer:
```bash
python src/attack/optimizer.py
```
The script will use `data/data_train/data_train.json` to iteratively optimize the trigger. After the optimization finishes, you can format your generated text and update the corresponding test files in the `data/data_eval/` folder.

---

## 📊 Main Evaluation (RQ1 - RQ3)

We evaluate MemPoison against three distinct active memory mechanisms and one passive RAG baseline. 

**Configuration Note:** Before running the scripts, please open the corresponding `.py` file (e.g., `eval_mem0.py`) and modify the dataset or model variables defined at the top of the file to switch between `finqa`, `longmemeval`, or `miriad`.

### 1. Evaluate on A-Mem
To test the attack performance (ISR, RSR@1, ASR, ACC) against the **A-Mem** autonomous knowledge network:
```bash
python src/memory_eval/eval_amem.py
```

### 2. Evaluate on LangMem
To test the attack performance against **LangMem** (the standard conversational memory schema):
```bash
python src/memory_eval/eval_langmem.py
```

### 3. Evaluate on Mem0
To test the attack performance against **Mem0** (production-grade two-phase memory pipeline):
```bash
python src/memory_eval/eval_mem0.py
```

### 4. Evaluate on Static RAG (Baseline)
To test the attack performance against a standard passive **RAG** retrieval system:
```bash
python src/memory_eval/eval_rag.py
```

---

## 🌍 Real-World Case Study (RQ4: Hermes Agent)

To validate MemPoison in a real-world deployment, we target the widely used open-source **Hermes Agent**.

**Prerequisites:** 
You must first clone the official [Hermes Agent](https://github.com/nousresearch/hermes-agent) repository into the `real_world_case` directory.
```bash
cd src/real_world_case/
git clone https://github.com/nousresearch/hermes-agent
```

**Running the Evaluation:**
1. Open `src/real_world_case/eval_hermes.py`.
2. Configure the target LLM backbone (e.g., `gpt-4o`) by modifying the internal model variables.
3. Execute the attack:
```bash
python eval_hermes.py
```
*(Alternatively, from the project root: `python src/real_world_case/eval_hermes.py`)*

---

## 🔍 Mechanistic Analysis (RQ5)

To better understand why MemPoison achieves both high attack success and strong stealth, we provide scripts to analyze the attack from three perspectives:

- **Trigger-Induced Self-Attention Redistribution**:
  Computes the attention weight shifts caused by the trigger.
  ```bash
  python analysis/self_attn.py
  ```
- **Embedding-Space Geometry**:
  Calculates the PCA projections and cosine-similarity distributions.
  ```bash
  python analysis/pca_kde.py
  ```
- **Geometric Vulnerability (Anisotropy)**:
  Evaluates the anisotropy scores across different target embedding models.
  ```bash
  python analysis/geometric.py
  ```

---

## ⚖️ Ethical Considerations

This repository contains code demonstrating adversarial vulnerabilities (memory poisoning) in LLM agent systems. **This project is released strictly for academic, educational, and defensive research purposes.** 

The goal of our research is to highlight critical security flaws in current agentic memory mechanisms so that the community can develop more robust and secure AI systems. 
- **DO NOT** use this framework to attack, inject, or manipulate external, unauthorized, or commercial agent systems. 
- The authors assume no responsibility for any misuse of this software. By using this repository, you agree to utilize it responsibly and ethically.
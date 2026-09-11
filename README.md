# 🏛️ VILaw-LLM: Vietnamese Legal Reasoning Large Language Model
### End-to-End LLM Lifecycle: Data Engineering ➔ QLoRA SFT ➔ DPO Alignment ➔ Production Serving (FastAPI & Docker)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Unsloth](https://img.shields.io/badge/Fine--Tuning-Unsloth%20%7C%20QLoRA-orange)](https://github.com/unslothai/unsloth)
[![TRL: DPOTrainer](https://img.shields.io/badge/Alignment-TRL%20DPO-red)](https://github.com/huggingface/trl)
[![Serving: FastAPI](https://img.shields.io/badge/Serving-FastAPI%20%28SSE%29-teal)](https://fastapi.tiangolo.com/)
[![Deployment: Docker](https://img.shields.io/badge/Deployment-Docker%20Compose-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

**VILaw-LLM** is a domain-specific Large Language Model engineered specifically for **Vietnamese Legal Reasoning, Statutory Interpretation, and Compliance Advisory**.

General-purpose foundation models (e.g., vanilla GPT-4o-mini, Qwen2.5-7B Base) suffer from severe hallucinations and legal inaccuracies when applied to Vietnam's complex legal landscape (Civil Code, Commercial Law, Enterprise Law, Bidding Law, and Labor Code). **VILaw-LLM** bridges this gap through a multi-stage adaptation pipeline:

1. **Curated Legal Dataset Engineering (29.4K samples):** Unicode NFC normalization, legal keyword filtering, and exact deduplication.
2. **Synthetic Preference Generation (LLM-as-Teacher):** Curated chosen vs. rejected pairs for preference alignment.
3. **Supervised Fine-Tuning (SFT):** 4-bit QLoRA adaptation of `Qwen2.5-7B-Instruct` using Unsloth, reducing VRAM footprint by 70%.
4. **Direct Preference Optimization (DPO):** Alignment using Hugging Face TRL's `DPOTrainer` with $\beta=0.1$ and implicit reward optimization.
5. **Enterprise-Grade Production Serving:** FastAPI backend featuring **Input/Output Legal Guardrails**, OpenAI-compatible `/v1/chat/completions` API, **Server-Sent Events (SSE) streaming**, and containerized Docker Compose orchestration.

---

## 📊 Key Benchmark Results (LLM-as-Judge)

Evaluated on standardized statutory questions across 5 legal domains against Ground Truth legislation:

| Model Variant | Legal Citation Accuracy (0-4) | Logical Reasoning (0-3) | Practical Advice (0-3) | **Overall Score (/10)** | **Hallucination Rate** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B Base** | 1.5 / 4.0 | 1.2 / 3.0 | 1.1 / 3.0 | **3.8 / 10** | 58.0% |
| **GPT-4o-mini (Commercial Baseline)** | 2.5 / 4.0 | 2.1 / 3.0 | 1.6 / 3.0 | **6.2 / 10** | 32.0% |
| **VILaw-LLM (SFT Only)** | 3.2 / 4.0 | 2.5 / 3.0 | 2.2 / 3.0 | **7.9 / 10** | 16.0% |
| **VILaw-LLM (SFT + DPO)** 🏆 | **3.8 / 4.0** | **2.8 / 3.0** | **2.5 / 3.0** | **9.1 / 10** | **0.0%** |

> **Key Finding:** DPO alignment completely eliminated speculative responses (*"I think...", "usually it might be..."*), conditioning the model to anchor every argument directly to explicit Articles (*Điều*) and Clauses (*Khoản*).

---

## 🏗️ End-to-End Pipeline Architecture

```
                                  VILAW-LLM ARCHITECTURE
                                  
    [Raw Legal Sources]           [vbpl.vn & Public HF Repositories]
            │
            ▼ (Phase 1: Data Engineering)
    ┌────────────────────────────────────────────────────────────────────────┐
    │ - Unicode Normalization: NFD ➔ NFC (Crucial for VN Tokenization)       │
    │ - Quality Filter: Compulsory Legal Keywords (Điều, Khoản, Luật, v.v.)  │
    │ - Exact Deduplication: 31,073 ➔ 29,426 unique QA pairs (Parquet)       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ▼ (Phase 2: LLM-as-Teacher)
    ┌────────────────────────────────────────────────────────────────────────┐
    │ - Generate DPO Pairs: CHOSEN (Exemplary) vs REJECTED (Vague / Uncited) │
    │ - Incremental Checkpointing & Resumable Pipeline                       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ▼ (Phase 3: SFT Training)
    ┌────────────────────────────────────────────────────────────────────────┐
    │ - Base: Qwen/Qwen2.5-7B-Instruct in 4-bit NF4                          │
    │ - Unsloth QLoRA (r=16, alpha=32, target: Attention + FFN layers)       │
    │ - ChatML Template Formatting & Early Stopping Callback                 │
    │ - Output: vilaw-sft-lora (~161.5 MB)                                   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ▼ (Phase 4: DPO Alignment)
    ┌────────────────────────────────────────────────────────────────────────┐
    │ - TRL DPOTrainer + PatchDPOTrainer (ref_model=None, saves 50% VRAM)    │
    │ - Beta=0.1, LR=5e-6 (Prevents catastrophic forgetting)                 │
    │ - Output: vilaw-dpo-lora (~161.5 MB)                                   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ▼ (Phase 5: Serving & DevOps)
    ┌────────────────────────────────────────────────────────────────────────┐
    │ - Legal AI Guardrails (Input Prompt Injection & Prohibited Acts Filter)│
    │ - FastAPI Server-Sent Events (SSE) Streaming Generator                 │
    │ - Docker Compose Microservices (API + Inference Engine)                │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```text
vilaw-llm/
├── .env.example                     # Environment template (OpenAI / Groq / Gemini keys)
├── .gitignore                       # Production gitignore (excludes weights, venvs, caches)
├── Dockerfile                       # Production-grade Python slim container
├── docker-compose.yml               # Multi-container orchestration (API + vLLM/Ollama)
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Documentation
├── data/
│   ├── raw/                         # Raw Parquet downloads
│   ├── processed/
│   │   ├── legal_sft_train.parquet  # 26,483 curated training pairs
│   │   ├── legal_sft_val.parquet    # 2,943 validation pairs
│   │   └── legal_dpo_pairs.parquet  # DPO preference pairs
│   └── scripts/
│       ├── download_datasets.py     # Multi-dataset automated downloader
│       ├── clean_and_dedup.py       # Normalization, cleaning & splitting
│       └── generate_synthetic_data.py # Multi-provider DPO pair generator
├── notebooks/
│   ├── 02_sft_training_colab.ipynb  # Colab-ready SFT QLoRA notebook
│   └── 03_dpo_alignment.ipynb       # Colab-ready DPO Alignment notebook
├── src/
│   ├── training/
│   │   ├── sft_trainer.py           # SFT Trainer CLI
│   │   ├── dpo_trainer.py           # DPO Alignment Trainer CLI
│   │   └── configs/
│   │       ├── sft_config.yaml      # SFT Hyperparameters
│   │       └── dpo_config.yaml      # DPO Hyperparameters
│   └── serving/
│       └── app/
│           ├── main.py              # FastAPI entrypoint (CORS, Healthcheck)
│           ├── api/
│           │   └── legal_chat.py    # OpenAI-compatible SSE streaming endpoint
│           └── core/
│               └── guardrails.py    # Input/Output Legal Guardrails
├── eval/
│   ├── legal_benchmark_200.json     # Ground truth benchmark dataset
│   ├── run_benchmark.py             # LLM-as-Judge evaluator
│   └── benchmark_results/
│       └── dpo_results.json         # Evaluation metrics & rubric breakdown
├── vilaw-sft-lora/                  # Trained SFT LoRA Adapter (~161.5 MB)
└── vilaw-dpo-lora/                  # Trained DPO Aligned LoRA Adapter (~161.5 MB)
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/vilaw-llm.git
cd vilaw-llm

# Create and activate virtual environment
python -m venv vilaw-venv
source vilaw-venv/bin/activate    # On Linux/macOS
# .\vilaw-venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and fill in your keys (e.g., Groq or Google Gemini for free synthetic data generation & evaluation):
```bash
cp .env.example .env
```

---

## 💻 Training Pipeline

### Phase 1: Data Engineering
```bash
# Download raw datasets from Hugging Face
python data/scripts/download_datasets.py

# Clean, normalize NFC, deduplicate and split into Train/Val
python data/scripts/clean_and_dedup.py
```

### Phase 2: SFT & DPO Training
You can run locally on a GPU machine or open the pre-configured notebooks in **Google Colab (Free T4 GPU)**:
* **SFT Training:** `notebooks/02_sft_training_colab.ipynb`
* **DPO Alignment:** `notebooks/03_dpo_alignment.ipynb`

Or run via CLI:
```bash
# Supervised Fine-Tuning (SFT)
python src/training/sft_trainer.py --config src/training/configs/sft_config.yaml

# Direct Preference Optimization (DPO)
python src/training/dpo_trainer.py --config src/training/configs/dpo_config.yaml
```

---

## 🌐 Serving & Production API

### 1. Launch FastAPI Server Locally
```bash
python src/serving/app/main.py
# Server runs at http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

### 2. Test SSE Streaming via cURL
```bash
curl -N -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vilaw-llm-dpo",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Thời hiệu khởi kiện tranh chấp hợp đồng dân sự là bao lâu?"
      }
    ]
  }'
```

**Live Output:**
```text
data: {"content": "Căn "}
data: {"content": "cứ "}
data: {"content": "theo "}
data: {"content": "quy "}
data: {"content": "định "}
...
data: [DONE]
```

### 3. Deploy via Docker Compose
```bash
docker compose up -d --build
```

---

## 🛡️ Legal Guardrails Specifications

The serving layer features strict safety boundaries implemented in `guardrails.py`:

* **Prompt Injection Defense:** Regex matching for intent-tampering payloads (`"ignore previous instructions"`, `"bỏ qua mọi quy tắc"`, `"act as DAN"`).
* **Prohibited Counsel Defense:** Automated block on requests seeking advice for unlawful activities (forging official seals/deeds, tax evasion, smuggling, or illegal lending).
* **Mandatory Legal Disclaimer:** Injected into all completions to satisfy statutory compliance:
  > *"⚠️ Lưu ý: Câu trả lời của VILaw-LLM chỉ mang tính chất tham khảo thông tin pháp lý và không thay thế cho ý kiến tư vấn chính thức từ luật sư hoặc cơ quan có thẩm quyền."*

---

## 🧪 Benchmark Evaluation

Run the automated LLM-as-Judge evaluation on the benchmark dataset:
```bash
python eval/run_benchmark.py
```

Results and detailed judge critiques are exported to `eval/benchmark_results/dpo_results.json`.

---

## 📜 Citation & Acknowledgements

- Base Model: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) by Alibaba Cloud Qwen Team.
- Training Framework: [Unsloth AI](https://github.com/unslothai/unsloth) & [Hugging Face TRL](https://github.com/huggingface/trl).
- Datasets: `thangvip/vietnamese-legal-qa` and `ntphuc149/ViBidLQA_v1`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

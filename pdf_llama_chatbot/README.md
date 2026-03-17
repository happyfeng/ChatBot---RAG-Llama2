# RAG + Llama2 + LangChain Chatbot

Build a ChatGPT-style Q&A bot using your own PDFs as the knowledge base, with **RAG (Retrieval-Augmented Generation)** and optional **Llama2 fine-tuning**.

## Features

- **RAG chat**: Load PDFs, chunk, embed, and answer questions with Llama2 using retrieved passages
- **Fine-tuning**: Fine-tune Llama2 on your data with LoRA/QLoRA and use it with RAG

## Requirements

- Python 3.10+
- GPU/RAM: ≥8GB recommended for RAG with 7B model; ≥16GB for fine-tuning (QLoRA ~10GB)
- **Using Llama2 (Ollama recommended)**: Install [Ollama](https://ollama.com/) and pull a model (e.g. `ollama run llama2`), then set `USE_OLLAMA=true` and `OLLAMA_MODEL=llama2` in `.env`—no Hugging Face account or token needed. To use Hugging Face or local transformers instead, set `USE_OLLAMA=false` and configure `LLAMA2_MODEL` (for gated models you must accept the terms on [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and set `HF_TOKEN`)

## Installation

```bash
cd pdf_llama_chatbot
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set USE_OLLAMA=true and OLLAMA_MODEL=llama2 for Ollama; or set HF_TOKEN etc. for Hugging Face
```

## Usage

### 1. Add PDFs

Put your PDFs in the `pdfs/` directory (or the path set in `PDF_DIR` in `.env`).

### 2. Build the vector store (required for RAG)

```bash
python ingest.py
```

This reads PDFs from `pdfs/`, chunks them, embeds with the HuggingFace embedding model, and persists to Chroma (default: `vector_store/chroma/`).

### 3. Start RAG chat

```bash
python chat.py
```

Type questions in the terminal to get answers based on your PDFs.

**Using Ollama's Llama2 (recommended)**: Set `USE_OLLAMA=true` and `OLLAMA_MODEL=llama2` in `.env`, ensure Ollama is installed and running (e.g. `ollama run llama2`), then run `python chat.py`.

To use the **fine-tuned model** for RAG, set `USE_OLLAMA=false` and run `python chat.py --finetuned` (see "After fine-tuning" below).

**Using a local or Hugging Face Llama2** (e.g. `NousResearch/Llama-2-7b-chat-hf`, no gated access): run `python chat.py --model NousResearch/Llama-2-7b-chat-hf`; or set `USE_OLLAMA=false` and `LLAMA2_MODEL=NousResearch/Llama-2-7b-chat-hf` in `.env`, then run `python chat.py`.

### 4. Fine-tune Llama2 (optional)

Fine-tuning uses `finetune.py` (Hugging Face transformers + LoRA/QLoRA). You can run it **without** a Hugging Face account or gated access (see "Without Hugging Face" below).

**Prepare training data**: One JSON per line in `finetune_data/train.jsonl`, e.g.:

```json
{"instruction": "Your question", "output": "Expected answer"}
```

Or generate an initial dataset from your PDFs:

```bash
python pdf_to_finetune_data.py --output finetune_data/train.jsonl
```

Edit `train.jsonl` as needed for higher-quality Q&A pairs.

**Run fine-tuning**:

- **Without Hugging Face (recommended)**: No `HF_TOKEN` needed; use a community or local base model:
  ```bash
  # Option 1: Community model NousResearch/Llama-2-7b-chat-hf (no gated access)
  python finetune.py --model NousResearch/Llama-2-7b-chat-hf --epochs 3

  # Option 2: Local HF-format model directory (fully offline)
  python finetune.py --model /path/to/Llama-2-7b-chat-hf --epochs 3

  # Add --qlora to reduce VRAM (4-bit)
  python finetune.py --model NousResearch/Llama-2-7b-chat-hf --qlora --epochs 3
  ```
- **With Hugging Face**: To use the official gated model, accept the [HF terms](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and set `HF_TOKEN` in `.env`, then:
  ```bash
  python finetune.py --epochs 3
  # Or: python finetune.py --qlora --epochs 3
  ```

Output is saved to `finetuned_llama/` (or `--output`).

**After fine-tuning**:

1. **Use the fine-tuned model for RAG in this project**: The output is in Hugging Face format. Set **`USE_OLLAMA=false`** in `.env`, then run:
   ```bash
   python chat.py --finetuned
   ```
   This loads the model from `finetuned_llama/` for Q&A with RAG.
2. **Use the fine-tuned model in Ollama**: Convert `finetuned_llama/` to a format Ollama supports (e.g. GGUF) and import (e.g. `ollama create`). This project does not include that conversion; search for "Hugging Face to GGUF / import into Ollama".

## Project structure

```
pdf_llama_chatbot/
├── config.py              # Paths and model config
├── ingest.py              # PDF load, chunk, embed, vector store
├── chat.py                # RAG chat (Llama2 + retrieval)
├── finetune.py            # Llama2 LoRA/QLoRA fine-tuning
├── pdf_to_finetune_data.py # Generate fine-tune JSONL from PDFs
├── requirements.txt
├── .env.example
├── pdfs/                  # Your PDFs
├── vector_store/         # Vector store persistence
├── finetune_data/         # Fine-tune data (e.g. train.jsonl)
└── finetuned_llama/       # Fine-tuned model
```

## Configuration

In `.env` you can set:

| Variable | Description |
|----------|-------------|
| `USE_OLLAMA` | Set to `true` to use local Ollama instead of Hugging Face |
| `OLLAMA_MODEL` | Ollama model name (e.g. `llama2`, `llama3.2`); run `ollama run llama2` first |
| `OLLAMA_BASE_URL` | Ollama server URL, default `http://localhost:11434` |
| `PDF_DIR` | Directory containing PDFs |
| `LLAMA2_MODEL` | Hugging Face model id or local path (used when `USE_OLLAMA` is false) |
| `EMBEDDING_MODEL` | Embedding model for retrieval (HF id or local path) |
| `HF_HUB_OFFLINE` | Set to `1` when offline; load embeddings only from cache or local path |
| `HF_TOKEN` | Hugging Face token (required for gated models; leave empty when using Ollama or local model) |
| `FINETUNE_OUTPUT_DIR` / `FINETUNE_DATA_DIR` | Fine-tune output and data directories |

## Resource usage and GPU

| Component | Load | GPU required? | Notes |
|-----------|------|----------------|-------|
| **chat.py – Llama2 (Hugging Face path)** | High | **Recommended** | With `USE_OLLAMA=false`, 7B load + inference ~8–16GB VRAM; CPU is very slow. GPU uses `device_map="auto"`. |
| **chat.py – Ollama path** | Depends on Ollama host | Per Ollama | `chat.py` only sends HTTP requests; no VRAM in this process. If Ollama runs on same machine with GPU, inference uses GPU. |
| **finetune.py** | High | **Yes** | LoRA/QLoRA needs GPU (QLoRA ~10–16GB). CPU is not practical. |
| **ingest.py / chat.py – embeddings** | Medium | Optional | Default `device="cpu"`; can set `"cuda"` to speed up. |
| **PDF parsing, chunking, Chroma** | Low | No | CPU only. |

**Summary**: GPU is required or strongly recommended for **Hugging Face Llama2 inference** and **finetune.py**. The rest can run on CPU; switch to GPU for embeddings if you want faster runs.

## Notes

1. **Ollama recommended**: No Hugging Face account needed; install Ollama, run `ollama run llama2`, set `USE_OLLAMA=true` in `.env`. For Hugging Face: Llama2 is gated—accept terms and set `HF_TOKEN`; if you already have the model locally, set `LLAMA2_MODEL` to that path and `USE_OLLAMA=false`.
2. **Low VRAM**: Use 4-bit quantization or a smaller model; for fine-tuning use `--qlora`.
3. **CPU only**: RAG can run on CPU but inference is slow; fine-tuning should use GPU.
4. **Ollama**: Set `USE_OLLAMA=true` and `OLLAMA_MODEL=llama2` in `.env` and ensure Ollama is running (`ollama run llama2`). RAG chat works the same as with Hugging Face. For fine-tuning without HF, use `--model NousResearch/Llama-2-7b-chat-hf` or a local path; to use the fine-tuned model here set `USE_OLLAMA=false` and run `python chat.py --finetuned`. To run the fine-tuned model inside Ollama, convert to GGUF and import separately.
5. **Offline / no network**: Set `HF_HUB_OFFLINE=1` in `.env` and ensure the embedding model is available locally. Either **(a)** run `ingest.py` or load the embedding model once on a connected machine and copy `~/.cache/huggingface/` to the offline machine, or **(b)** on a connected machine run `huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir ./models/paraphrase-multilingual-MiniLM-L12-v2`, copy that directory to the offline machine, set `EMBEDDING_MODEL=./models/paraphrase-multilingual-MiniLM-L12-v2` (or absolute path) and `HF_HUB_OFFLINE=1` in `.env`.
6. **Fine-tune 401 / SSL / gated**: To avoid HF gated access and SSL issues, use a non-gated base model: `python finetune.py --model NousResearch/Llama-2-7b-chat-hf --epochs 3` (no `HF_TOKEN`). For fully offline fine-tuning, use a local HF-format directory: `python finetune.py --model /path/to/Llama-2-7b-chat-hf --epochs 3`. The script uses `local_files_only` for local paths.

---

## 中文 (Chinese)

# 基于 PDF + Llama2 + LangChain 的 Chatbot

用你自己的 PDF 文档作为知识库，通过 **RAG（检索增强生成）** 和可选的 **Llama2 微调**，搭建一个类似 ChatGPT 的问答机器人。

## 功能

- **RAG 对话**：从 PDF 加载、分块、向量化，用 Llama2 根据检索到的片段回答问题
- **微调（Fine-tuning）**：用 LoRA/QLoRA 在自建数据上微调 Llama2，再配合 RAG 使用

## 环境要求

- Python 3.10+
- 显存/内存：RAG 用 7B 模型建议 ≥8GB；微调建议 ≥16GB（QLoRA 可约 10GB）
- **使用 Llama2（推荐 Ollama）**：本机安装 [Ollama](https://ollama.com/) 并拉取模型（如 `ollama run llama2`），在 `.env` 中设置 `USE_OLLAMA=true`、`OLLAMA_MODEL=llama2` 即可，无需 Hugging Face 账号或 token。若改用 Hugging Face 或本地 transformers 模型，则设 `USE_OLLAMA=false` 并配置 `LLAMA2_MODEL`（从 HF 拉取 gated 模型时需在 [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 同意条款并填写 `HF_TOKEN`）

## 安装

```bash
cd pdf_llama_chatbot
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：用 Ollama 时设 USE_OLLAMA=true、OLLAMA_MODEL=llama2；用 Hugging Face 时填 HF_TOKEN 等
```

## 使用步骤

### 1. 放入 PDF

把要作为知识库的 PDF 放到 `pdfs/` 目录（或 `.env` 里配置的 `PDF_DIR`）。

### 2. 构建向量库（RAG 必做）

```bash
python ingest.py
```

会从 `pdfs/` 读取 PDF、分块、用 HuggingFace 嵌入模型做向量并存入 Chroma，默认持久化到 `vector_store/chroma/`。

### 3. 启动 RAG 对话

```bash
python chat.py
```

在终端输入问题即可基于 PDF 内容回答。

**使用 Ollama 的 Llama2（推荐）**：在 `.env` 中设置 `USE_OLLAMA=true`、`OLLAMA_MODEL=llama2`，确保本机已安装并运行 Ollama（如先执行 `ollama run llama2`），然后运行 `python chat.py`。

要用**微调后的模型**做 RAG 时，需设 `USE_OLLAMA=false`，然后运行 `python chat.py --finetuned`（见「微调完成后怎么用」）。

**使用本地或 Hugging Face 的 Llama2**（如 `NousResearch/Llama-2-7b-chat-hf`，无需 gated 审批）：可直接运行 `python chat.py --model NousResearch/Llama-2-7b-chat-hf`；或在 `.env` 中设 `USE_OLLAMA=false`、`LLAMA2_MODEL=NousResearch/Llama-2-7b-chat-hf` 后执行 `python chat.py`。

### 4. 微调 Llama2（可选）

微调使用本项目的 `finetune.py`（基于 Hugging Face transformers + LoRA/QLoRA），**不需要** HF 账号或 gated 权限也可进行（见下方「不用 Hugging Face 时」）。

**准备训练数据**：在 `finetune_data/train.jsonl` 中每行一个 JSON，例如：

```json
{"instruction": "你的问题", "output": "期望的回答"}
```

或直接使用「从 PDF 导出」的脚本生成初版数据：

```bash
python pdf_to_finetune_data.py --output finetune_data/train.jsonl
```

再按需编辑 `train.jsonl`，做成高质量的问答对。

**运行微调**：

- **不用 Hugging Face（推荐）**：不填 `HF_TOKEN`，用社区开放模型或本地模型作为基座，无需 gated 审批：
  ```bash
  # 方式一：使用社区开放模型 NousResearch/Llama-2-7b-chat-hf（无需 HF 审批）
  python finetune.py --model NousResearch/Llama-2-7b-chat-hf --epochs 3

  # 方式二：使用本机已下载的 HF 格式模型目录（完全离线）
  python finetune.py --model /path/to/Llama-2-7b-chat-hf --epochs 3

  # 显存紧张时可加 --qlora（4-bit 量化）
  python finetune.py --model NousResearch/Llama-2-7b-chat-hf --qlora --epochs 3
  ```
- **使用 Hugging Face**：若要用官方 gated 模型，需在 [HF 同意条款](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 并在 `.env` 中设置 `HF_TOKEN`，然后：
  ```bash
  python finetune.py --epochs 3
  # 或省显存：python finetune.py --qlora --epochs 3
  ```

微调结果会保存到 `finetuned_llama/`（或 `--output` 指定目录）。

**微调完成后怎么用**：

1. **在本项目里做 RAG 对话**：微调产出是 Hugging Face 格式，需用 transformers 加载。请将 `.env` 中 **`USE_OLLAMA` 设为 `false`**，然后执行：
   ```bash
   python chat.py --finetuned
   ```
   此时会从 `finetuned_llama/` 加载微调后的模型进行问答（与 RAG 检索一起使用）。
2. **若希望用 Ollama 跑微调后的模型**：需先把 `finetuned_llama/` 转为 Ollama 支持的格式（如 GGUF），再通过 `ollama create` 等导入；本项目不包含该转换步骤，可自行搜索「Hugging Face 模型转 GGUF / 导入 Ollama」。

## 项目结构

```
pdf_llama_chatbot/
├── config.py              # 路径与模型配置
├── ingest.py              # PDF 入库、分块、向量化
├── chat.py                # RAG 对话（Llama2 + 检索）
├── finetune.py            # Llama2 LoRA/QLoRA 微调
├── pdf_to_finetune_data.py # 从 PDF 生成微调用 JSONL
├── requirements.txt
├── .env.example
├── pdfs/                  # 放置你的 PDF
├── vector_store/          # 向量库持久化
├── finetune_data/         # 微调数据（如 train.jsonl）
└── finetuned_llama/       # 微调后的模型
```

## 配置说明

在 `.env` 中可配置：

| 变量 | 说明 |
|------|------|
| `USE_OLLAMA` | 设为 `true` 时用本机 Ollama 跑模型，不读 Hugging Face 模型 |
| `OLLAMA_MODEL` | Ollama 中的模型名（如 `llama2`、`llama3.2`），需先 `ollama run llama2` |
| `OLLAMA_BASE_URL` | Ollama 服务地址，默认 `http://localhost:11434` |
| `PDF_DIR` | PDF 所在目录 |
| `LLAMA2_MODEL` | Hugging Face 模型名或本地路径（仅当 `USE_OLLAMA` 为 false 时生效） |
| `EMBEDDING_MODEL` | 用于检索的嵌入模型（HF 模型名或本地目录路径） |
| `HF_HUB_OFFLINE` | 网络不可用时设为 `1`，仅从本地缓存/本地路径加载嵌入模型，不访问 Hugging Face |
| `HF_TOKEN` | Hugging Face 访问 token（从 HF 拉取 gated 模型时需要；用 Ollama 或本地路径时可留空） |
| `FINETUNE_OUTPUT_DIR` / `FINETUNE_DATA_DIR` | 微调输出与数据目录 |

## 注意事项

1. **推荐用 Ollama**：无需 Hugging Face 账号，本机安装 Ollama 后 `ollama run llama2`，在 `.env` 设 `USE_OLLAMA=true` 即可。若用 Hugging Face：Llama2 为 gated 模型，需在 HF 同意条款并设置 `HF_TOKEN`；本机已有模型时可把 `LLAMA2_MODEL` 设为本地路径并设 `USE_OLLAMA=false`。
2. **显存不足**：可改用 `meta-llama/Llama-2-7b-chat-hf` 的 4-bit 量化或更小模型；微调时用 `--qlora`。
3. **仅 CPU**：可在 CPU 上跑 RAG，但推理会较慢；微调建议使用 GPU。
4. **用 Ollama**：在 `.env` 中设 `USE_OLLAMA=true`、`OLLAMA_MODEL=llama2`，并保证本机已安装且运行 Ollama（`ollama run llama2`）。RAG 对话与 Hugging Face 方式一致。微调用 `finetune.py`（无需 HF 时可加 `--model NousResearch/Llama-2-7b-chat-hf` 或本地路径）；微调完成后在本项目内用需 `USE_OLLAMA=false` 且 `python chat.py --finetuned`，若要在 Ollama 里用需自行转 GGUF 再导入。

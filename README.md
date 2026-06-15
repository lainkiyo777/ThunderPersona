# ThunderPersona

RAG-backed WeChat persona workbench: export a chat history, build a searchable style memory, generate persona prompts, and optionally run DeepSeek or a local LoRA adapter behind a small browser UI.

> This repo contains code only. It intentionally excludes WeChat exports, vector indexes, local model weights, LoRA adapters, logs, and API keys.

## What It Does

- Cleans WeChat markdown exports into structured JSONL messages.
- Builds RAG chunks for BM25 retrieval and optional FAISS vector search.
- Generates a persona prompt from historical style statistics and retrieved chat snippets.
- Serves a local web workbench at `http://127.0.0.1:8765`.
- Supports answer backends:
  - `DeepSeek` via OpenAI-compatible chat completions.
  - `Local LoRA` via Qwen + PEFT adapter when local weights are available.
- Keeps browser-session context in `localStorage` for follow-up questions.

## Repository Layout

```text
persona_app/                 Local web UI and HTTP server
scripts/                     Data prep, RAG, vector index, service launcher
training/lora/               LoRA/QLoRA training and inference scripts
*.cmd                        Windows convenience wrappers
```

## Private Files Not Committed

These are generated locally and ignored by git:

```text
.wechat-exports/             Cleaned messages, RAG chunks, SFT data, prompts, vector indexes
.wechat-cli/                 WeChat extraction cache/config
.venv-wechat-cli/            Local Python environment
.hf-models/                  Downloaded base models
.hf-cache/                   Hugging Face cache
.persona-app*.log            Runtime logs
.env                         API keys
```

## Quick Start

Create a Python environment and install the core dependencies:

```powershell
python -m venv .venv-wechat-cli
.\.venv-wechat-cli\Scripts\python.exe -m pip install -r requirements.txt
```

Prepare your WeChat export into the expected local files:

```powershell
.\.venv-wechat-cli\Scripts\python.exe scripts\prepare_wechat_persona_dataset.py `
  --input .wechat-exports\your-chat.md `
  --output-prefix .wechat-exports\persona
```

Run the web workbench:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
.\start-persona-app.cmd --provider deepseek `
  --messages-file .wechat-exports\persona.messages.jsonl `
  --rag-file .wechat-exports\persona.rag_docs.jsonl
```

Open:

```text
http://127.0.0.1:8765
```

## Vector Search

Build a FAISS index after generating RAG chunks:

```powershell
.\.venv-wechat-cli\Scripts\python.exe scripts\wechat_vector_index.py build `
  --rag-file .wechat-exports\persona.rag_docs.jsonl `
  --index-dir .wechat-exports\vector_index\persona
```

Then start the app with:

```powershell
.\start-persona-app.cmd --provider deepseek `
  --messages-file .wechat-exports\persona.messages.jsonl `
  --rag-file .wechat-exports\persona.rag_docs.jsonl `
  --vector-index-dir .wechat-exports\vector_index\persona
```

## LoRA / QLoRA

Prepare SFT splits:

```powershell
.\.venv-wechat-cli\Scripts\python.exe scripts\prepare_lora_dataset.py `
  --input .wechat-exports\persona.sft.jsonl `
  --output-dir .wechat-exports\lora `
  --prefix persona
```

Train with the configs in `training/lora/`. Base models and adapters are intentionally kept outside git.

## Safety Notes

- Do not commit raw chat exports or generated persona data.
- Do not commit API keys. Use environment variables or a local `.env` file ignored by git.
- The generated persona is a writing aid, not the real person. Avoid using it to impersonate someone or mislead third parties.


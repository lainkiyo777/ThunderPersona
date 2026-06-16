# ThunderPersona

RAG-backed WeChat persona workbench: export a chat history, build a searchable style memory, generate persona prompts, and optionally run DeepSeek or a local LoRA adapter behind a small browser UI.

> This repo contains code only. It intentionally excludes WeChat exports, vector indexes, local model weights, LoRA adapters, logs, and API keys.

## What It Does

- Cleans WeChat markdown exports into structured JSONL messages.
- Builds RAG chunks for BM25 retrieval and optional FAISS vector search.
- Generates a persona prompt from historical style statistics, retrieved chat snippets, and a Persona DNA profile.
- Surfaces a Persona DNA panel with rhythm, voice, interaction style, phrase bank, anti-patterns, confidence, and honest boundaries.
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
  --target-name "Contact Name" `
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

## Contact Distillation

The browser app includes a contact distillation panel:

- `Index` refreshes `.wechat-exports/contact_index.json` from `wechat-cli contacts`. It stores contact names, remarks, aliases, and wxids only.
- Search matches names, remarks, aliases, and wxids with fuzzy matching.
- `Distill & Switch` exports only the selected chat with `wechat-cli export`, runs the same data prep pipeline, and switches the workbench to the new persona.
- Distillation runs as a background job with stage, progress percent, and elapsed-time polling in the browser.
- `Distilled Personas` lists generated personas and lets you switch without reloading every WeChat chat history.

Distillation levels:

- Light: up to 5,000 messages. Fast preview, lower style coverage.
- Medium: up to 20,000 messages. Recommended default for most contacts.
- Full: up to 200,000 messages and FAISS recommended. Best coverage, slowest run, largest disk use.

Generated contact indexes and persona data stay under `.wechat-exports/`, which is ignored by Git.

## Chat Sessions

The workbench keeps two chat states:

- Recent context: the last turns sent back into RAG generation for follow-up continuity.
- Current transcript: the full visible chat session for saving.

Use `New Chat` to start a clean conversation with the active persona. Use `Save Chat` to write the current transcript to `.wechat-exports/chat_sessions/` as both JSON and Markdown.

## Persona DNA

ThunderPersona borrows the strongest idea from high-quality skill distillation projects: do not only mimic surface wording. The app now extracts a compact profile before each prompt:

- Rhythm: average length, short-reply rate, and burst behavior.
- Voice: laugh markers, question rate, emoji rate, and punctuation intensity.
- Interaction: when to comfort, tease, ask back, or answer directly.
- Phrase bank: recurring short replies and口头禅 candidates.
- Anti-patterns: reply shapes to avoid, such as customer-service prose, AI analysis, over-stuffed catchphrases, and fabricated context.
- Honest boundaries: what the simulation should not infer or fabricate.

This is used both in the browser panel and inside generated prompts, so the model has a clearer operating contract instead of only seeing raw examples.

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


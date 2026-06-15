# Persona LoRA / QLoRA Training

This folder contains a portable fine-tuning scaffold for persona SFT data.

## Prepare Data

```powershell
$env:THUNDER_PERSONA_SLUG="persona"
.\prepare-lora-data.cmd
```

Outputs:

- `.wechat-exports/lora/persona.train.jsonl`
- `.wechat-exports/lora/persona.val.jsonl`
- `.wechat-exports/lora/persona.test.jsonl`

## Install Training Dependencies

The current venv has CPU-only PyTorch. For actual local training, install a CUDA-enabled PyTorch build first, then:

```powershell
.\.venv-wechat-cli\Scripts\python.exe -m pip install datasets peft trl accelerate
```

For QLoRA, `bitsandbytes` is also required and is more reliable on Linux/WSL than native Windows:

```powershell
.\.venv-wechat-cli\Scripts\python.exe -m pip install bitsandbytes
```

## Recommended Order

1. Start with `Qwen/Qwen2.5-1.5B-Instruct` LoRA.
2. If that works and VRAM is enough, try `Qwen/Qwen2.5-3B-Instruct` QLoRA.
3. Keep RAG enabled even after fine-tuning; LoRA improves tone, RAG improves memory.

## Commands

LoRA:

```powershell
.\train-qwen-lora.cmd
```

QLoRA:

```powershell
.\train-qwen-qlora.cmd
```

Your current GPU has 8GB VRAM. Native Windows QLoRA may be brittle; WSL/Linux or a cloud GPU is the safer path for 3B+ models.

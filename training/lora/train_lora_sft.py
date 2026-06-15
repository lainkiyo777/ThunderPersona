from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_training_deps():
    missing = []
    for package, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("accelerate", "accelerate"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + "\nInstall with: python -m pip install datasets peft trl accelerate"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT trainer for persona chat data.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate dependencies, CUDA, and config without loading a model.")
    args = parser.parse_args()
    require_training_deps()

    import torch
    import bitsandbytes
    import peft
    import trl
    import transformers
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available() and cfg.get("require_cuda", True):
        raise SystemExit(
            "CUDA is not available in this Python environment. "
            "Use a CUDA-enabled PyTorch install, WSL/Linux GPU env, or set require_cuda=false for a CPU smoke run."
        )

    for key in ("model_name", "train_file", "val_file", "output_dir"):
        if not cfg.get(key):
            raise SystemExit(f"Missing required config key: {key}")
    if not Path(cfg["train_file"]).is_file():
        raise SystemExit(f"Train file not found: {cfg['train_file']}")
    if not Path(cfg["val_file"]).is_file():
        raise SystemExit(f"Validation file not found: {cfg['val_file']}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "model_name": cfg["model_name"],
                    "load_in_4bit": bool(cfg.get("load_in_4bit")),
                    "train_file": cfg["train_file"],
                    "val_file": cfg["val_file"],
                    "output_dir": cfg["output_dir"],
                    "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                    "trl": trl.__version__,
                    "bitsandbytes": getattr(bitsandbytes, "__version__", "unknown"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.get("load_in_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quantization_config,
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "float16")),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_file"],
            "validation": cfg["val_file"],
        },
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    )

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        logging_steps=cfg.get("logging_steps", 10),
        eval_steps=cfg.get("eval_steps", 100),
        save_steps=cfg.get("save_steps", 100),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=cfg.get("save_total_limit", 3),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to=cfg.get("report_to", "none"),
        max_length=cfg.get("max_seq_length", 1024),
        do_train=True,
        do_eval=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick generation test with a persona LoRA adapter.")
    parser.add_argument("--base-model", default=".hf-models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=".wechat-exports/lora/output/qwen2.5-1.5b-persona-lora")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-new-tokens", default=80, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个私人的聊天风格模拟助手。只输出目标联系人风格的微信回复正文，"
                "不要解释，不要声称自己是真人。"
            ),
        },
        {"role": "user", "content": args.query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()

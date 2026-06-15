@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
"%ROOT%.venv-wechat-cli\Scripts\python.exe" "%ROOT%training\lora\train_lora_sft.py" --config "%ROOT%training\lora\qwen2_5_3b_qlora.json"

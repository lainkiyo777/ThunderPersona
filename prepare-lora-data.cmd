@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
if "%THUNDER_PERSONA_SLUG%"=="" set "THUNDER_PERSONA_SLUG=persona"
"%ROOT%.venv-wechat-cli\Scripts\python.exe" "%ROOT%scripts\prepare_lora_dataset.py" --input "%ROOT%.wechat-exports\%THUNDER_PERSONA_SLUG%.sft.jsonl" --output-dir "%ROOT%.wechat-exports\lora" --prefix "%THUNDER_PERSONA_SLUG%" %*

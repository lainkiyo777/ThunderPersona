@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
if "%THUNDER_PERSONA_SLUG%"=="" set "THUNDER_PERSONA_SLUG=persona"
"%ROOT%.venv-wechat-cli\Scripts\python.exe" "%ROOT%scripts\wechat_persona_rag_mvp.py" --messages-file "%ROOT%.wechat-exports\%THUNDER_PERSONA_SLUG%.messages.jsonl" --rag-file "%ROOT%.wechat-exports\%THUNDER_PERSONA_SLUG%.rag_docs.jsonl" %*

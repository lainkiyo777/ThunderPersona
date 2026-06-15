@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
if "%THUNDER_PERSONA_SLUG%"=="" set "THUNDER_PERSONA_SLUG=persona"
"%ROOT%.venv-wechat-cli\Scripts\python.exe" "%ROOT%scripts\wechat_vector_index.py" build --rag-file "%ROOT%.wechat-exports\%THUNDER_PERSONA_SLUG%.rag_docs.jsonl" --index-dir "%ROOT%.wechat-exports\vector_index\%THUNDER_PERSONA_SLUG%" %*

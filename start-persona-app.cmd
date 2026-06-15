@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
set "PYLAUNCHER=%ROOT%.venv-wechat-cli\Scripts\pythonw.exe"
if not exist "%PYLAUNCHER%" set "PYLAUNCHER=%ROOT%.venv-wechat-cli\Scripts\python.exe"
"%PYLAUNCHER%" "%ROOT%scripts\start_persona_service.py" %*

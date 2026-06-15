@echo off
set "PYTHONIOENCODING=utf-8"
set "ROOT=%~dp0"
"%ROOT%.venv-wechat-cli\Scripts\python.exe" "%ROOT%persona_app\server.py" %*

@echo off
set "USERPROFILE=%~dp0"
set "PYTHONIOENCODING=utf-8"
"%~dp0.venv-wechat-cli\Scripts\wechat-cli.exe" %*

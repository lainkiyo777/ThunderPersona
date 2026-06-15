@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Uri http://127.0.0.1:8765/api/shutdown -Method Post | Out-Null; Write-Output 'Persona app stopped.' } catch { Write-Output $_.Exception.Message }"

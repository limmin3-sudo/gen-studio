@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
  echo install.bat 을 먼저 실행하세요.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m genstudio %*
pause

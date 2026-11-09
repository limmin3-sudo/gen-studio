@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist ".git" (
  git pull --ff-only
) else (
  echo git으로 받은 폴더가 아니라서 코드 업데이트는 건너뜁니다. GitHub에서 새 ZIP을 받아 덮어쓰세요.
)
"%USERPROFILE%\.local\bin\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
".venv\Scripts\python.exe" scripts\download_models.py
pause

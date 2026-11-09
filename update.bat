@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Gen Studio 업데이트 (모델과 만든 이미지는 그대로 둡니다)
if exist ".git" (
  git pull --ff-only
) else (
  if not exist tools mkdir tools
  curl.exe -L --fail -o tools\update.zip https://github.com/limmin3-sudo/gen-studio/archive/refs/heads/main.zip
  if errorlevel 1 (
    echo 최신 코드를 받지 못했어요. 인터넷 연결을 확인하세요.
    pause
    exit /b 1
  )
  if exist tools\gen-studio-main rmdir /s /q tools\gen-studio-main
  tar -xf tools\update.zip -C tools
  robocopy tools\gen-studio-main . /E /XD .venv data outputs ComfyUI_windows_portable tools /NFL /NDL /NJH /NJS /NP
  rmdir /s /q tools\gen-studio-main
  del tools\update.zip
)
"%USERPROFILE%\.local\bin\uv.exe" pip install --python .venv\Scripts\python.exe -r requirements.txt
".venv\Scripts\python.exe" scripts\download_models.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\make_shortcut.ps1
echo.
echo 업데이트가 끝났어요. 이제 바탕화면의 Gen Studio 아이콘으로 실행하세요.
echo 앞으로 업데이트는 앱 안의 설정 화면에서 버튼으로 할 수 있어요.
pause

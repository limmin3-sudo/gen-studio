@echo off
rem 노트북 등 다른 기기에서도 쓸 수 있게 켭니다 (Tailscale 필요, docs\remote.md 참고)
call "%~dp0start.bat" --remote %*

# Gen Studio installer (run through install.bat)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what 실패 (코드 $LASTEXITCODE)" } }

Write-Host "Gen Studio 설치를 시작합니다. 처음에는 30분~1시간 걸릴 수 있어요." -ForegroundColor Green
Write-Host "중간에 멈추거나 창을 닫아도, 다시 실행하면 이어서 진행합니다."

Step "1/5 그래픽카드 확인"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Write-Host "  nvidia-smi를 찾지 못했어요. NVIDIA 드라이버를 최신으로 설치해 주세요." -ForegroundColor Yellow
}

Step "2/5 uv(파이썬 관리 도구) 준비"
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (-not (Test-Path $uv)) {
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        Check "uv 설치"
    }
}
if (-not (Test-Path $uv)) { throw "uv를 찾을 수 없어요: $uv" }
Write-Host "  $uv"

Step "3/5 Gen Studio 프로그램 설치"
& $uv venv .venv --python 3.12 --allow-existing
Check "가상환경 만들기"
& $uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Check "프로그램 설치"

Step "4/5 ComfyUI 설치 (약 2GB)"
$comfyMain = Join-Path $Root 'ComfyUI_windows_portable\ComfyUI\main.py'
if (Test-Path $comfyMain) {
    Write-Host "  이미 설치돼 있어요."
} else {
    New-Item -ItemType Directory -Force -Path tools | Out-Null
    $sevenZip = Join-Path $Root 'tools\7zr.exe'
    if (-not (Test-Path $sevenZip)) {
        & curl.exe -L --fail -o $sevenZip https://www.7-zip.org/a/7zr.exe
        Check "7-Zip 받기"
    }
    $archive = Join-Path $Root 'tools\ComfyUI_windows_portable_nvidia.7z'
    & curl.exe -L --fail -C - -o $archive https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z
    Check "ComfyUI 받기"
    Write-Host "  압축 푸는 중..."
    & $sevenZip x $archive "-o$Root" -y | Out-Null
    Check "ComfyUI 압축 풀기"
    Remove-Item $archive -Force
    if (-not (Test-Path $comfyMain)) { throw "ComfyUI 폴더를 찾을 수 없어요: $comfyMain" }
}

Step "5/5 모델 다운로드 (약 33GB)"
& .venv\Scripts\python.exe scripts\download_models.py
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "설치가 끝났어요! start.bat 을 더블클릭하면 실행됩니다." -ForegroundColor Green
} else {
    Write-Host "일부 모델을 받지 못했어요. install.bat 을 다시 실행하면 이어서 받습니다." -ForegroundColor Yellow
}

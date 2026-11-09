# Creates "Gen Studio" shortcuts on the desktop and in the Start menu (run by install.bat / update.bat / the app updater)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $Root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path $target)) { Write-Host "  .venv 가 없어서 바로가기를 만들지 않았어요 (install.bat 먼저)."; exit 0 }
$icon = Join-Path $Root 'web\icon.ico'
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
foreach ($dir in $places) {
    if (-not $dir -or -not (Test-Path $dir)) { continue }
    $lnk = $shell.CreateShortcut((Join-Path $dir 'Gen Studio.lnk'))
    $lnk.TargetPath = $target
    $lnk.Arguments = '-m genstudio.desktop'
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = 'Gen Studio - 로컬 이미지 생성/편집'
    $lnk.Save()
}
Write-Host "  바탕화면과 시작 메뉴에 'Gen Studio' 아이콘을 만들었어요."

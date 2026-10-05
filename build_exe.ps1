$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python -m PyInstaller --noconfirm --clean --windowed --onedir --name PostureGuard --add-data "pose_landmarker.task;." --add-data "posture_break_guard.config.json;." --collect-data customtkinter posture_break_guard.py

Copy-Item -LiteralPath "pose_landmarker.task" -Destination "dist\PostureGuard\pose_landmarker.task" -Force
Copy-Item -LiteralPath "posture_break_guard.config.json" -Destination "dist\PostureGuard\posture_break_guard.config.json" -Force
if (Test-Path "README_posture_break_guard.md") {
    Copy-Item -LiteralPath "README_posture_break_guard.md" -Destination "dist\PostureGuard\README_posture_break_guard.md" -Force
}

Write-Host "Build listo en dist\PostureGuard"

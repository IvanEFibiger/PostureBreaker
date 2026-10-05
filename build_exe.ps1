$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1. Modelo (externo al build, verificable por hash)
if (-not (Test-Path "pose_landmarker.task")) {
    python scripts/fetch_model.py --variant heavy
}

# 2. Tests
python -m unittest discover -s tests -t . -v

# 3. Build (fuente de verdad: PostureGuard.spec)
python -m PyInstaller --noconfirm --clean PostureGuard.spec

# 4. Modelo junto al ejecutable (para poder cambiar de variante sin recompilar)
Copy-Item -LiteralPath "pose_landmarker.task" -Destination "dist\PostureGuard\pose_landmarker.task" -Force

Write-Host "Build listo en dist\PostureGuard"

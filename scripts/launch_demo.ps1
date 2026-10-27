param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
Write-Host "SafeSight-FL local demo: http://127.0.0.1:$Port"
Write-Host 'Simulated sites; no full training, database reset or remote publication.'
python -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false

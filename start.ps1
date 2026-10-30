$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw '请先按 README 创建 .venv 并安装依赖。'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'dist\index.html'))) {
    throw '请先运行 npm install 和 npm run build。'
}
$env:AEGIS_DEVICE = 'cuda'
Write-Host 'Aegis Lab: http://127.0.0.1:8765'
Write-Host '默认使用 CUDA；关闭终端或 Ctrl+C 可停止服务。'
& $projectPython -m uvicorn backend.app:app --host 127.0.0.1 --port 8765

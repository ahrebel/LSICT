# LSICT no-trace runner for Windows (PowerShell).
#
# Downloads everything (uv, Python, all dependencies, model weights) into ONE
# temporary folder, runs the app, and deletes that folder when the app stops.
# Stop the app with Ctrl+C in the terminal so the cleanup runs; if the window
# is force-closed instead, the leftover folder sits in %TEMP% (named
# lsict-once-*) and can be deleted by hand or by Windows disk cleanup.
#
# Usage:  powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ahrebel/LSICT/main/run-once.ps1 | iex"
$ErrorActionPreference = "Stop"

$LsictSpec = "lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"

$Work = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("lsict-once-" + -join ((48..57) + (97..122) | Get-Random -Count 8 | ForEach-Object { [char]$_ }))
New-Item -ItemType Directory -Path $Work | Out-Null

# Confine EVERYTHING to the temp folder: uv itself, its caches, the Python it
# downloads, the tool environment, and the model weights fetched at runtime.
$env:UV_INSTALL_DIR = Join-Path $Work "uv"
$env:UV_CACHE_DIR = Join-Path $Work "uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Work "python"
$env:UV_TOOL_DIR = Join-Path $Work "tools"
$env:XDG_CACHE_HOME = Join-Path $Work "cache"
$env:HF_HOME = Join-Path $Work "cache\huggingface"
$env:TORCH_HOME = Join-Path $Work "cache\torch"
$env:YOLO_CONFIG_DIR = Join-Path $Work "cache\ultralytics"

try {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $UvExe = "uv"
    } else {
        Write-Host "==> Downloading uv into the temp folder..."
        $env:UV_NO_MODIFY_PATH = "1"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $UvExe = Join-Path $Work "uv\uv.exe"
    }

    Write-Host "==> Downloading LSICT and its dependencies (~2 GB) into:"
    Write-Host "        $Work"
    Write-Host "    Nothing outside this folder is touched; it is deleted on exit."
    Write-Host "    (Because nothing is kept, every launch re-downloads everything -"
    Write-Host "     use install.ps1 instead if you'll run LSICT more than once.)"
    Set-Location $Work
    & $UvExe tool run --python 3.12 --from $LsictSpec lsict gui
}
finally {
    Set-Location ([System.IO.Path]::GetTempPath())
    Write-Host "==> Cleaning up - deleting $Work ..."
    Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
    Write-Host "==> Done. Nothing was left on this machine."
}

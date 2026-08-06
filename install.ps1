# LSICT one-command installer for Windows (PowerShell).
# Usage:  powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ahrebel/LSICT/main/install.ps1 | iex"
$ErrorActionPreference = "Stop"

$LsictSpec = "lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"

Write-Host "==> LSICT installer"

# 1) Make sure uv is available (it manages Python and the install for us).
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "==> Installing uv (Python tool manager)..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    # uv installs to %USERPROFILE%\.local\bin; make it visible for this script.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 2) Install LSICT (uv downloads a suitable Python if none is present).
Write-Host "==> Installing LSICT and its dependencies (~2 GB, mostly PyTorch)."
Write-Host "    This can take several minutes on the first install..."
uv tool install --force --python 3.12 $LsictSpec

# 3) Make sure the tool directory is on PATH for future shells.
try { uv tool update-shell | Out-Null } catch { }

Write-Host ""
Write-Host "==> Done! Open a NEW terminal window and run:"
Write-Host ""
Write-Host "        lsict gui"
Write-Host ""
Write-Host "    (Model weights download automatically the first time you run a job.)"

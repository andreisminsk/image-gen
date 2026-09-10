<#
.SYNOPSIS
    Install image-gen proxy scripts to ~/.local/bin
#>

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvBin  = Join-Path $ScriptDir "venv\Scripts"
$BinDir   = Join-Path $env:USERPROFILE ".local\bin"

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

foreach ($cmd in @("image-gen", "image-gen-anim", "i2i-gen", "i2i-gen-anim", "remove-object")) {
    $src = Join-Path $VenvBin "$cmd.exe"
    if (-not (Test-Path $src)) {
        Write-Error "Not found: $src — run 'pip install -e .' first"
        exit 1
    }

    $proxy = Join-Path $BinDir "$cmd.cmd"
    # Use absolute path to venv entry point
    $target = $src -replace '\\', '\\'
    @"
@echo off
"$src" %*
"@ | Set-Content -Path $proxy -Encoding ASCII
    Write-Host "Installed $proxy -> $src"
}

# Check if ~/.local/bin is in PATH
$pathParts = $env:PATH -split ";"
if ($pathParts -notcontains $BinDir) {
    # Also check with trailing slash variations
    $normalized = $pathParts | ForEach-Object { $_.TrimEnd("\\") }
    if ($normalized -notcontains $BinDir.TrimEnd("\\")) {
        Write-Host ""
        Write-Host "Add ~/.local/bin to your PATH by running this in PowerShell:"
        Write-Host '  [Environment]::SetEnvironmentVariable("Path", [Environment]::GetEnvironmentVariable("Path","User") + ";' + $BinDir + '", "User")'
        Write-Host "Then restart your shell."
    } else {
        Write-Host ""
        Write-Host "Done! ~/.local/bin is already in your PATH."
    }
} else {
    Write-Host ""
    Write-Host "Done! ~/.local/bin is already in your PATH."
}

<#
.SYNOPSIS
    Remove image-gen proxy scripts from ~/.local/bin
#>

$ErrorActionPreference = "Stop"

$BinDir = Join-Path $env:USERPROFILE ".local\bin"

foreach ($cmd in @("image-gen", "image-gen-anim", "i2i-gen", "i2i-gen-anim", "remove-object")) {
    $proxy = Join-Path $BinDir "$cmd.cmd"
    if (Test-Path $proxy) {
        Remove-Item $proxy
        Write-Host "Removed $proxy"
    } else {
        Write-Host "Not found: $proxy"
    }
}

Write-Host "Done."

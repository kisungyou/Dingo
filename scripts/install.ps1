param(
    [string]$Version = "latest",
    [string]$Repo = "kisungyou/Dingo"
)

$ErrorActionPreference = "Stop"

$archMap = @{
    "AMD64" = "x86_64"
    "ARM64" = "arm64"
}
$arch = $archMap[$env:PROCESSOR_ARCHITECTURE]
if (-not $arch) {
    $arch = $env:PROCESSOR_ARCHITECTURE.ToLower()
}

if ($Version -eq "latest") {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
    $Version = $release.tag_name.TrimStart("v")
}

$msi = "dingo-$Version-windows-$arch.msi"
$url = "https://github.com/$Repo/releases/download/v$Version/$msi"
$out = Join-Path $env:TEMP $msi
Invoke-WebRequest -Uri $url -OutFile $out
Start-Process msiexec.exe -ArgumentList "/i `"$out`" /qn /norestart" -Wait -NoNewWindow
Write-Host "Installed Dingo $Version ($arch)"

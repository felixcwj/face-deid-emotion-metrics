$scriptPath = $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptPath)
$deidScript = Join-Path $repoRoot "scripts\deid.ps1"

if (-not (Test-Path -LiteralPath $deidScript)) {
    throw "Unable to locate deid.ps1 at $deidScript"
}

$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path -LiteralPath $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force | Out-Null
}

$markerStart = "# face-deid-emotion-metrics deid alias start"
$markerEnd = "# face-deid-emotion-metrics deid alias end"
$aliasBlock = @"
$markerStart
function deid {
    & "$deidScript" @args
}
$markerEnd
"@.Trim()

$profileContent = Get-Content -LiteralPath $profilePath -Raw
$escapedStart = [regex]::Escape($markerStart)
$escapedEnd = [regex]::Escape($markerEnd)
$pattern = "(?s)$escapedStart.*?$escapedEnd`r?`n?"

if ($profileContent -match $pattern) {
    $profileContent = [regex]::Replace($profileContent, $pattern, $aliasBlock + [Environment]::NewLine)
} else {
    if ($profileContent.Length -gt 0 -and -not $profileContent.EndsWith([Environment]::NewLine)) {
        $profileContent += [Environment]::NewLine
    }
    $profileContent += $aliasBlock + [Environment]::NewLine
}

Set-Content -LiteralPath $profilePath -Value $profileContent -Encoding UTF8
Write-Host "Registered deid() in $profilePath. Open a new PowerShell window (or run `. $profilePath`) to use it."

param(
    [string]$BaseDir = 'D:\RAPA',
    [string]$Output = 'D:\RAPA\rapa_report_full_wsl.xlsx',
    [int]$MaxFiles,
    [string]$VideoBackend,
    [string]$LogLevel,
    [string]$WslDistro,
    [string[]]$AdditionalArgs = @()
)

$RepoWindowsPath = 'C:\projects\face-deid-emotion-metrics'

function Test-WslAvailable {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw "wsl.exe not found. Enable WSL and install a CUDA-enabled distro before running this script."
    }
}

function Invoke-WslCommand {
    param(
        [Parameter(Mandatory)]
        [string[]]$CommandArgs
    )
    $wslArgs = @()
    if ($WslDistro) {
        $wslArgs += @('-d', $WslDistro)
    }
    $wslArgs += '--'
    $wslArgs += $CommandArgs
    $result = & wsl.exe @wslArgs
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code $LASTEXITCODE."
    }
    return $result
}

function Convert-ToWslPath {
    param([string]$Path)
    if (-not $Path) { return '' }
    $expanded = [System.IO.Path]::GetFullPath($Path)
    if ($expanded -match '^[A-Za-z]:(.*)$') {
        $drive = $expanded.Substring(0,1).ToLowerInvariant()
        $rest = $expanded.Substring(2).Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "Unsupported path format: $Path"
}

function Escape-WslSingleQuotes {
    param([string]$Value)
    if ($Value -eq $null) { return '' }
    $replacement = "'" + '"' + "'" + '"' + "'"
    return $Value -replace "'", $replacement
}

function Format-WslArg {
    param([string]$Value)
    $escaped = Escape-WslSingleQuotes -Value $Value
    return "'$escaped'"
}

function Invoke-WslBash {
    param([string]$Command)
    $bashArgs = @('bash', '-lc', $Command)
    Invoke-WslCommand -CommandArgs $bashArgs | ForEach-Object { $_ }
}

Test-WslAvailable
if (-not (Test-Path -LiteralPath $BaseDir)) {
    throw "Base directory not found: $BaseDir"
}

$repoWsl = Convert-ToWslPath -Path $RepoWindowsPath
$baseWsl = Convert-ToWslPath -Path $BaseDir
$outputWsl = Convert-ToWslPath -Path $Output

if (-not $repoWsl) {
    throw "Unable to translate $RepoWindowsPath into a WSL path."
}
if (-not $baseWsl) {
    throw "Unable to translate $BaseDir into a WSL path."
}
if (-not $outputWsl) {
    throw "Unable to translate $Output into a WSL path."
}

$cmd = "cd $(Format-WslArg $repoWsl) && ./scripts/wsl/run_pipeline.sh --base-dir $(Format-WslArg $baseWsl) --output $(Format-WslArg $outputWsl)"
if ($PSBoundParameters.ContainsKey('MaxFiles')) {
    $cmd += " --max-files $MaxFiles"
}
if ($PSBoundParameters.ContainsKey('VideoBackend') -and $VideoBackend) {
    $cmd += " --video-backend $(Format-WslArg $VideoBackend)"
}
if ($PSBoundParameters.ContainsKey('LogLevel') -and $LogLevel) {
    $cmd += " --log-level $(Format-WslArg $LogLevel)"
}
if ($AdditionalArgs.Count -gt 0) {
    foreach ($arg in $AdditionalArgs) {
        $cmd += " " + (Format-WslArg $arg)
    }
}

Write-Host "Invoking WSL pipeline..."
Invoke-WslBash -Command $cmd

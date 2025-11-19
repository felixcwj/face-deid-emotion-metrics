param(
    [string]$BaseDir = 'D:\RAPA',
    [string]$Output = '',
    [string[]]$Args
)

$defaultBaseDir = 'D:\RAPA'
$topBottomMode = $false
if ($BaseDir -eq '40' -and -not (Test-Path -LiteralPath $BaseDir)) {
    $topBottomMode = $true
    $BaseDir = $defaultBaseDir
}

if (-not $Output) {
    if ($topBottomMode) {
        $Output = Join-Path $BaseDir 'rapa_report_top_bottom_40.xlsx'
    } else {
        $Output = Join-Path $BaseDir 'rapa_report_samples.xlsx'
    }
}

$projectRoot = 'C:\projects\face-deid-emotion-metrics'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = 'python'
}

Set-Location $projectRoot
$cliArgs = @(
    '-m', 'face_deid_emotion_metrics.cli',
    '--base-dir', $BaseDir,
    '--output', $Output,
    '--video-backend', 'ffmpeg'
)

if ($topBottomMode) {
    $cliArgs += '--top-bottom-40-only'
}

if ($Args) {
    $cliArgs += $Args
}

& $pythonExe @cliArgs

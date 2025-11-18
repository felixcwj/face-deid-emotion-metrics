param(
    [string]$BaseDir = 'D:\RAPA',
    [string]$Output = '',
    [string[]]$Args
)
if (-not $Output) {
    $Output = Join-Path $BaseDir 'rapa_report_samples.xlsx'
}
$projectRoot = 'C:\projects\face-deid-emotion-metrics'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
Set-Location $projectRoot
& $pythonExe -m face_deid_emotion_metrics.cli --base-dir $BaseDir --output $Output @Args

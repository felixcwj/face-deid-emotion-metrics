# WSL2 Ubuntu GPU workflow is recommended for full runs; this Windows script is best for quick samples or debugging.

$baseDir = 'D:\RAPA'
$outputPath = 'D:\RAPA\rapa_report.xlsx'
$pythonExe = 'python'
if (-not (Test-Path $baseDir)) {
    Write-Error "Base directory not found: $baseDir"
    exit 1
}
& $pythonExe -m face_deid_emotion_metrics.cli --base-dir $baseDir --output $outputPath

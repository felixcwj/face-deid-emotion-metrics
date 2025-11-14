$scriptPath = $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptPath)
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Write-Host "Where is the dataset folder (must contain input/ and output/)?"
$baseDir = Read-Host "Base directory path"
if ([string]::IsNullOrWhiteSpace($baseDir) -or -not (Test-Path $baseDir)) {
    Write-Error "Base directory not found."
    exit 1
}

$defaultOutput = Join-Path $baseDir "rapa_report_interactive.xlsx"
$outputPrompt = "Output Excel path (Enter for $defaultOutput)"
$outputPath = Read-Host $outputPrompt
if ([string]::IsNullOrWhiteSpace($outputPath)) {
    $outputPath = $defaultOutput
}

$confirmation = Read-Host "Ready to start? Press Y to proceed"
if ($confirmation -notin @("Y", "y")) {
    Write-Host "Cancelled."
    exit 0
}

& $pythonExe -m face_deid_emotion_metrics.cli --base-dir $baseDir --output $outputPath

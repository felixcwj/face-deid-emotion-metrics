[CmdletBinding()]
param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\face_deid_decord",
    [string]$CudaToolkit = $env:CUDA_PATH,
    [string]$PythonExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message"
}

function Ensure-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "필수 명령 '$Name' 을(를) 찾을 수 없습니다. PATH 설정 또는 설치 여부를 확인하세요."
    }
}

function Resolve-PythonExe {
    param([string]$Requested)
    if ($Requested) {
        return $Requested
    }
    $repoPython = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"
    if (Test-Path $repoPython) {
        return $repoPython
    }
    return "python"
}

function Ensure-VisualStudio {
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe 를 찾을 수 없습니다. Visual Studio Installer가 필요한데, Visual Studio Build Tools를 먼저 설치하세요."
    }
    $args = @("-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild", "-property", "installationPath")
    $path = & $vswhere @args
    if ($path) {
        return $path
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Visual Studio Build Tools가 없어 winget으로 설치를 시도합니다."
        $wingetArgs = @(
            "install",
            "--id", "Microsoft.VisualStudio.2022.BuildTools",
            "-e",
            "--source", "winget",
            "--accept-package-agreements",
            "--accept-source-agreements"
        )
        & winget @wingetArgs
        $path = & $vswhere @args
        if ($path) {
            return $path
        }
    }
    throw "Visual Studio Build Tools를 찾거나 설치하지 못했습니다. https://visualstudio.microsoft.com/ko/downloads/ 에서 설치 후 다시 시도하세요."
}

function Download-File {
    param(
        [string]$Url,
        [string]$Destination
    )
    Write-Info "다운로드: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
}

function Ensure-FFmpeg {
    param([string]$Root)
    $ffmpegCache = Join-Path $Root "ffmpeg"
    $ffmpegZip = Join-Path $Root "ffmpeg.zip"
    $ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1-essentials_build.zip"
    if (-not (Test-Path $ffmpegCache)) {
        New-Item -ItemType Directory -Path $ffmpegCache | Out-Null
    }
    $existing = Get-ChildItem -Path $ffmpegCache -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $existing) {
        Download-File -Url $ffmpegUrl -Destination $ffmpegZip
        Expand-Archive -Path $ffmpegZip -DestinationPath $ffmpegCache -Force
        Remove-Item $ffmpegZip -Force
        $existing = Get-ChildItem -Path $ffmpegCache -Directory | Select-Object -First 1
    }
    if (-not $existing) {
        throw "FFmpeg 디렉터리를 찾을 수 없습니다."
    }
    $binPath = Join-Path $existing.FullName "bin"
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($userPath -notlike "*$binPath*") {
        [Environment]::SetEnvironmentVariable("PATH", "$binPath;$userPath", "User")
        Write-Info "FFmpeg bin 경로를 사용자 PATH에 추가했습니다: $binPath"
    }
    $env:PATH = "$binPath;$env:PATH"
    return $existing.FullName
}

function Ensure-DecordSource {
    param(
        [string]$Root,
        [string]$RepoUrl
    )
    $sourceDir = Join-Path $Root "decord"
    if (-not (Test-Path $sourceDir)) {
        Write-Info "decord 저장소를 복제합니다."
        git clone --recursive $RepoUrl $sourceDir | Out-Null
    } else {
        Write-Info "decord 저장소를 최신 상태로 가져옵니다."
        git -C $sourceDir pull --recurse-submodules | Out-Null
    }
    return $sourceDir
}

function Install-DecordGpu {
    param(
        [string]$SourceDir,
        [string]$PythonExePath,
        [string]$CudaRoot,
        [string]$FfmpegRoot
    )
    $env:FFMPEG_DIR = $FfmpegRoot
    $cmakeOptions = "-DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=`"$CudaRoot`" -DFFMPEG_DIR=`"$FfmpegRoot`""
    $env:DECORD_CMAKE_OPTS = $cmakeOptions
    $pythonDir = Join-Path $SourceDir "python"
    Push-Location $pythonDir
    try {
        Write-Info "pip 를 통해 GPU 지원 decord를 빌드/설치합니다."
        & $PythonExePath -m pip install --upgrade pip
        & $PythonExePath -m pip install --force-reinstall --no-binary decord --no-build-isolation .
    } finally {
        Pop-Location
        Remove-Item Env:DECORD_CMAKE_OPTS -ErrorAction SilentlyContinue
        Remove-Item Env:FFMPEG_DIR -ErrorAction SilentlyContinue
    }
}

function Verify-Decord {
    param(
        [string]$PythonExePath,
        [string]$Workspace
    )
    $sampleUrl = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
    $samplePath = Join-Path $Workspace "decord_sample.mp4"
    $verifyScript = Join-Path $Workspace "verify_decord.py"
    Download-File -Url $sampleUrl -Destination $samplePath
    $scriptContent = @"
from decord import VideoReader, gpu
vr = VideoReader(r"$samplePath", ctx=gpu(0))
print("Decord GPU verification OK - frames:", len(vr))
"@
    Set-Content -Path $verifyScript -Value $scriptContent -Encoding UTF8
    try {
        & $PythonExePath $verifyScript
    } finally {
        Remove-Item $samplePath, $verifyScript -ErrorAction SilentlyContinue
    }
}

if (-not $CudaToolkit -or -not (Test-Path $CudaToolkit)) {
    throw "CUDA Toolkit 경로를 찾을 수 없습니다. `-CudaToolkit` 인수를 직접 지정하거나 CUDA가 설치돼 있는지 확인하세요."
}

Ensure-Command -Name git
Ensure-Command -Name cmake
Ensure-Command -Name "powershell"
$pythonPath = Resolve-PythonExe -Requested $PythonExe
Write-Info "Python 실행 파일: $pythonPath"

$vsPath = Ensure-VisualStudio
Write-Info "Visual Studio 경로: $vsPath"

if (-not (Test-Path $InstallRoot)) {
    New-Item -ItemType Directory -Path $InstallRoot | Out-Null
}

$ffmpegRoot = Ensure-FFmpeg -Root $InstallRoot
Write-Info "FFmpeg 루트: $ffmpegRoot"

$sourceDir = Ensure-DecordSource -Root $InstallRoot -RepoUrl "https://github.com/dmlc/decord.git"

Install-DecordGpu -SourceDir $sourceDir -PythonExePath $pythonPath -CudaRoot $CudaToolkit -FfmpegRoot $ffmpegRoot

Verify-Decord -PythonExePath $pythonPath -Workspace $InstallRoot

Write-Info "GPU 지원 decord 설치를 완료했습니다. PowerShell을 새로 열고 파이프라인을 다시 실행하세요."

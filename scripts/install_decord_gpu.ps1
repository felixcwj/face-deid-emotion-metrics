[CmdletBinding()]
param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\face_deid_decord",
    [string]$CudaToolkit = $env:CUDA_PATH,
    [string]$PythonExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Administrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "관리자 권한 PowerShell에서 실행해야 합니다. PowerShell을 관리자 모드로 다시 열어 주세요."
    }
}

Assert-Administrator

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
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing -ErrorAction Stop
    } catch {
        throw "다운로드 실패: $Url (`$($_.Exception.Message)`)"
    }
}

function Ensure-FFmpeg {
    param([string]$Root)
    $ffmpegCache = Join-Path $Root "ffmpeg"
    $ffmpegZip = Join-Path $Root "ffmpeg.zip"
    $ffmpegUrls = @(
        "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1-essentials_build.zip",
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip"
    )
    if (-not (Test-Path $ffmpegCache)) {
        New-Item -ItemType Directory -Path $ffmpegCache | Out-Null
    }
    $existing = Get-ChildItem -Path $ffmpegCache -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $existing) {
        $downloaded = $false
        foreach ($url in $ffmpegUrls) {
            try {
                Download-File -Url $url -Destination $ffmpegZip
                $downloaded = $true
                break
            } catch {
                Write-Info "FFmpeg 다운로드 실패: $_"
            }
        }
        if (-not $downloaded) {
            throw "FFmpeg 패키지를 다운로드할 수 없습니다. 스크립트 상단의 URL 목록이 모두 실패했습니다."
        }
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

function Ensure-CudaToolkit {
    param(
        [string]$Requested,
        [string]$WorkRoot
    )
    function Test-CudaToolkitPath {
        param([string]$Path)
        if (-not (Test-Path $Path)) {
            return $false
        }
        $nvcc = Join-Path $Path "bin\nvcc.exe"
        if (-not (Test-Path $nvcc)) {
            return $false
        }
        try {
            $output = & $nvcc --version 2>&1
            if ($LASTEXITCODE -ne 0) {
                return $false
            }
            if ($output -match "release\s+(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                if ($major -lt 12) {
                    return $false
                }
            }
            return $true
        } catch {
            return $false
        }
    }
    $candidates = @()
    if ($Requested) {
        $candidates += $Requested
    }
    if ($env:CUDA_PATH) {
        $candidates += $env:CUDA_PATH
    }
    $defaultRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $defaultRoot) {
        $versions = Get-ChildItem -Path $defaultRoot -Directory -ErrorAction SilentlyContinue | Sort-Object -Property Name -Descending
        foreach ($versionDir in $versions) {
            $candidates += $versionDir.FullName
        }
    }
    foreach ($candidate in $candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-CudaToolkitPath -Path $candidate) {
            return $candidate
        }
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "CUDA Toolkit을 찾을 수 없습니다. winget으로 설치를 시도합니다 (NVIDIA.CUDA)."
        & winget install --id NVIDIA.CUDA -e `
            --source winget `
            --accept-package-agreements `
            --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "winget으로 CUDA Toolkit 설치에 실패했습니다. 수동으로 설치 후 다시 시도하세요."
        }
        if (Test-Path $defaultRoot) {
            $versions = Get-ChildItem -Path $defaultRoot -Directory | Sort-Object -Property Name -Descending
            foreach ($versionDir in $versions) {
                if (Test-CudaToolkitPath -Path $versionDir.FullName) {
                    return $versionDir.FullName
                }
            }
        }
    } elseif ($WorkRoot) {
        $cudaInstallerUrl = "https://developer.download.nvidia.com/compute/cuda/12.6.0/network_installers/cuda_12.6.0_windows_network.exe"
        $installerPath = Join-Path $WorkRoot "cuda_windows_network.exe"
        Write-Info "winget 사용 불가. NVIDIA CUDA 네트워크 인스톨러를 다운로드합니다."
        Download-File -Url $cudaInstallerUrl -Destination $installerPath
        Write-Info "CUDA Toolkit 설치 프로그램을 실행합니다 (시간이 걸릴 수 있음)."
        Start-Process -FilePath $installerPath -ArgumentList "-s", "-loglevel:6" -Wait -NoNewWindow
        Remove-Item $installerPath -ErrorAction SilentlyContinue
        if (Test-Path $defaultRoot) {
            $versions = Get-ChildItem -Path $defaultRoot -Directory | Sort-Object -Property Name -Descending
            foreach ($versionDir in $versions) {
                if (Test-CudaToolkitPath -Path $versionDir.FullName) {
                    return $versionDir.FullName
                }
            }
        }
    }
    throw "CUDA Toolkit 경로를 찾을 수 없습니다. `-CudaToolkit` 인수로 올바른 경로를 지정하거나 CUDA Toolkit을 설치하세요."
}

function Invoke-VsCommand {
    param(
        [string]$VsDevCmd,
        [string]$Command
    )
    $wrapped = "`"$VsDevCmd`" -arch=x64 && $Command"
    Write-Info "VS 환경 명령 실행: $Command"
    cmd.exe /c $wrapped
    if ($LASTEXITCODE -ne 0) {
        throw "Visual Studio 환경 명령 실패: $Command"
    }
}

function Build-Decord {
    param(
        [string]$SourceDir,
        [string]$CudaRoot,
        [string]$FfmpegRoot,
        [string]$VsDevCmd
    )
    $buildDir = Join-Path $SourceDir "build"
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    $cudaCompiler = Join-Path $CudaRoot "bin\nvcc.exe"
    $cmakeArgs = @(
        "-S", $SourceDir,
        "-B", $buildDir,
        "-DUSE_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCUDA_TOOLKIT_ROOT_DIR=$CudaRoot",
        "-DFFMPEG_DIR=$FfmpegRoot",
        "-DCMAKE_CUDA_COMPILER=$cudaCompiler"
    )
    $escaped = $cmakeArgs | ForEach-Object {
        if ($_ -match "\s") { "`"$_`"" } else { $_ }
    }
    $configureCommand = "cmake " + ($escaped -join " ")
    Invoke-VsCommand -VsDevCmd $VsDevCmd -Command $configureCommand
    $buildCommand = "cmake --build `"$buildDir`" --config Release"
    Invoke-VsCommand -VsDevCmd $VsDevCmd -Command $buildCommand
}

function Install-DecordGpu {
    param(
        [string]$SourceDir,
        [string]$PythonExePath,
        [string]$CudaRoot,
        [string]$FfmpegRoot,
        [string]$VsDevCmd
    )
    Build-Decord -SourceDir $SourceDir -CudaRoot $CudaRoot -FfmpegRoot $FfmpegRoot -VsDevCmd $VsDevCmd
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
        if ($LASTEXITCODE -ne 0) {
            throw "decord GPU 검증에 실패했습니다. 설치 로그를 확인하세요."
        }
    } finally {
        Remove-Item $samplePath, $verifyScript -ErrorAction SilentlyContinue
    }
}

# ensure installation workspace exists early (needed for CUDA installer downloads)
if (-not (Test-Path $InstallRoot)) {
    New-Item -ItemType Directory -Path $InstallRoot | Out-Null
}

$CudaToolkitPath = Ensure-CudaToolkit -Requested $CudaToolkit -WorkRoot $InstallRoot
# refresh env in case winget just installed
$env:CUDA_PATH = $CudaToolkitPath
$env:CUDA_PATH_V12_0 = $CudaToolkitPath

Ensure-Command -Name git
Ensure-Command -Name cmake
Ensure-Command -Name "powershell"
$pythonPath = Resolve-PythonExe -Requested $PythonExe
Write-Info "Python 실행 파일: $pythonPath"

$vsPath = Ensure-VisualStudio
Write-Info "Visual Studio 경로: $vsPath"
$vsDevCmd = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "VsDevCmd.bat을 찾을 수 없습니다: $vsDevCmd"
}

$ffmpegRoot = Ensure-FFmpeg -Root $InstallRoot
Write-Info "FFmpeg 루트: $ffmpegRoot"

$sourceDir = Ensure-DecordSource -Root $InstallRoot -RepoUrl "https://github.com/dmlc/decord.git"

Install-DecordGpu -SourceDir $sourceDir -PythonExePath $pythonPath -CudaRoot $CudaToolkitPath -FfmpegRoot $ffmpegRoot -VsDevCmd $vsDevCmd

Verify-Decord -PythonExePath $pythonPath -Workspace $InstallRoot

Write-Info "GPU 지원 decord 설치를 완료했습니다. PowerShell을 새로 열고 파이프라인을 다시 실행하세요."

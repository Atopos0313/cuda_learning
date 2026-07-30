param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$visualStudioRoot = "D:\software\VS2022"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

$developerEnvironment = Join-Path $visualStudioRoot `
    "VC\Auxiliary\Build\vcvars64.bat"

$cmake = Join-Path $visualStudioRoot `
    "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"

foreach ($requiredTool in @(
    $developerEnvironment,
    $cmake,
    $nvcc
)) {
    if (-not (Test-Path -LiteralPath $requiredTool)) {
        throw "Required build tool was not found: $requiredTool"
    }
}

$buildPreset = switch ($Configuration) {
    "Debug" { "debug" }
    "Release" { "release" }
    "RelWithDebInfo" { "profile" }
}

$command = (
    "call `"$developerEnvironment`" >nul " +
    "&& cd /d `"$projectRoot`" " +
    "&& set `"CUDA_PATH=$cudaRoot`" " +
    "&& set `"CUDACXX=$nvcc`" " +
    "&& `"$cmake`" --preset windows-msvc " +
    "&& `"$cmake`" --build --preset $buildPreset"
)

& cmd.exe /d /s /c $command
exit $LASTEXITCODE

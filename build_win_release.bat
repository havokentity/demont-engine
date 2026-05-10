@echo off
REM Incremental build of win-clang-release without re-running cmake configure.
REM Use this when sources change but the cache is fine; bootstrap_win_release
REM is for fresh configure.

setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do (
    set "VSDEV_DIR=%%i\VC\Auxiliary\Build"
)
call "%VSDEV_DIR%\vcvars64.bat" >nul

cmake --build build\win-clang-release --parallel

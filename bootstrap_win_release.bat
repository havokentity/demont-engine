@echo off
REM Configure + build win-clang-release from scratch.
REM Variant of bootstrap_win.bat for the clang-cl release preset.

setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
)
if not exist "%VSWHERE%" (
    echo [bootstrap_win_release] vswhere.exe not found.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do (
    set "VSDEV_DIR=%%i\VC\Auxiliary\Build"
)

if not exist "%VSDEV_DIR%\vcvars64.bat" (
    echo [bootstrap_win_release] vcvars64.bat missing.
    exit /b 1
)

call "%VSDEV_DIR%\vcvars64.bat" >nul

cmake --preset win-clang-release
if errorlevel 1 exit /b %errorlevel%
cmake --build build\win-clang-release --parallel

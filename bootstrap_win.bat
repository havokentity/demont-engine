@echo off
REM Configure + build win-debug from scratch.
REM Locates the active Visual Studio install via vswhere (ships with the
REM VS Installer at the path below), so this works on Community / Pro /
REM Enterprise / Build Tools across VS 2019..2026 without hardcoding a
REM specific year/edition path.
REM
REM Useful after wiping build/ -- build_win.bat alone only does
REM `cmake --build` and assumes the tree is already configured.

setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
)
if not exist "%VSWHERE%" (
    echo [bootstrap_win] vswhere.exe not found at expected paths.
    echo                 Install Visual Studio + try again, or set VSDEV_DIR
    echo                 to the directory containing vcvars64.bat.
    exit /b 1
)

REM Find the latest VS install with the C++ workload installed.
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * ^
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    -property installationPath`) do (
    set "VSDEV_DIR=%%i\VC\Auxiliary\Build"
)

if not exist "%VSDEV_DIR%\vcvars64.bat" (
    echo [bootstrap_win] Found VS install but vcvars64.bat is missing at:
    echo                 %VSDEV_DIR%
    exit /b 1
)

call "%VSDEV_DIR%\vcvars64.bat" >nul

REM cmake comes with VS so it's on PATH after vcvars; if not, fall back to
REM the cmake.exe shipped with the active install.
where cmake >nul 2>&1
if errorlevel 1 (
    echo [bootstrap_win] cmake.exe not on PATH after vcvars.
    exit /b 1
)

cmake --preset win-debug
if errorlevel 1 exit /b %errorlevel%
cmake --build build\win-debug --parallel

@echo off
REM Run the win-clang-release demont.exe build with the worktree root as cwd.
REM Anchoring cwd to the worktree root means demont.cfg + assets/ resolve
REM correctly regardless of where the script is invoked from. Forwards any
REM extra args after the script name to demont.exe.
REM
REM Multi-instance ports: pass --net-port=N and --net-line-port=M to
REM run alongside another demont.exe without TCP/HTTP collisions, e.g.
REM   run_win_release.bat --net-port=27962 --net-line-port=27963
REM Defaults are 27960 (HTTP/WebSocket) and 27961 (TCP console). Args
REM beat archived cvars from demont.cfg.

setlocal

set "EXE=%~dp0build\win-clang-release\src\app\demont.exe"
if not exist "%EXE%" (
    echo [run_win_release] %EXE% not found.
    echo                  Run bootstrap_win_release.bat first.
    exit /b 1
)

pushd "%~dp0"
"%EXE%" %*
set "RC=%ERRORLEVEL%"
popd

exit /b %RC%

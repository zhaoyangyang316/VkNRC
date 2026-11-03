@echo off
setlocal
cd /d "%~dp0"
if not exist "build\Release\VkNRC.exe" (
    echo Build the Release executable first. See README.md for the CMake commands.
    pause
    exit /b 1
)
set "SCENE=%~1"
if not defined SCENE (
    "build\Release\VkNRC.exe" "assets\cornell-box\cornell-box.obj" --camera 0 -0.005007153 3.86123033 --fov 39.30765
) else (
    "build\Release\VkNRC.exe" "%SCENE%"
)
if errorlevel 1 (
    echo VkNRC failed. Review the error above.
    pause
    exit /b 1
)

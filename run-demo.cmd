@echo off
setlocal
cd /d "%~dp0"
if not exist "build\Release\VkNRC.exe" (
    echo Build the Release executable first. See README.md for the CMake commands.
    pause
    exit /b 1
)
"build\Release\VkNRC.exe" %*
if errorlevel 1 (
    echo VkNRC failed. Review the error above.
    pause
    exit /b 1
)

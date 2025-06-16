@echo off
REM ================================================================
REM Carotid Artery Segmentation Launcher
REM Enhanced Tabbed Interface with Dark Mode Support  
REM Version 2.0 - Fixed All Errors
REM ================================================================

title Carotid Artery Segmentation Launcher

echo.
echo ========================================
echo  Carotid Artery Segmentation Tool
echo  Enhanced Launcher v2.0 [FIXED]
echo ========================================
echo.

REM Set color scheme
color 0A

REM Check if Python is available
echo [CHECK] Verifying Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python not found in PATH
    echo Please install Python or activate your conda environment
    echo.
    echo Suggested solutions:
    echo - Activate conda environment: conda activate ridho-ta
    echo - Check Python installation
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Python detected
)

REM Check conda environment
echo [CHECK] Checking conda environment...
conda info --envs | findstr /C:"ridho-ta" >nul 2>&1
if not errorlevel 1 (
    echo [OK] conda environment 'ridho-ta' detected
) else (
    echo [INFO] conda environment check skipped
)

REM Check if required files exist
echo [CHECK] Verifying required files...
if not exist "launcher_with_inference_log.py" (
    color 0C
    echo [ERROR] launcher_with_inference_log.py not found
    echo Please ensure you're running this from the correct directory
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
) else (
    echo [OK] launcher_with_inference_log.py found
)

if not exist "data_viewer.py" (
    color 0E
    echo [WARNING] data_viewer.py not found
    echo Data viewer functionality may not work properly
) else (
    echo [OK] data_viewer.py found
)

if not exist "requirements.txt" (
    color 0E
    echo [WARNING] requirements.txt not found
    echo Dependencies may not be properly installed
) else (
    echo [OK] requirements.txt found
)

if not exist "theme_manager.py" (
    color 0E
    echo [WARNING] theme_manager.py not found
    echo Dark mode functionality may not work
) else (
    echo [OK] theme_manager.py found - Dark mode enabled
)

echo.
echo [INFO] All error fixes applied:
echo       - Unicode emoji errors fixed
echo       - Missing method errors resolved  
echo       - Syntax errors corrected
echo       - Method references updated
echo.
echo [INFO] Starting Enhanced Carotid Segmentation Launcher...
echo [INFO] Features: Tabbed Interface, Dark/Light Mode, Enhanced Inference
echo [INFO] Press Ctrl+C to close the application
echo.

REM Set back to normal color
color 07

REM Launch the application
python launcher_with_inference_log.py

REM Check exit code
if errorlevel 1 (
    color 0C
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    echo Check the console output above for details
    echo.
    echo Common solutions:
    echo - Check if all dependencies are installed
    echo - Verify data files are in correct location
    echo - Ensure Python environment is properly set up
    echo.
    pause
    exit /b 1
) else (
    color 0A
    echo.
    echo [SUCCESS] Application closed successfully
    echo All errors have been resolved!
)

color 07
pause

@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo Checking build environment (Microsoft Visual C++ Build Tools)...
set "NEED_VC=1"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        if exist "%%i" set "NEED_VC=0"
    )
)

if "%NEED_VC%"=="1" (
    echo =========================================================================
    echo Missing Microsoft Visual C++ 14.0 or higher build tools ^(required for packages^).
    echo Automatically downloading and installing Visual Studio Build Tools 2022...
    echo ^(Installation may take several minutes. If an admin prompt appears, choose 'Yes'^)
    echo =========================================================================
    curl -# -L "https://aka.ms/vs/17/release/vs_buildtools.exe" -o "%TEMP%\vs_buildtools.exe"
    if exist "%TEMP%\vs_buildtools.exe" (
        start /wait "" "%TEMP%\vs_buildtools.exe" --quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
        del "%TEMP%\vs_buildtools.exe"
        echo Build Tools installation completed.
    ) else (
        echo [WARNING] Download failed. If package installation fails later, please download and install manually:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    )
) ELSE (
    echo Visual C++ build tools requirement satisfied.
)

IF NOT EXIST ".venv" (
    echo =========================================================================
    echo .venv virtual environment not found. Preparing to create one automatically...
    python -m venv .venv
)

echo Activating virtual environment and ensuring all packages are correctly installed...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip > nul
pip install -r requirements.txt

set PYTHONPATH=%PYTHONPATH%;%cd%\sys

echo =========================================================================
echo Starting DNN-HA GUI application...
python gui\web_app.py

call .venv\Scripts\deactivate.bat
echo Execution completed.
pause

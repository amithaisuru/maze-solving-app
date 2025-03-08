@echo off
setlocal EnableDelayedExpansion

:: Script to install dependencies for the Python maze project on Windows and run maze_app.py

echo Checking for Python 3...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python 3 is not installed!
    echo Please download and install Python 3 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo Found Python !PYTHON_VERSION!
)

echo Checking for pip...
python -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo pip is not installed! Installing now...
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    del get-pip.py
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install pip!
        pause
        exit /b 1
    )
    echo pip installed successfully!
) else (
    for /f "tokens=2" %%i in ('python -m pip --version') do set PIP_VERSION=%%i
    echo Found pip !PIP_VERSION!
)

echo Checking for requirements.txt...
if not exist "requirements.txt" (
    echo requirements.txt not found in the current directory!
    pause
    exit /b 1
)

echo Installing project dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies!
    pause
    exit /b 1
)

echo All dependencies installed successfully!
echo Starting the maze application...
python maze_app.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to run maze_app.py! Please check the file exists and there are no errors.
    pause
    exit /b 1
)

echo Application closed.
pause
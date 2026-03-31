@echo off
REM ============================================
REM Iris 3.0 - Automatic Startup Script (Windows)
REM ============================================

chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Iris 3.0 - Иридологичен Анализатор
echo ============================================
echo.

REM Get the directory where the script is located
cd /d "%~dp0"

REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Грешка: Python не е инсталиран!
    echo     Моля инсталирайте Python 3.8+ от https://www.python.org/downloads/
    echo     Уверете се, че сте избрали "Add Python to PATH" при инсталация
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Използва се: %PYTHON_VERSION%

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo.
    echo [*] Създаване на виртуална среда...
    python -m venv venv
)

REM Activate virtual environment
echo [*] Активиране на виртуална среда...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [*] Обновяване на pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo [*] Инсталиране на зависимости...
pip install -r requirements.txt --quiet

echo.
echo [OK] Всички зависимости са инсталирани!
echo.

REM Check if IRIS_WORKER_URL is set
if defined IRIS_WORKER_URL (
    echo [AI] AI анализ: ВКЛЮЧЕН
    echo      Worker URL: %IRIS_WORKER_URL%
) else (
    echo [i] AI анализ: ИЗКЛЮЧЕН (само обработка на изображения)
    echo     За да включите AI анализ, стартирайте с:
    echo     set IRIS_WORKER_URL=https://your-worker.workers.dev ^&^& start.bat
)

echo.
echo [*] Стартиране на сървъра...
echo     Отворете браузъра на: http://localhost:5000
echo.
echo     Натиснете Ctrl+C за спиране на сървъра
echo ============================================
echo.

REM Start the Flask app
python app.py

pause

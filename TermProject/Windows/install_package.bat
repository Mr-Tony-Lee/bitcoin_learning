@echo off
TITLE Bitcoin Project Package Installer

REM 設定虛擬環境的資料夾名稱
SET VENV_NAME=venv_bitcoin

echo.
echo --- Step 1: Checking for Python ---
REM 檢查 Python 是否已安裝並在系統路徑中
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not found. Attempting to install using Winget...
    
    REM 檢查 Winget 是否存在
    winget --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo Winget is not found. Please install Python manually from python.org and add it to your PATH.
        pause
        exit /b
    )
    
    REM 使用 Winget 安裝最新版的 Python 3
    echo Found Winget. Installing Python...
    winget install Python.Python.3 --source winget
    
    REM 再次檢查 Python 是否安裝成功
    python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo Python installation failed or was cancelled. Please try installing manually.
        pause
        exit /b
    )
    echo Python installed successfully.

) else (
    echo Python found.
)


echo.
echo --- Step 2: Creating virtual environment '%VENV_NAME%' ---
REM 如果虛擬環境資料夾不存在，則創建它
if not exist %VENV_NAME% (
    python -m venv %VENV_NAME%
    echo Virtual environment created successfully.
) else (
    echo Virtual environment '%VENV_NAME%' already exists.
)

echo.
echo --- Step 3: Activating virtual environment ---
REM 啟用虛擬環境
call .\%VENV_NAME%\Scripts\activate.bat
echo Virtual environment activated.

echo.
echo --- Step 4: Creating requirements.txt file ---
REM 建立一個包含所有必要套件的 requirements.txt 檔案
(
    echo psutil
    echo pandas
    echo blockchain_parser
    echo numpy
    echo tensorflow
    echo yfinance
    echo matplotlib
    echo scikit-learn
    echo statsmodels
    echo pytorch
    echo torch-geometric
    echo fastai
    echo statsmodels
    echo typing 
    echo torch
    echo argparse
    echo random
    echo gc
    echo networkx
    echo datetime
) > requirements.txt
echo requirements.txt created.

echo.
echo --- Step 5: Installing required packages ---
REM 使用 pip 從 requirements.txt 安裝所有套件
pip install -r requirements.txt

echo.
echo ==================================================
echo      Installation Complete!
echo ==================================================
echo.
echo To activate this environment again in the future, run this command:
echo %VENV_NAME%\Scripts\activate.bat
echo.
pause
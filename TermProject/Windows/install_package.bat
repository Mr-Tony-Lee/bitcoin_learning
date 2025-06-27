@echo off

REM 設定虛擬環境的資料夾名稱
SET VENV_NAME=venv_bitcoin

echo.
echo --- Step 1: Checking for Python ---
REM 檢查 Python 是否已安裝並在系統路徑中
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
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
.\%VENV_NAME%\Scripts\activate.ps1
echo Virtual environment activated.

echo.
echo --- Step 4: Creating requirements.txt file ---
pip install psutil
pip install pandas
pip install blockchain_parser
pip install numpy
pip install tensorflow
pip install yfinance
pip install matplotlib
pip install scikit-learn
pip install statsmodels
pip install pytorch
pip install torch-geometric
pip install fastai
pip install statsmodels
pip install typing 
pip install torch
pip install argparse
pip install random
pip install gc
pip install networkx
pip install datetime
echo install end.

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
echo %VENV_NAME%\Scripts\activate.ps1
echo.
pause
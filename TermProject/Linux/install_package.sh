#!/bin/bash

# 設定虛擬環境的資料夾名稱
VENV_NAME="venv_bitcoin"
# 設定要使用的 Python 命令
PYTHON_CMD="python3"

echo "--- Step 1: Checking for Python and venv ---"

# 檢查 Python 3 是否已安裝，如果沒有則嘗試安裝
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Python 3 is not found. Attempting to install..."
    sudo apt update
    sudo apt install -y python3
    # 再次檢查，如果安裝失敗則退出
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: Python 3 installation failed. Please install it manually."
        exit 1
    fi
fi

# 檢查 Python 的 venv 模組是否存在，如果沒有則嘗試安裝
if ! $PYTHON_CMD -m venv -h &> /dev/null
then
    echo "Python's 'venv' module is not found. Attempting to install..."
    sudo apt update
    sudo apt install -y python3-venv
    # 再次檢查，如果安裝失敗則退出
    if ! $PYTHON_CMD -m venv -h &> /dev/null; then
        echo "Error: python3-venv installation failed. Please install it manually."
        exit 1
    fi
fi

echo "Python 3 and venv module found."
echo

echo "--- Step 2: Creating virtual environment '$VENV_NAME' ---"
# 如果虛擬環境資料夾不存在，則創建它
if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_CMD -m venv "$VENV_NAME"
    echo "Virtual environment created successfully."
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi
echo

echo "--- Step 3: Activating virtual environment ---"
# 啟用虛擬環境
source "$VENV_NAME/bin/activate"
echo "Virtual environment activated for this script's session."
echo

echo "--- Step 4: Creating requirements.txt file ---"
# 建立一個包含所有必要套件的 requirements.txt 檔案
cat > requirements.txt << EOF
psutil
pandas
blockchain_parser
numpy
tensorflow
yfinance
matplotlib
scikit-learn
statsmodels
pytorch
torch-geometric
fastai
statsmodels
typing 
torch
argparse
random
gc
networkx
datetime
EOF
echo "requirements.txt created."
echo

echo "--- Step 5: Installing required packages ---"
# 使用 pip 從 requirements.txt 安裝所有套件
pip install -r requirements.txt
echo

echo "=================================================="
echo "      Installation Complete!"
echo "=================================================="
echo
echo "To activate this environment in your terminal, run:"
echo "source $VENV_NAME/bin/activate"
echo

echo "--- Step 6: chmod the sh file ---"
# 設定腳本檔案的執行權限
chmod +x run_Meta-IFD.sh
echo "--- run_Meta-IFD.sh +x ---"
chmod +x run_BitcoinPrediction.sh
echo "--- run_BitcoinPrediction.sh +x ---"
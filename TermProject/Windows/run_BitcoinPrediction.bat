@echo off
TITLE Prediction Runner

echo ==================================================
echo       Entering BitcoinPrediction Directory...
echo ==================================================
echo.

echo ==================================================
echo       Start BitcoinPrediction Execution !
echo ==================================================
echo.

REM 進入 BitcoinPrediction-master/Bpredict 資料夾並執行腳本
cd BitcoinPrediction-master
python Bitcoinprediction.py 

echo.
echo ==================================================
echo       BitcoinPrediction Execution Complete!
echo ==================================================
echo.

echo ==================================================
echo       Start StockPrediction Execution !
echo ==================================================
echo.

REM 進入 StockPredict 資料夾並執行腳本
cd StockPredict
python PredictStockPricesRNN.py
cd ..

echo.
echo ==================================================
echo       StockPrediction Execution Complete!
echo ==================================================
echo.

REM 返回上一層資料夾
cd ..

echo ==================================================
echo       Back From BitcoinPrediction Directory!
echo ==================================================
echo.
pause
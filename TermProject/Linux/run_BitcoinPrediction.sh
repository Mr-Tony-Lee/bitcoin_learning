echo "=================================================="
echo "      Entering BitcoinPrediction Directory..."
echo "=================================================="
echo

echo "=================================================="
echo "      Start BitcoinPrediction Execution !"
echo "=================================================="
echo

cd BitcoinPrediction-master/Bpredict
python3 Bitcoinprediction.py 
cd ..

echo "=================================================="
echo "      BitcoinPrediction Execution Complete!"
echo "=================================================="
echo

echo "=================================================="
echo "      Start StockPrediction Execution !"
echo "=================================================="
echo

cd StockPredict
python3 PredictStockPricesRNN.py
cd ..

echo "=================================================="
echo "      StockPrediction Execution Complete!"
echo "=================================================="
echo

cd ..

echo "=================================================="
echo "      Back From BitcoinPrediction Directory!"
echo "=================================================="
echo
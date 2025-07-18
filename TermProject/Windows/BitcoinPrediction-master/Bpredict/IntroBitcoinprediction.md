# Bitcoin 價格預測專案分析

## 專案簡介

本專案旨在利用三種不同的機器學習與統計模型，對比特幣（BTC-USD）的歷史價格數據進行分析與預測。目標是比較各個模型的預測能力，並找出在這次的實驗設置下，哪種方法最為準確。

使用的模型包含：
1.  **隨機森林 (Random Forest Regressor)**
2.  **長短期記憶網絡 (LSTM - Long Short-Term Memory)**
3.  **ARIMA (Autoregressive Integrated Moving Average)**

專案的完整流程涵蓋了從數據下載、特徵工程、模型訓練、步進預測到結果評估與視覺化的所有步驟。

---

## 原理與流程

### 1. 數據獲取與前處理 (`download_data`)

-   **數據源**: 使用 `yfinance` 函式庫從 Yahoo Finance 下載過去五年的比特幣日度價格數據。
-   **數據清洗**:
    -   處理 `yfinance` 可能回傳的多層級(MultiIndex)欄位，將其扁平化為單層欄位（如 'Open', 'High', 'Low', 'Close'）。
    -   選取 'Open', 'High', 'Low', 'Close', 'Volume' 這五個核心欄位。
    -   為了方便除錯與驗證，將下載的原始數據保存為 CSV 檔案 (`./data_output/BTC-USD_5y_raw_data.csv`)。

### 2. 模型準備與訓練

#### a. 隨機森林 (Random Forest)

-   **原理**: 隨機森林是一種集成學習模型，它透過建立多個決策樹並綜合其結果來進行預測。它擅長處理特徵之間的非線性關係。
-   **過程 (`prepare_random_forest_data`, `run_random_forest`)**:
    1.  **特徵工程**: 除了原始的 OHLCV 數據，還計算了 7 日簡單移動平均線 (`SMA_7`) 和 30 日簡單移動平均線 (`SMA_30`) 作為額外特徵，以提供更多市場趨勢的資訊。
    2.  **目標變數**: 預測的目標 (`target`) 是下一天的收盤價 (`Close`)。
    3.  **數據分割**: 數據按時間順序分割，前 80% 作為訓練集，後 20% 作為測試集。
    4.  **訓練**: 使用訓練集數據來訓練 `RandomForestRegressor` 模型。

#### b. 長短期記憶網絡 (LSTM)

-   **原理**: LSTM 是一種特殊的循環神經網絡 (RNN)，非常適合處理和預測時間序列數據。它能學習到數據中的長期依賴性。
-   **過程 (`prepare_lstm_data`, `run_lstm`)**:
    1.  **數據標準化**: 僅使用 `Close` 價格，並透過 `MinMaxScaler` 將其數值縮放到 0 到 1 之間。這有助於神經網絡的穩定訓練。
    2.  **序列創建**: 將數據轉換為監督式學習的序列格式。例如，使用過去 3 天的價格 (`look_back=3`) 作為輸入 (X)，來預測第 4 天的價格 (y)。
    3.  **模型建構**: 建立一個包含兩層 LSTM 和一個輸出層 (Dense) 的 `Sequential` 模型。
    4.  **訓練與評估**: 模型訓練 50 個週期 (epochs)，並計算在原始價格尺度（反標準化後）的均方根誤差 (RMSE)。

#### c. ARIMA 模型

-   **原理**: ARIMA 是一個經典的時間序列統計模型，它利用數據本身的自相關性、趨勢（差分）和過去的預測誤差來進行預測。
-   **過程 (`run_arima`)**:
    1.  **數據對齊**: 為了公平比較，ARIMA 的訓練集和測試集時間段與隨機森林的測試集完全對齊。
    2.  **步進驗證 (Walk-Forward Validation)**: 這是此模型最關鍵的部分。它並非一次性預測所有未來數據，而是：
        -   用現有歷史數據預測下一天。
        -   將下一天的**真實觀測值**加入歷史數據中。
        -   重複以上步驟，直到預測完所有測試集數據。
        這種方法更貼近真實世界的預測情境，因為我們每天都會獲得新的市場數據。

### 3. 結果匯總與視覺化 (`main`, `plot_predictions`)

1.  **對齊預測**: 由於不同模型的數據處理方式略有差異，程式碼確保了所有模型的最終預測結果都與相同的測試日期對齊。
2.  **建立比較表**: 將三個模型的預測值與真實價格放入一個 `pandas.DataFrame` 中，方便比較。
3.  **計算 RMSE**: 計算每個模型在測試集上的均方根誤差 (RMSE)，這是一個衡量預測誤差大小的關鍵指標，數值越小代表模型越準確。
4.  **繪製圖表**: 使用 `matplotlib` 將真實價格與三個模型的預測價格繪製在同一張圖上，並保存為圖片 (`prediction_comparison.png`)，以進行直觀的視覺比較。

---

## 執行結果與結論

根據您提供的最新一次執行日誌：

### 效能報告 (RMSE)

-   **Random Forest RMSE**: $22,399.33
-   **LSTM Test RMSE**: $2,585.75
-   **ARIMA RMSE**: $2,127.42

### 結論

1.  **模型表現**: 從 RMSE 指標來看，**ARIMA 模型表現最佳**（誤差最小），緊隨其後的是 **LSTM 模型**。這兩個模型都顯著優於隨機森林模型。

2.  **方法有效性**:
    -   ARIMA 的成功歸功於其經典的統計學基礎和**步進驗證**的預測策略，使其能不斷根據最新資訊進行調整。
    -   LSTM 作為專為序列數據設計的神經網絡，也展現了強大的學習能力，能夠捕捉到價格變動的模式。
    -   隨機森林的表現較差，可能原因是在這個問題中，僅加入兩條移動平均線的特徵工程不足以讓模型捕捉到複雜的金融市場動態。時間序列的內在順序和依賴性比單純的特徵組合更為重要。

3.  **總結**: 對於比特幣這種高度時間相關的數據，專門為時間序列設計的模型（如 ARIMA 和 LSTM）比通用的監督式學習模型（如隨機森林）表現得更為出色。此專案成功地建立了一個有效的框架來比較不同預測方法，並清楚地展示了在當前設定下，ARIMA 和 LSTM 是更優越的選擇
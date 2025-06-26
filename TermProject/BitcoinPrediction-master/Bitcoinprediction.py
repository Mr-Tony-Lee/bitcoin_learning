#!/usr/bin/env python3
"""Download Bitcoin prices and forecast using three models."""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def download_data(ticker="BTC-USD", period="5y"): # 定義一個名為 download_data 的函數，設定預設股票代碼和時間週期
    print(f"Downloading {ticker} data for {period}...") # 在控制台印出正在下載數據的提示訊息
    df = yf.download(ticker, period=period) # 使用 yfinance 的 download 函數下載指定股票和時間的歷史數據
    
    if isinstance(df.columns, pd.MultiIndex): # 檢查下載的 DataFrame 的欄位是否為多層級索引
        new_columns = [] # 初始化一個空列表，用來存放新的、單層級的欄位名稱
        for col_tuple in df.columns: # 遍歷所有的多層級欄位元組
            meaningful_part = [part for part in col_tuple if part and part != ticker] # 從元組中篩選出非空且不等於股票代碼的部分
            if meaningful_part: # 如果找到了有意義的欄位名部分
                new_columns.append(meaningful_part[-1]) # 將有意義部分的最後一個元素（通常是 'Close', 'Open' 等）加入新欄位列表
            else: # 如果沒有找到有意義的部分
                new_columns.append('_'.join(col_tuple)) # 將元組中的所有部分用底線連接起來作為備用欄位名
        df.columns = new_columns # 將 DataFrame 的欄位更新為處理過後的新欄位名
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy() # 選取 'Open', 'High', 'Low', 'Close', 'Volume' 這五個欄位，並創建一個副本以避免警告
    
    print("Data downloaded:\n", df.head()) # 印出數據下載完成的訊息以及數據框的前五行內容
    
    output_dir = './data_output' # 設定一個字串變數，表示保存數據的輸出文件夾路徑
    os.makedirs(output_dir, exist_ok=True) # 使用 os 模組創建輸出文件夾，如果文件夾已存在則不執行任何操作
    csv_path = os.path.join(output_dir, f"{ticker}_{period}_raw_data.csv") # 組合出要保存的 CSV 檔案的完整路徑和檔名
    df.to_csv(csv_path) # 將整理好的 DataFrame 數據保存為 CSV 檔案
    print(f"Raw data saved to: {csv_path}") # 印出提示訊息，告知原始數據已保存到指定路徑

    return df # 回傳處理好的 DataFrame


def prepare_random_forest_data(df): # 定義一個為隨機森林模型準備數據的函數
    print("Preparing data for RandomForest...") # 印出提示訊息，表示開始為隨機森林準備數據
    df = df.copy() # 創建 DataFrame 的一個副本，以避免在後續操作中修改到原始傳入的數據
    df['SMA_7'] = df['Close'].rolling(window=7).mean() # 計算7日簡單移動平均線（SMA），並將其作為新的一欄 'SMA_7' 加入 DataFrame
    df['SMA_30'] = df['Close'].rolling(window=30).mean() # 計算30日簡單移動平均線（SMA），並將其作為新的一欄 'SMA_30' 加入 DataFrame
    
    df["Close"] = pd.to_numeric(df["Close"]) # 確保 'Close' 欄位是數值類型，以便進行計算
    df["target"] = df["Close"].shift(-1) # 將 'Close' 欄位的數據向上移動一格，創建 'target' 欄位，代表下一天的收盤價
    df = df.dropna() # 刪除所有包含空值（NaN）的行，這些空值是因計算移動平均和 shift 操作產生的
    features = ["Open", "High", "Low", "Close", "Volume", "SMA_7", "SMA_30"] # 定義一個列表，包含所有要用作模型輸入的特徵欄位名稱
    for feature in features: # 遍歷特徵列表中的每一個特徵名稱
        df[feature] = pd.to_numeric(df[feature]) # 確保每一個特徵欄位都是數值類型
    X = df[features] # 將包含所有特徵欄位的數據提取出來，賦值給變數 X
    y = df["target"] # 將目標欄位 'target' 的數據提取出來，賦值給變數 y
    X_train, X_test, y_train, y_test = train_test_split( # 使用 train_test_split 函數分割數據
        X, y, test_size=0.2, shuffle=False # 將 X 和 y 分割，測試集佔20%，並設定 shuffle=False 確保數據按時間順序分割
    )
    test_dates = y_test.index # 獲取測試集 y_test 的日期索引
    return X_train, X_test, y_train, y_test, test_dates # 回傳分割好的訓練數據、測試數據和測試集的日期


def run_random_forest(X_train, X_test, y_train, y_test): # 定義一個訓練並運行隨機森林模型的函數
    print("\n=== Training RandomForestRegressor ===") # 印出一個標題，表示開始訓練隨機森林回歸模型
    model = RandomForestRegressor(n_estimators=100, random_state=0) # 初始化隨機森林回歸模型，設定建立100棵決策樹，並固定隨機種子以保證結果可複現
    model.fit(X_train, y_train) # 使用訓練數據 X_train 和 y_train 來擬合（訓練）模型
    preds = model.predict(X_test) # 使用訓練好的模型對測試集 X_test 進行預測
    rmse = mean_squared_error(y_test, preds) ** 0.5 # 計算預測值(preds)和真實值(y_test)之間的均方根誤差(RMSE)
    print(f"Random Forest RMSE: {rmse:.2f}") # 印出隨機森林模型的 RMSE 值，格式化到小數點後兩位
    print("RandomForest training complete!") # 印出隨機森林訓練完成的訊息
    return model, preds, rmse # 回傳訓練好的模型、在測試集上的預測結果和計算出的 RMSE


def prepare_lstm_data(df, look_back=3): # 定義一個為 LSTM 模型準備數據的函數，預設回看天數為3
    print("Preparing data for LSTM...") # 印出提示訊息，表示開始為 LSTM 準備數據
    values = df[["Close"]].values.astype("float32") # 從 DataFrame 中選取 'Close' 欄位，轉換為 numpy 陣列，並設定數據類型為 float32
    scaler = MinMaxScaler() # 初始化一個 MinMaxScaler 物件，用於將數據縮放到 [0, 1] 區間
    scaled = scaler.fit_transform(values) # 對收盤價數據進行擬合和標準化轉換

    X, y = [], [] # 初始化兩個空列表，分別用於存放輸入序列 (X) 和目標值 (y)
    for i in range(len(scaled) - look_back - 1): # 遍歷標準化後的數據，以創建有時間步的序列數據
        X.append(scaled[i:(i + look_back), 0]) # 將從索引 i 到 i+look_back 的數據作為一個輸入序列，添加到 X 列表
        y.append(scaled[i + look_back, 0]) # 將第 i+look_back 個數據作為對應的目標值，添加到 y 列表
    X = np.array(X) # 將列表 X 轉換為 numpy 陣列
    y = np.array(y) # 將列表 y 轉換為 numpy 陣列

    train_size = int(len(X) * 0.67) # 計算訓練集的大小，設定為總數據量的 67%
    X_train, X_test = X[:train_size], X[train_size:] # 將輸入序列 X 按計算出的比例分割為訓練集和測試集
    y_train, y_test = y[:train_size], y[train_size:] # 將目標值 y 按計算出的比例分割為訓練集和測試集

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1])) # 將訓練集 X 的形狀重塑為 LSTM 模型需要的 [樣本數, 時間步, 特徵數] 格式
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])) # 將測試集 X 的形狀重塑為 LSTM 模型需要的格式
    return X_train, X_test, y_train, y_test, scaler # 回傳分割和重塑後的數據以及用於標準化的 scaler 物件


def run_lstm(X_train, X_test, y_train, y_test, scaler): # 定義一個訓練並運行 LSTM 模型的函數
    print("\n=== Training LSTM ===") # 印出一個標題，表示開始訓練 LSTM 模型
    model = Sequential() # 初始化一個 Keras 的序貫模型
    model.add(LSTM(256, return_sequences=True, input_shape=(1, X_train.shape[2]))) # 添加一個有256個單元的 LSTM 層，因下一層還是 LSTM，所以設定 return_sequences=True
    model.add(LSTM(256)) # 添加第二個有256個單元的 LSTM 層
    model.add(Dense(1)) # 添加一個全連接層(Dense)作為輸出層，只有一個單元，用於輸出最終的預測值
    model.compile(loss="mean_squared_error", optimizer="adam") # 編譯模型，設定損失函數為均方誤差，優化器為 adam
    model.fit(X_train, y_train, epochs=50, batch_size=50, verbose=1, shuffle=False) # 訓練模型，設定50個訓練週期，批次大小為50，顯示進度，且不打亂數據順序

    train_pred = model.predict(X_train) # 使用訓練好的模型對訓練集進行預測
    test_pred = model.predict(X_test) # 使用訓練好的模型對測試集進行預測
    
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)) # 將標準化的測試集真實目標值反轉，還原為原始價格尺度
    test_pred_unscaled = scaler.inverse_transform(test_pred) # 將標準化的測試集預測結果反轉，還原為原始價格尺度
    
    unscaled_rmse = mean_squared_error(y_test_unscaled, test_pred_unscaled) ** 0.5 # 計算在原始價格尺度上的均方根誤差(RMSE)
    scaled_rmse = mean_squared_error(y_test, test_pred) ** 0.5 # 計算在標準化尺度上的均方根誤差(RMSE)

    print(f"LSTM Test RMSE (scaled): {scaled_rmse:.4f}") # 印出標準化後的 RMSE，格式化到小數點後四位
    print(f"LSTM Test RMSE (unscaled): ${unscaled_rmse:.2f}") # 印出原始價格尺度上的 RMSE，格式化到小數點後兩位
    print("LSTM training complete!") # 印出 LSTM 訓練完成的訊息
    return model, test_pred.flatten(), unscaled_rmse # 回傳訓練好的模型、扁平化的預測結果和未標準化的 RMSE


def prepare_arima_data(series, test_size=0.2): # 定義一個為 ARIMA 模型準備數據的函數
    print("Preparing data for ARIMA...") # 印出提示訊息，表示開始為 ARIMA 準備數據
    series = pd.to_numeric(series) # 確保傳入的時間序列數據是數值類型
    split = int(len(series) * (1 - test_size)) # 計算訓練集和測試集的分割點位置，測試集佔20%
    train, test = series[:split], series[split:] # 將時間序列按計算出的分割點分割為訓練集和測試集
    return train, test # 回傳分割好的訓練集和測試集


def run_arima(train_series, test_series): # 定義一個使用步進驗證方式運行 ARIMA 模型的函數
    print("\n=== Training ARIMA with Walk-Forward Validation ===") # 印出一個標題，表示開始使用步進驗證法訓練 ARIMA 模型
    history = [x for x in train_series] # 將訓練集數據複製到一個名為 history 的列表中，作為初始的歷史數據
    predictions = [] # 初始化一個空列表，用於存放每一步的預測結果
    
    for t in range(len(test_series)): # 遍歷測試集中的每一個時間點
        model = ARIMA(history, order=(5, 1, 0)) # 使用當前的歷史數據初始化 ARIMA 模型，(p,d,q) 參數設為 (5,1,0)
        model_fit = model.fit() # 擬合（訓練）ARIMA 模型
        output = model_fit.forecast() # 使用訓練好的模型預測下一個時間點的值
        yhat = output[0] # 從預測輸出中獲取預測值
        predictions.append(yhat) # 將預測結果添加到 predictions 列表中
        history.append(test_series.iloc[t]) # 將當前時間點的真實觀測值添加到歷史數據中，用於下一次預測
        if (t + 1) % 50 == 0: # 每預測50次，執行一次以下操作
            print(f"ARIMA predicted {t+1}/{len(test_series)}") # 印出目前的預測進度

    rmse = mean_squared_error(test_series, predictions) ** 0.5 # 在所有預測完成後，計算全部預測值和真實值之間的均方根誤差
    print(f"ARIMA Walk-Forward RMSE: {rmse:.2f}") # 印出 ARIMA 模型的最終 RMSE 值
    print("ARIMA training complete!") # 印出 ARIMA 訓練完成的訊息
    return model_fit, pd.Series(predictions, index=test_series.index), rmse # 回傳最後一次的模型、帶有日期索引的預測序列和 RMSE


def plot_predictions(compare_df): # 定義一個繪製預測結果比較圖的函數
    plt.figure(figsize=(14, 7)) # 創建一個新的圖表，並設定其大小為 14x7 英吋
    plt.plot(compare_df['Date'], compare_df['Actual'], label='Actual Price', color='blue') # 繪製真實價格曲線，設定標籤和顏色
    plt.plot(compare_df['Date'], compare_df['RandomForest'], label='Random Forest Prediction', color='green', linestyle='--') # 繪製隨機森林的預測曲線，設定標籤、顏色和線型
    plt.plot(compare_df['Date'], compare_df['LSTM'], label='LSTM Prediction', color='red', linestyle='--') # 繪製 LSTM 的預測曲線，設定標籤、顏色和線型
    plt.plot(compare_df['Date'], compare_df['ARIMA'], label='ARIMA Prediction', color='orange', linestyle='--') # 繪製 ARIMA 的預測曲線，設定標籤、顏色和線型
    
    plt.title('Bitcoin Price Prediction Comparison') # 設定圖表的標題
    plt.xlabel('Date') # 設定 X 軸的標籤為 'Date'
    plt.ylabel('Price (USD)') # 設定 Y 軸的標籤為 'Price (USD)'
    plt.legend() # 顯示圖例，用於標識每條曲線
    plt.grid(True) # 顯示圖表的網格線
    
    plot_path = os.path.join('./data_output', 'prediction_comparison.png') # 組合出要保存的圖片檔案在 data_output 文件夾下的完整路徑
    plt.savefig(plot_path) # 將圖表保存到指定路徑
    print(f"\nPrediction plot saved to: {plot_path}") # 印出圖片已保存的路徑
    plt.savefig("bitcoin_prediction_comparison.png") # 將圖表也保存在當前程式運行的目錄下


def main(): # 定義主執行函數
    print("Starting Bitcoin prediction pipeline...") # 印出整個預測流程開始的訊息
    df = download_data() # 呼叫 download_data 函數下載並獲取處理好的數據
    
    print(f"DEBUG: Max Close price in downloaded data: {df['Close'].max()}") # 印出下載數據中的最高收盤價，用於除錯
    print(f"DEBUG: Min Close price in downloaded data: {df['Close'].min()}") # 印出下載數據中的最低收盤價，用於除錯

    X_train, X_test, y_train, y_test, test_dates = prepare_random_forest_data(df) # 為隨機森林模型準備數據並獲取分割後的數據集
    
    print(f"DEBUG: y_test shape after RF split: {y_test.shape}") # 印出隨機森林測試集 y 的形狀，用於除錯
    print(f"DEBUG: y_test (RF test set actuals) head:\n{y_test.head()}") # 印出隨機森林測試集 y 的前五行數據，用於除錯
    print(f"DEBUG: y_test (RF test set actuals) tail:\n{y_test.tail()}") # 印出隨機森林測試集 y 的末五行數據，用於除錯

    rf_model, rf_preds, rf_rmse = run_random_forest(X_train, X_test, y_train, y_test) # 訓練並運行隨機森林模型，獲取模型、預測結果和 RMSE

    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(df) # 為 LSTM 模型準備數據
    lstm_model, lstm_preds_scaled, lstm_rmse = run_lstm( # 訓練並運行 LSTM 模型
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler
    )
    lstm_preds_all = scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten() # 將 LSTM 的標準化預測結果反標準化並轉換為一維陣列

    align_len = min(len(test_dates), len(lstm_preds_all)) # 計算需要對齊的長度，取決於隨機森林測試集和 LSTM 預測結果的較小者
    lstm_preds = lstm_preds_all[-align_len:] # 截取 LSTM 預測結果的最後部分以對齊長度
    test_dates = test_dates[-align_len:] # 截取測試日期的最後部分以對齊長度
    y_test_aligned = y_test[-align_len:] # 截取隨機森林測試集 y 的最後部分以對齊長度，創建一個新的對齊後變數

    start_test_date = test_dates.min() # 獲取對齊後測試集的最早日期
    train_series = df.loc[df.index < start_test_date, "Close"] # 選取所有早於測試集開始日期的收盤價作為 ARIMA 的訓練序列
    test_series = df.loc[test_dates, "Close"] # 選取與對齊後測試日期完全對應的收盤價作為 ARIMA 的測試序列

    arima_model, arima_preds, arima_rmse = run_arima(train_series, test_series) # 訓練並運行 ARIMA 模型

    compare_df = pd.DataFrame({ # 創建一個 DataFrame 來匯總所有模型的預測結果和真實值，以便比較
        "Date": test_dates, # 設定 'Date' 欄位為對齊後的測試日期
        "Actual": y_test_aligned.values, # 設定 'Actual' 欄位為對齊後的真實值
        "RandomForest": rf_preds[-align_len:], # 設定 'RandomForest' 欄位為對齊後的隨機森林預測值
        "LSTM": lstm_preds, # 設定 'LSTM' 欄位為對齊後的 LSTM 預測值
        "ARIMA": arima_preds.values, # 設定 'ARIMA' 欄位為對齊後的 ARIMA 預測值
    })

    print("\n=== Prediction Comparison (last 5 points) ===") # 印出結果比較表的標題
    print(compare_df.tail(5).to_string(index=False)) # 印出比較表的最後五行內容，並且不顯示 DataFrame 的索引

    print("\n=== RMSE Report ===") # 印出 RMSE 效能報告的標題
    print(f"Random Forest RMSE: ${rf_rmse:.2f}") # 印出隨機森林的 RMSE
    print(f"LSTM Test RMSE: ${lstm_rmse:.2f}") # 印出 LSTM 的 RMSE (使用未標準化的值)
    print(f"ARIMA RMSE: ${arima_rmse:.2f}") # 印出 ARIMA 的 RMSE
    print("Pipeline complete!") # 印出整個流程完成的訊息

    plot_predictions(compare_df) # 呼叫繪圖函數來視覺化所有模型的預測結果


if __name__ == "__main__": # 檢查此腳本是否是作為主程式被直接執行
    main() # 如果是直接執行，則呼叫 main 函數來啟動整個流程

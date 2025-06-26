import pandas as pd 
import numpy as np 
import tensorflow as tf 
import yfinance as yf 
import matplotlib.pyplot as plt 
import os 

def get_data(): # 定義一個名為 get_data 的函數，用於下載和預處理數據
    msft = yf.Ticker("MSFT") # 創建一個代表微軟公司股票 (MSFT) 的 Ticker 物件
    df = msft.history(period="5d", interval="5m") # 獲取微軟過去5天、每5分鐘的歷史股價數據

    df = df.dropna() # 刪除數據中所有包含缺失值 (NaN) 的行
    df = df[['Close']] # 從 DataFrame 中只選取 'Close' (收盤價) 這一欄
    
    original_close = df['Close'].copy() # 複製一份未經標準化的原始收盤價數據，用於後續繪圖
    
    scaler = df['Close'].iloc[0] # 選取第一個收盤價作為標準化的基準值 (scaler)
    df['Close'] = df['Close'] / scaler # 將所有收盤價除以基準值，進行標準化
    
    output_dir = './data_output' # 設定一個字串變數，表示保存數據的輸出文件夾路徑
    if not os.path.exists(output_dir): # 檢查輸出文件夾是否不存在
        os.makedirs(output_dir) # 如果文件夾不存在，則創建它
    df.to_csv(os.path.join(output_dir, 'msft_stock_data.csv')) # 將處理後的數據保存為 CSV 檔案
    return df, original_close, scaler # 回傳標準化後的數據、原始收盤價數據和標準化基準值

def create_dataset(data): # 定義一個函數，用於創建監督式學習的數據集 (X=t, y=t+1)
    X, y = [], [] # 初始化兩個空列表，X 用於存放特徵，y 用於存放目標
    for i in range(len(data) - 1): # 遍歷數據，但不包括最後一個元素
        X.append(data[i]) # 將當前時間點的數據 (t) 作為特徵加入 X
        y.append(data[i + 1]) # 將下一個時間點的數據 (t+1) 作為目標加入 y
    return np.array(X), np.array(y) # 將列表 X 和 y 轉換為 numpy 陣列並回傳

def split_data(df, train_frac=0.8): # 定義一個函數，用於將數據分割為訓練集和測試集，預設訓練集比例為80%
    train_size = int(len(df) * train_frac) # 計算訓練集的大小
    train_data = df.iloc[:train_size] # 選取前 80% 的數據作為訓練數據
    test_data = df.iloc[train_size:] # 選取後 20% 的數據作為測試數據

    X_train, y_train = create_dataset(train_data['Close'].values) # 從訓練數據中創建特徵 (X_train) 和目標 (y_train)
    X_test, y_test = create_dataset(test_data['Close'].values) # 從測試數據中創建特徵 (X_test) 和目標 (y_test)
    
    return X_train, y_train, X_test, y_test # 回傳分割好的訓練集和測試集

def train_rnn(X_train, y_train): # 定義一個函數，用於創建和訓練 RNN 模型
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, 1)) # 將訓練數據 X 重塑為 RNN 需要的 3D 格式 [樣本數, 時間步, 特徵數]

    model = tf.keras.Sequential() # 初始化一個 Keras 的序貫模型
    model.add(tf.keras.layers.LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))) # 添加一個有50個單元的 LSTM 層，並指定輸入數據的形狀
    model.add(tf.keras.layers.Dense(units=1)) # 添加一個全連接層作為輸出層，只有一個單元，用於輸出預測值
    model.compile(loss='mean_squared_error', optimizer='adam') # 編譯模型，設定損失函數為均方誤差，優化器為 adam

    print(f"X_train_reshaped shape: {X_train_reshaped.shape}") # 印出重塑後訓練數據的形狀，用於除錯

    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=1, verbose=2) # 訓練模型，設定100個訓練週期，批次大小為1，並顯示訓練過程

    return model # 回傳訓練好的模型

def evaluate_rnn(model, X_test, y_test): # 定義一個函數，用於評估 RNN 模型
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, 1)) # 將測試數據 X 重塑為 RNN 需要的 3D 格式
    predictions = model.predict(X_test_reshaped) # 使用訓練好的模型對重塑後的測試數據進行預測

    mse = np.mean((predictions.flatten() - y_test)**2) # 計算預測值和真實值之間的均方誤差 (MSE)

    return mse, predictions.flatten() # 回傳計算出的 MSE 和扁平化後的預測結果

def plot_results(original_data, test_start_index, predictions, scaler): # 定義一個函數，用於繪製和保存結果圖表
    original_prices = original_data.values # 獲取原始收盤價的 numpy 陣列
    predicted_prices = predictions * scaler # 將標準化的預測結果乘以基準值，反標準化為原始價格

    actual_test_prices = original_prices[test_start_index+1:] # 從原始價格中獲取測試集的真實價格部分
    test_dates = original_data.index[test_start_index+1:] # 從原始數據中獲取測試集的日期索引

    min_len = min(len(actual_test_prices), len(predicted_prices)) # 確保真實值和預測值的長度一致，取較小者
    
    plt.figure(figsize=(14, 7)) # 創建一個新的圖表，並設定其大小為 14x7 英吋
    plt.plot(test_dates[:min_len], actual_test_prices[:min_len], label='Actual Price', color='blue') # 繪製真實價格曲線
    plt.plot(test_dates[:min_len], predicted_prices[:min_len], label='Predicted Price', color='red', linestyle='--') # 繪製預測價格曲線
    plt.title('MSFT Stock Price Prediction') # 設定圖表標題
    plt.xlabel('Date') # 設定 X 軸標籤
    plt.ylabel('Price (USD)') # 設定 Y 軸標籤
    plt.legend() # 顯示圖例
    plt.grid(True) # 顯示網格線
    
    output_dir = './data_output' # 設定輸出文件夾路徑
    if not os.path.exists(output_dir): # 檢查文件夾是否存在
        os.makedirs(output_dir) # 如果不存在，則創建文件夾
    plot_path = os.path.join(output_dir, 'rnn_prediction.png') # 組合出要保存的圖片檔案的完整路徑
    plt.savefig(plot_path) # 將圖表保存到指定路徑
    print(f"\nPrediction plot saved to: {plot_path}") # 印出圖片已保存的路徑

def main(): # 定義主執行函數
    df, original_close, scaler = get_data() # 呼叫 get_data 函數獲取數據

    X_train, y_train, X_test, y_test = split_data(df) # 呼叫 split_data 函數分割數據

    model = train_rnn(X_train, y_train) # 呼叫 train_rnn 函數訓練模型

    mse, predictions = evaluate_rnn(model, X_test, y_test) # 呼叫 evaluate_rnn 函數評估模型並獲取預測
    print(f"Mean Squared Error: {mse}") # 印出模型的均方誤差

    test_start_index = len(y_train) # 計算測試集在原始數據中的起始索引位置
    plot_results(original_close, test_start_index, predictions, scaler) # 呼叫 plot_results 函數繪製結果

if __name__ == "__main__": # 檢查此腳本是否是作為主程式被直接執行
    main() # 如果是，則呼叫 main 函數   

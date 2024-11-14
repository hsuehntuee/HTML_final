import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 定義必要的欄位
required_columns = [
    'away_batting_wpa_bat_skew', 
    'away_batting_RBI_skew', 
    'home_pitching_H_batters_faced_std', 
    'home_pitching_BB_batters_faced_mean',
    'home_pitching_BB_batters_faced_std',
    'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 
    'home_pitching_leverage_index_avg_std', 
    'home_pitching_wpa_def_skew', 
    'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_std', 
    'away_pitching_H_batters_faced_mean',
    'away_pitching_H_batters_faced_std', 
    'away_pitching_leverage_index_avg_skew', 
    'away_pitching_wpa_def_mean', 
    'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 
    'home_pitcher_earned_run_avg_mean',
    'home_pitcher_earned_run_avg_std', 
    'home_pitcher_SO_batters_faced_mean',
    'home_pitcher_H_batters_faced_mean',
    'home_pitcher_BB_batters_faced_skew', 
    'home_pitcher_leverage_index_avg_mean',
    'home_pitcher_leverage_index_avg_std', 
    'home_pitcher_leverage_index_avg_skew',
    'home_pitcher_wpa_def_mean', 
    'home_pitcher_wpa_def_std', 
    'home_pitcher_wpa_def_skew',
    'away_pitcher_earned_run_avg_mean', 
    'away_pitcher_earned_run_avg_std', 
    'away_pitcher_earned_run_avg_skew', 
    'away_pitcher_SO_batters_faced_mean', 
    'away_pitcher_SO_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 
    'away_pitcher_BB_batters_faced_mean',
    'away_pitcher_BB_batters_faced_std'
]

# 讀取訓練資料
train_df = pd.read_csv('train_data.csv')

if 'date'  in train_df:
    print("haha") 
    print(train_df['date'])
print(train_df.columns)
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())
train_df['date'] = pd.to_datetime(train_df['date'], format='%Y/%m/%d', errors='coerce')  # 確保日期欄位為 datetime 格式
train_df.sort_values('date', inplace=True)  # 按日期排序

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)

# 讀取測試資料
test_df = pd.read_csv('same_season_test_data.csv')
test_df[required_columns] = test_df[required_columns].fillna(test_df[required_columns].mean())
test_df['date'] = pd.to_datetime(test_df['date'])  # 確保日期欄位為 datetime 格式
test_df.sort_values('date', inplace=True)  # 按日期排序
X_test = test_df[required_columns].to_numpy().astype(float)

# 調整數據形狀以適應 RNN 模型
X = X.reshape((X.shape[0], X.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 定義模型訓練函數
def build_and_train_model(X, Y, X_test, output_filename='result.csv'):
    model = Sequential()
    
    # LSTM 第一層
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # LSTM 第二層
    model.add(LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # LSTM 第三層
    model.add(LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # Dense 層
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    # 編譯模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # EarlyStopping 回調
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 滾動預測
    predictions = []
    for i in range(len(X_test)):
        # 訓練模型，只使用到 i 位置的訓練資料
        model.fit(X[:len(X) - len(X_test) + i + 1], Y[:len(X) - len(X_test) + i + 1], epochs=5, batch_size=32, verbose=0, callbacks=[early_stopping])
        
        # 預測下一個值
        prediction = model.predict(X_test[i:i+1])
        predictions.append(prediction[0, 0])
        
        # 將預測結果添加到訓練集的最後以便進行後續預測
        if i < len(X_test) - 1:
            new_data = np.array([[prediction[0, 0]]]).reshape(1, 1, 1)  # 將預測結果轉為適合 RNN 的形狀
            X = np.concatenate((X, new_data), axis=0)

    # 將預測結果轉為 DataFrame 並保存
    result_df = pd.DataFrame({'date': test_df['date'], 'home_team_win': predictions})
    result_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

# 執行模型訓練和預測
build_and_train_model(X, Y, X_test, output_filename='result57_77.csv')

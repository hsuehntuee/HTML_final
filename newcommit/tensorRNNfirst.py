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
 #'season',
    #'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 'home_batting_onbase_plus_slugging_10RA', 
    #'home_batting_leverage_index_avg_10RA', 'home_batting_RBI_10RA', 'away_batting_batting_avg_10RA', 
    #'away_batting_onbase_perc_10RA', 'away_batting_onbase_plus_slugging_10RA', 'away_batting_leverage_index_avg_10RA', 
    #'away_batting_RBI_10RA', 'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA', 
    #'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA', 'away_pitching_earned_run_avg_10RA', 
    #'away_pitching_SO_batters_faced_10RA', 'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA', 
    #'home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 
    #'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 
    #'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA', 'home_team_errors_mean', 'home_team_errors_std', 'home_team_errors_skew', 'away_team_errors_mean', 
    #'away_team_errors_std', 'away_team_errors_skew', 'home_team_spread_mean', 'home_team_spread_std', 
    #'home_team_spread_skew', 'away_team_spread_mean', 'away_team_spread_std', 'away_team_spread_skew', 
    #'home_team_wins_mean', 'home_team_wins_std', 'home_team_wins_skew', 'away_team_wins_mean', 'away_team_wins_std', 
    #'away_team_wins_skew', 'home_batting_batting_avg_mean', 'home_batting_batting_avg_std', 
    #'home_batting_batting_avg_skew', 'home_batting_onbase_perc_mean', 'home_batting_onbase_perc_std', 
    'home_batting_onbase_perc_skew', 
    'home_batting_onbase_plus_slugging_mean',
     'home_batting_onbase_plus_slugging_std', 
    #'home_batting_onbase_plus_slugging_skew', 'home_batting_leverage_index_avg_mean', 
    #'home_batting_leverage_index_avg_std', 'home_batting_leverage_index_avg_skew', 'home_batting_wpa_bat_mean', 
    #'home_batting_wpa_bat_std', 'home_batting_wpa_bat_skew', 'home_batting_RBI_mean', 'home_batting_RBI_std', 
    #'home_batting_RBI_skew', 'away_batting_batting_avg_mean', 'away_batting_batting_avg_std', 
    #'away_batting_batting_avg_skew', 'away_batting_onbase_perc_mean', 'away_batting_onbase_perc_std', 
    #'away_batting_onbase_perc_skew', 'away_batting_onbase_plus_slugging_mean', 'away_batting_onbase_plus_slugging_std', 
    #'away_batting_onbase_plus_slugging_skew', 'away_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_std',

    #'away_batting_leverage_index_avg_skew', 'away_batting_wpa_bat_mean', 'away_batting_wpa_bat_std', 
    
    'away_batting_wpa_bat_skew', 
    'away_batting_RBI_mean', 
    'away_batting_RBI_std',
      'away_batting_RBI_skew', 
    'home_pitching_earned_run_avg_mean', 
    'home_pitching_earned_run_avg_std', 
    'home_pitching_earned_run_avg_skew', 
    
    
    #'home_pitching_SO_batters_faced_mean', 'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew', 
    
    'home_pitching_H_batters_faced_mean', 
    'home_pitching_H_batters_faced_std', 
    'home_pitching_H_batters_faced_skew', 
    'home_pitching_BB_batters_faced_mean',
      'home_pitching_BB_batters_faced_std',
        'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 
    'home_pitching_leverage_index_avg_std', 
    #'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean', 'home_pitching_wpa_def_std', 
    'home_pitching_wpa_def_skew', 
    'away_pitching_earned_run_avg_mean', 
    'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_std', 
    'away_pitching_SO_batters_faced_skew',
     'away_pitching_H_batters_faced_mean',
        'away_pitching_H_batters_faced_std', 
    #'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean', 'away_pitching_BB_batters_faced_std', 
    #'away_pitching_BB_batters_faced_skew', 'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std', 
    'away_pitching_leverage_index_avg_skew', 
    'away_pitching_wpa_def_mean', 
    'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 
    'home_pitcher_earned_run_avg_mean',
     'home_pitcher_earned_run_avg_std', 
    'home_pitcher_earned_run_avg_skew', 
    'home_pitcher_SO_batters_faced_mean',
      'home_pitcher_SO_batters_faced_std', 
    'home_pitcher_SO_batters_faced_skew',
     'home_pitcher_H_batters_faced_mean',
       'home_pitcher_H_batters_faced_std', 
    #'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean', 'home_pitcher_BB_batters_faced_std', 
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
    #'away_pitcher_SO_batters_faced_skew', 'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 
    'away_pitcher_BB_batters_faced_mean',
     'away_pitcher_BB_batters_faced_std', 
    'away_pitcher_BB_batters_faced_skew', 'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std', 
    'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean', 'away_pitcher_wpa_def_std', 
    #'away_pitcher_wpa_def_skew'
]

# 讀取訓練資料
train_df = pd.read_csv('newtrain.csv')
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)

# 讀取驗證資料並填補空值
validation_df = pd.read_csv('newsnewvalid.csv')
validation_df[required_columns] = validation_df[required_columns].fillna(validation_df[required_columns].mean())
X_val = validation_df[required_columns].to_numpy().astype(float)
y_true = validation_df['home_team_win'].to_numpy().astype(float)

dddcheck = pd.read_csv('train_data.csv')
dddcheck[required_columns] = dddcheck[required_columns].fillna(dddcheck[required_columns].mean())
X_pron = dddcheck[required_columns].to_numpy().astype(float)
y_pron = dddcheck['home_team_win'].to_numpy().astype(float)

# 讀取測試資料
test_df = pd.read_csv('same_season_test_data.csv')
test_df[required_columns] = test_df[required_columns].fillna(validation_df[required_columns].mean())
X_test = test_df[required_columns].to_numpy().astype(float)

# 調整數據形狀以適應 RNN 模型
X = X.reshape((X.shape[0], X.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_pron = X_pron.reshape((X_pron.shape[0], X_pron.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 定義模型訓練函數
def build_and_train_model(X, Y, X_val, y_true, X_test, output_filename='result.csv'):
    model = Sequential()
    
    # LSTM 第一層
    model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=regularizers.l2(0.1)))
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

    # 訓練模型
    model.fit(X, Y, validation_data=(X_val, y_true), epochs=10, batch_size=32, callbacks=[early_stopping])
    
    # 預測驗證集
    predictions = model.predict(X_val)
    predictions = np.where(predictions >= 0.5, 1, 0)
    
    # 計算驗證集的正確率
    accuracy = np.mean(predictions == y_true)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    predictions2 = model.predict(X_pron)
    predictions2 = np.where(predictions2 >= 0.5, 1, 0)
    
    # 計算驗證集的正確率
    accuracy = np.mean(predictions2 == y_pron)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    
    # 預測測試集
    predictions_test = model.predict(X_test)
    predictions_test = np.where(predictions_test >= 0.5, 1, 0)
    


    # 建立結果 DataFrame 並輸出為 CSV
    result_df = pd.DataFrame({
        'id': np.arange(len(predictions_test)),
        'home_team_win': predictions_test.flatten()
    })
    result_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

# 執行模型訓練和預測
build_and_train_model(X, Y, X_val, y_true, X_test, output_filename='result57_77.csv')

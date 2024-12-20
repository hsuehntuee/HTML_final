import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 讀取資料
train_df = pd.read_csv('2016.csv')

# 處理空值
required_columns = [
'away_pitching_SO_batters_faced_mean',
'away_pitching_earned_run_avg_mean',
'home_pitching_earned_run_avg_mean',
'away_pitcher_SO_batters_faced_mean',
'away_pitching_H_batters_faced_mean',
'home_pitching_earned_run_avg_std',
'home_batting_onbase_plus_slugging_mean',
'home_pitcher_wpa_def_mean',
'away_pitching_wpa_def_mean',
'home_pitching_H_batters_faced_mean',
'away_pitching_wpa_def_skew',
'home_pitcher_SO_batters_faced_mean',
'away_pitching_earned_run_avg_std',
'away_batting_wpa_bat_skew',
'away_pitching_SO_batters_faced_std',
'home_pitching_BB_batters_faced_mean',
'away_pitching_wpa_def_std',
'away_batting_RBI_std',
'away_pitching_H_batters_faced_std',
'home_batting_onbase_plus_slugging_std',
'home_pitching_leverage_index_avg_std',
'home_pitching_H_batters_faced_skew']
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].rolling(window=20).mean())

# 特徵縮放
scaler = StandardScaler()
X = scaler.fit_transform(train_df[required_columns].to_numpy().astype(float))[:1458]
Y = train_df['home_team_win'].to_numpy().astype(float)[:1458]

# 資料格式調整
X = X.reshape((X.shape[0], 1, X.shape[1]))  # 調整為 (samples, timesteps, features)

# 資料切分
X_train = X[:1300]
Y_train = Y[:1300]
X_val = X[1301:1458]
Y_val = Y[1301:1458]

# 構建 RNN 模型
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 編譯模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

# 驗證準確率
Y_val_pred = (model.predict(X_val) > 0.5).astype(int)
accuracy = accuracy_score(Y_val, Y_val_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 測試資料
test_df = pd.read_csv('same_season_test_data.csv')
X_test = scaler.transform(test_df[required_columns].fillna(train_df[required_columns].median()).to_numpy().astype(float))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 測試資料預測
predictions = (model.predict(X_test) > 0.5).astype(int)

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(predictions)),
    'home_team_win': predictions.flatten()
})
result_df.to_csv('result_rnn.csv', index=False)

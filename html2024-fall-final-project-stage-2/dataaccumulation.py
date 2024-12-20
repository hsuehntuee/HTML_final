import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 讀取資料
train_df = pd.read_csv('train_dataall.csv')

# 設定需要的特徵欄位
required_columns = [
    'away_pitching_SO_batters_faced_mean',
    'home_team_spread_mean',
    'away_batting_wpa_bat_std',
    'away_batting_onbase_perc_mean',
    'away_batting_RBI_std',
    'home_pitching_H_batters_faced_10RA',
    'away_pitching_BB_batters_faced_mean',
    'home_batting_onbase_plus_slugging_mean',
    'home_pitching_wpa_def_mean',
    'away_pitching_SO_batters_faced_10RA',
    'home_team_wins_mean',
    'away_team_spread_std',
    'away_batting_onbase_perc_std',
    'away_batting_batting_avg_mean',
    'home_batting_wpa_bat_std',
    'home_batting_batting_avg_mean',
    'home_pitching_wpa_def_std',
    'away_batting_RBI_mean',
    'home_batting_wpa_bat_mean',
    'away_pitching_H_batters_faced_std',
    'away_team_wins_std',
    'home_pitching_H_batters_faced_std',
    'home_pitching_leverage_index_avg_mean',
    'home_pitching_SO_batters_faced_std',
    'home_pitching_SO_batters_faced_mean'
]

# 處理空值
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
y = train_df['home_team_win'].to_numpy().astype(int)

# 訓練集和驗證集切分 (後2000筆為驗證集)
X_train = X
y_train = y
X_val = X[-2000:]
y_val = y[-2000:]

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 訓練 SVM 模型
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_model.fit(X_train_scaled, y_train)

# 預測
y_pred_val = svm_model.predict(X_val_scaled)

# 計算驗證準確率
validation_accuracy = accuracy_score(y_val, y_pred_val)

print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# 測試資料（同賽季資料）
same_season_df = pd.read_csv('2024_test_data.csv')
same_season_df[required_columns] = same_season_df[required_columns].fillna(same_season_df[required_columns].median())

X_test = same_season_df[required_columns].to_numpy().astype(float)
X_test_scaled = scaler.transform(X_test)

# 預測
y_test_pred = svm_model.predict(X_test_scaled)

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(y_test_pred)),
    'home_team_win': y_test_pred
})

result_df.to_csv('haha.csv', index=False)

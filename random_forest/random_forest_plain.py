import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 去除訓練資料中包含空值的行
required_columns = [
   
    'home_batting_onbase_perc_mean',
    'home_batting_onbase_plus_slugging_mean',
    'home_batting_RBI_mean',
    'away_batting_onbase_perc_mean',
    'away_batting_onbase_plus_slugging_mean',
    'away_batting_RBI_mean',
    'home_pitching_earned_run_avg_mean',
    'home_pitching_H_batters_faced_mean',
    'home_pitching_BB_batters_faced_mean',
    'away_pitching_earned_run_avg_mean',
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_H_batters_faced_mean',
    
    'home_pitcher_earned_run_avg_mean',
    'home_pitcher_SO_batters_faced_mean',
    'away_pitcher_earned_run_avg_mean',
    'away_pitcher_SO_batters_faced_mean',
    'home_team_errors_mean',
    'away_team_errors_mean',
    'home_pitching_wpa_def_skew',
    'away_pitching_wpa_def_skew'
    

]

train_df = pd.read_csv('balanced_train_data.csv')
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 創建隨機森林分類模型並進行訓練
model = RandomForestClassifier(n_estimators=12000, random_state=42)  # 可調整 n_estimators 和其他參數
model.fit(X_train, Y_train)

# 預測結果
predictions = model.predict(X_val)

# 計算正確率
accuracy = accuracy_score(Y_val, predictions)

# 顯示驗證集的正確率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 計算訓練集預測
train_predictions = model.predict(X_train)
Ein = np.mean(train_predictions != Y_train)  # 計算錯誤率

# 顯示 Ein
print(f"In-sample Error (Ein): {Ein * 100:.2f}%")
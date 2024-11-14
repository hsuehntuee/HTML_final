import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 讀取訓練資料
train_df = pd.read_csv('train_data.csv')

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

# 填補缺失值
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型

# 創建隨機森林分類模型並進行訓練
model = RandomForestClassifier(n_estimators=500, random_state=42)  # 可調整 n_estimators 和其他參數
model.fit(X, Y)

# 讀取驗證資料並填補空值
validation_df = pd.read_csv('validation.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
y_true = validation_df['home_team_win'].to_numpy()  # 驗證集的真實標籤
y_true = y_true[:2998]
X_val = X_val[:2998]

# 預測結果
predictions = model.predict(X_val)

# 計算正確率
accuracy = accuracy_score(y_true, predictions)

# 顯示驗證集的正確率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 計算訓練集預測
train_predictions = model.predict(X)
Ein = np.mean(train_predictions != Y)  # 計算錯誤率

# 顯示 Ein
print(f"In-sample Error (Ein): {Ein * 100:.2f}%")

# 讀取測試資料並進行預測
newo = pd.read_csv('same_season_test_data.csv')
X_test = newo[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
predictions2 = model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),
    'home_team_win': predictions2
})

# 輸出為 CSV
result_df.to_csv('result57_77.csv', index=False)

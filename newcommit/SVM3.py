import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 讀取訓練資料
train_df = pd.read_csv('train_data (2).csv')

# 去除訓練資料中包含空值的行
required_columns = [
    'season',
    #'home_batting_onbase_perc_skew', 
    #'home_batting_onbase_plus_slugging_mean',
    #'home_batting_onbase_plus_slugging_std', 
    'away_batting_wpa_bat_skew', 
    #'away_batting_RBI_mean', 
    #'away_batting_RBI_std',
    'away_batting_RBI_skew', 
    #'home_pitching_earned_run_avg_mean', 
    #'home_pitching_earned_run_avg_std', 
    #'home_pitching_earned_run_avg_skew', 
    #'home_pitching_H_batters_faced_mean', 
    'home_pitching_H_batters_faced_std', 
    #'home_pitching_H_batters_faced_skew', 
    'home_pitching_BB_batters_faced_mean',
    'home_pitching_BB_batters_faced_std',
    'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 
    'home_pitching_leverage_index_avg_std', 
    'home_pitching_wpa_def_skew', 
    #'away_pitching_earned_run_avg_mean', 
    'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_std', 
    #'away_pitching_SO_batters_faced_skew',
    'away_pitching_H_batters_faced_mean',
    'away_pitching_H_batters_faced_std', 
    'away_pitching_leverage_index_avg_skew', 
    'away_pitching_wpa_def_mean', 
    'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 
    'home_pitcher_earned_run_avg_mean',
    'home_pitcher_earned_run_avg_std', 
    #'home_pitcher_earned_run_avg_skew', 
    'home_pitcher_SO_batters_faced_mean',
    #'home_pitcher_SO_batters_faced_std', 
    #'home_pitcher_SO_batters_faced_skew',
    'home_pitcher_H_batters_faced_mean',
    #'home_pitcher_H_batters_faced_std', 
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
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())
poly = PolynomialFeatures(degree=1)  # 調整 degree 值以控制多項式的次數


# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)


# 使用 SVM 訓練
model = SVC(kernel='poly')  # 或使用 kernel='linear' 根據需要
model.fit(X_scaled, Y)

# 讀取驗證資料並填補空值
validation_df = pd.read_csv('validation.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
y_true = validation_df['home_team_win'].to_numpy()  # 驗證集的真實標籤
X_val = poly.fit_transform(X_val)
X_val = scaler.fit_transform(X_val)
# 預測結果
predictions = model.predict(X_val)

# 計算正確率
accuracy = accuracy_score(y_true, predictions)

# 顯示驗證集的正確率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 讀取測試資料
test_df = pd.read_csv('same_season_test_data.csv')
X_test = test_df[required_columns].fillna(test_df[required_columns].mean()).to_numpy().astype(float)
X_test = poly.fit_transform(X_test)
# 預測測試集
X_test =  scaler.fit_transform(X_test)
predictions_test = model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions_test)),  # id 從 0 開始
    'home_team_win': predictions_test  # 預測結果
})

# 輸出為 CSV
result_df.to_csv('resultSVM.csv', index=False)

import pandas as pd
import numpy as np

# 定義線性回歸類別
class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        # 計算權重 w
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏差項
        self.w = np.linalg.pinv(X_b)@(y)  # 使用伪逆計算權重

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏差項
        return X_b.dot(self.w)

# 讀取訓練資料
train_df = pd.read_csv('train_data.csv')

# 去除訓練資料中包含空值的行
required_columns = [
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
    'home_batting_onbase_perc_skew', 'home_batting_onbase_plus_slugging_mean', 'home_batting_onbase_plus_slugging_std', 
    #'home_batting_onbase_plus_slugging_skew', 'home_batting_leverage_index_avg_mean', 
    #'home_batting_leverage_index_avg_std', 'home_batting_leverage_index_avg_skew', 'home_batting_wpa_bat_mean', 
    #'home_batting_wpa_bat_std', 'home_batting_wpa_bat_skew', 'home_batting_RBI_mean', 'home_batting_RBI_std', 
    #'home_batting_RBI_skew', 'away_batting_batting_avg_mean', 'away_batting_batting_avg_std', 
    #'away_batting_batting_avg_skew', 'away_batting_onbase_perc_mean', 'away_batting_onbase_perc_std', 
    #'away_batting_onbase_perc_skew', 'away_batting_onbase_plus_slugging_mean', 'away_batting_onbase_plus_slugging_std', 
    #'away_batting_onbase_plus_slugging_skew', 'away_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_std',

    #'away_batting_leverage_index_avg_skew', 'away_batting_wpa_bat_mean', 'away_batting_wpa_bat_std', 
    'away_batting_wpa_bat_skew', 'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew', 
    'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std', 'home_pitching_earned_run_avg_skew', 
    #'home_pitching_SO_batters_faced_mean', 'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew', 
    'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std', 'home_pitching_H_batters_faced_skew', 
    'home_pitching_BB_batters_faced_mean', 'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std', 
    #'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean', 'home_pitching_wpa_def_std', 
    'home_pitching_wpa_def_skew', 'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean', 'away_pitching_SO_batters_faced_std', 
    'away_pitching_SO_batters_faced_skew', 'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std', 
    #'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean', 'away_pitching_BB_batters_faced_std', 
    #'away_pitching_BB_batters_faced_skew', 'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std', 
    'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean', 'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std', 
    'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean', 'home_pitcher_SO_batters_faced_std', 
    'home_pitcher_SO_batters_faced_skew', 'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std', 
    #'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean', 'home_pitcher_BB_batters_faced_std', 
    'home_pitcher_BB_batters_faced_skew', 'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std', 
    'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean', 'home_pitcher_wpa_def_std', 
    'home_pitcher_wpa_def_skew', 'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std', 
    'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean', 'away_pitcher_SO_batters_faced_std', 
    #'away_pitcher_SO_batters_faced_skew', 'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean', 'away_pitcher_BB_batters_faced_std', 
    #'away_pitcher_BB_batters_faced_skew', 'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std', 
    #'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean', 'away_pitcher_wpa_def_std', 
    #'away_pitcher_wpa_def_skew'
]
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
#X = np.hstack((X, X**2, X**3, X**4, X**5)) # 添加多項式特徵
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型
#print(Y)
# 創建線性回歸模型並進行訓練
print(len(Y))
model = LinearRegression()
model.fit(X, Y)

# 讀取驗證資料並填補空值
validation_df = pd.read_csv('validation.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
#X_val = np.hstack((X_val, X_val ** 2, X_val ** 3, X_val**4, X_val**5))  # 添加多項式特徵
y_true = validation_df['home_team_win'].to_numpy()  # 驗證集的真實標籤

y_true = y_true[:2998]
X_val = X_val[:2998]
# 預測結果
predictions = model.predict(X_val)
predictions = np.where(predictions >= 0.5, 1, 0)  # 將預測值轉換為 0 或 1

# 計算正確率
accuracy = np.mean(predictions == y_true)

# 顯示 predictions 和 y_true 的長度
#predictions = predictions[:2998]
#y_true = y_true[:2998]
print(f"Length of predictions: {len(predictions)}")
print(f"Shape of y_true: {y_true.shape}")
print(np.mean(y_true))
# 顯示驗證集的正確率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 計算訓練集預測
train_predictions = model.predict(X)
train_predictions = np.where(train_predictions >= 0.5, 1, 0)  # 將預測值轉換為 0 或 1

# 計算訓練集錯誤率 (Ein)
print(np.mean(train_predictions))
Ein = np.mean(train_predictions != Y)  # 計算錯誤率

# 顯示 Ein
print(f"In-sample Error (Ein): {Ein * 100:.2f}%")
# 讀取驗證資料並填補空值


newo = pd.read_csv('same_season_test_data.csv')
X_test =newo[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
predictions2 = model.predict(X_test)
predictions2 = np.where(predictions2 >= 0.5, 1, 0)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),  # id 從 0 到 6184
    'home_team_win': predictions2  # 預測結果
})

# 輸出為 CSV
result_df.to_csv('result57_77.csv', index=False)

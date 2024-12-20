import pandas as pd
import numpy as np
from itertools import combinations

# 定義線性回歸類別
class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏差項
        self.w = np.linalg.pinv(X_b) @ y  # 使用伪逆計算權重

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 添加偏差項
        return X_b.dot(self.w)

# 讀取資料
train_df = pd.read_csv('train_data.csv')
validation_df = pd.read_csv('validation.csv')

# 定義所有候選特徵
all_columns = [
    
    'home_batting_onbase_perc_skew', 'home_batting_onbase_plus_slugging_mean', 'home_batting_onbase_plus_slugging_std',
    'away_batting_wpa_bat_skew', 'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew', 
    'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std', 'home_pitching_earned_run_avg_skew',
    'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std', 'home_pitching_H_batters_faced_skew',
    'home_pitching_BB_batters_faced_mean', 'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew',
    'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std',
    'home_pitching_wpa_def_skew', 'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std',
    'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean', 'away_pitching_SO_batters_faced_std',
    'away_pitching_SO_batters_faced_skew', 'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std',
    'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean', 'away_pitching_wpa_def_std',
    'away_pitching_wpa_def_skew', 'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std',
    'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean', 'home_pitcher_SO_batters_faced_std',
    'home_pitcher_SO_batters_faced_skew', 'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std',
    'home_pitcher_BB_batters_faced_skew', 'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std',
    'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean', 'home_pitcher_wpa_def_std',
    'home_pitcher_wpa_def_skew', 'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std',
    'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean', 'away_pitcher_SO_batters_faced_std',
    'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean', 'away_pitcher_BB_batters_faced_std'
]

# 預先填補空值
train_df[all_columns] = train_df[all_columns].fillna(train_df[all_columns].mean())
validation_df[all_columns] = validation_df[all_columns].fillna(validation_df[all_columns].mean())

# 提取目標變量
Y_train = train_df['home_team_win'].to_numpy().astype(float)
y_true = validation_df['home_team_win'].to_numpy().astype(float)

# 設置變量來追踪最佳的準確率和對應的特徵組合
best_accuracy = 0
best_columns = []

# 對所有可能的特徵組合進行測試，這裡每次測試移除 N 個特徵
for r in range(1, len(all_columns) + 1):
    for subset in combinations(all_columns, r):
        selected_columns = [col for col in all_columns if col not in subset]

        # 提取訓練和驗證特徵
        X_train = train_df[selected_columns].to_numpy().astype(float)
        X_val = validation_df[selected_columns].to_numpy().astype(float)

        # 建立線性回歸模型
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # 在驗證集上進行預測
        predictions = model.predict(X_val)
        predictions = np.where(predictions >= 0.5, 1, 0)

        # 計算當前特徵組合的準確率
        accuracy = np.mean(predictions == y_true)

        # 如果新的準確率高於最佳準確率，則更新最佳準確率和對應特徵組合
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_columns = selected_columns
            print(f"New Best Accuracy: {best_accuracy * 100:.2f}% with {len(best_columns)} features")

print(f"Optimal Validation Accuracy: {best_accuracy * 100:.2f}%")
print("Optimal Feature Combination:", best_columns)

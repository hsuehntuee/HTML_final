import pandas as pd
import numpy as np

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

# 讀取訓練資料
train_df = pd.read_csv('train_data.csv')

# 清理訓練資料
required_columns = ['home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 'home_batting_onbase_plus_slugging_10RA', 
    'home_batting_leverage_index_avg_10RA', 'home_batting_RBI_10RA', 'away_batting_batting_avg_10RA', 
    'away_batting_onbase_perc_10RA', 'away_batting_onbase_plus_slugging_10RA', 'away_batting_leverage_index_avg_10RA', 
    'away_batting_RBI_10RA', 'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA', 
    'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA', 'away_pitching_earned_run_avg_10RA', 
    'away_pitching_SO_batters_faced_10RA', 'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA', 
    'home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 
    'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 
    'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA', 'home_team_errors_mean', 'home_team_errors_std', 'home_team_errors_skew', 'away_team_errors_mean', 
    'away_team_errors_std', 'away_team_errors_skew', 'home_team_spread_mean', 'home_team_spread_std', 
    'home_team_spread_skew', 'away_team_spread_mean', 'away_team_spread_std', 'away_team_spread_skew', 
    'home_team_wins_mean', 'home_team_wins_std', 'home_team_wins_skew', 'away_team_wins_mean', 'away_team_wins_std', 
    'away_team_wins_skew', 'home_batting_batting_avg_mean', 'home_batting_batting_avg_std', 
    'home_batting_batting_avg_skew', 'home_batting_onbase_perc_mean', 'home_batting_onbase_perc_std', 
    'home_batting_onbase_perc_skew', 'home_batting_onbase_plus_slugging_mean', 'home_batting_onbase_plus_slugging_std', 
    'home_batting_onbase_plus_slugging_skew', 'home_batting_leverage_index_avg_mean', 
    'home_batting_leverage_index_avg_std', 'home_batting_leverage_index_avg_skew', 'home_batting_wpa_bat_mean', 
]  # 這是您之前定義的需要的列
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 提取主場球隊資訊
home_teams = ['JBM', 'UPV', 'QPO', 'ZQF', 'FBW', 'XFB', 'MZG', 'ECN', 'SAJ', 'DPS', 'VQC', 'KFH', 'RAV', 'GLO', 'PJT', 'PDF', 'STC']
models = {}

# 根據主場球隊訓練模型
for home_team in home_teams:
    # 根據主場球隊過濾資料
    home_games = train_df[train_df['home_team_abbr'] == home_team]
    
    if home_games.empty:
        continue  # 如果該主場球隊沒有比賽資料，則跳過

    # 提取特徵和目標變量
    X = home_games[required_columns].to_numpy().astype(float)
    X = np.hstack((X, X**2, X**3, X**4, X**5))  # 添加多項式特徵
    Y = home_games['home_team_win'].to_numpy().astype(float)

    # 訓練模型
    model = LinearRegression()
    model.fit(X, Y)
    models[home_team] = model  # 儲存模型

# 讀取驗證資料並填補空值
validation_df = pd.read_csv('validation.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
X_val = np.hstack((X_val, X_val ** 2, X_val ** 3, X_val ** 4, X_val ** 5))  # 添加多項式特徵
y_true = validation_df['home_team_win'].to_numpy()  # 驗證集的真實標籤

# 對每一行驗證資料進行預測
predictions = []
for index, row in validation_df.iterrows():
    home_team = row['home_team_abbr']  # 獲取當前行的主場球隊
    if home_team in models:
        model = models[home_team]  # 根據主場球隊選擇模型
        pred = model.predict(X_val[index:index+1])  # 預測當前行
        predictions.append(1 if pred >= 0.5 else 0)  # 將預測值轉換為 0 或 1
    else:
        predictions.append(0)  # 如果沒有模型，預測為 0

# 計算驗證集的正確率
accuracy = np.mean(predictions == y_true[:len(predictions)])
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 計算訓練集預測和錯誤率
for home_team, model in models.items():
    home_games = train_df[train_df['home_team_abbr'] == home_team]
    if home_games.empty:
        continue
    
    X_train = home_games[required_columns].to_numpy().astype(float)
    X_train = np.hstack((X_train, X_train ** 2, X_train ** 3, X_train ** 4, X_train ** 5))  # 添加多項式特徵
    Y_train = home_games['home_team_win'].to_numpy().astype(float)

    train_predictions = model.predict(X_train)
    train_predictions = np.where(train_predictions >= 0.5, 1, 0)  # 將預測值轉換為 0 或 1
    Ein = np.mean(train_predictions != Y_train)  # 計算錯誤率
    print(f"In-sample Error (Ein) for {home_team}: {Ein * 100:.2f}%")

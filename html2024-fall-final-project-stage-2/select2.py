import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# 讀取資料
train_df = pd.read_csv('train_dataall.csv')

# 設定需要的特徵欄位
required_columns =  [
    #'is_night_game',
    #'home_team_win',
    #'home_pitcher',
    #'away_pitcher',
    #'home_team_rest',
    #'away_team_rest',
    'season',
    'home_batting_batting_avg_10RA',
    'home_batting_onbase_perc_10RA',
    'home_batting_onbase_plus_slugging_10RA',
    'home_batting_leverage_index_avg_10RA',
    'home_batting_RBI_10RA',
    'away_batting_batting_avg_10RA',
    'away_batting_onbase_perc_10RA',
    'away_batting_onbase_plus_slugging_10RA',
    #'away_batting_leverage_index_avg_10RA',
    'away_batting_RBI_10RA',
    'home_pitching_earned_run_avg_10RA',
    'home_pitching_SO_batters_faced_10RA',
    'home_pitching_H_batters_faced_10RA',
    'home_pitching_BB_batters_faced_10RA',
    'away_pitching_earned_run_avg_10RA',
    'away_pitching_SO_batters_faced_10RA',
    'away_pitching_H_batters_faced_10RA',
    'away_pitching_BB_batters_faced_10RA',
    #'home_team_season',
    #'away_team_season',
    'home_team_errors_mean',
    'home_team_errors_std',
    'away_team_errors_mean',
    'away_team_errors_std',
    'home_team_spread_mean',
    'home_team_spread_std',
    #'away_team_spread_mean',
    'away_team_spread_std',
    'home_team_wins_mean',
    'home_team_wins_std',
    'away_team_wins_mean',
    'away_team_wins_std',
    'home_batting_batting_avg_mean',
    'home_batting_batting_avg_std',
    'home_batting_onbase_perc_mean',
    'home_batting_onbase_perc_std',
    'home_batting_onbase_plus_slugging_mean',
    'home_batting_onbase_plus_slugging_std',
    'home_batting_leverage_index_avg_mean',
    'home_batting_leverage_index_avg_std',
    'home_batting_wpa_bat_mean',
    'home_batting_wpa_bat_std',
    'home_batting_RBI_mean',
    'home_batting_RBI_std',
    'away_batting_batting_avg_mean',
    'away_batting_batting_avg_std',
    'away_batting_onbase_perc_mean',
    'away_batting_onbase_perc_std',
    'away_batting_onbase_plus_slugging_mean',
    'away_batting_onbase_plus_slugging_std',
    #'away_batting_leverage_index_avg_mean',
    'away_batting_leverage_index_avg_std',
    'away_batting_wpa_bat_mean',
    'away_batting_wpa_bat_std',
    'away_batting_RBI_mean',
    'away_batting_RBI_std',
    #'home_pitching_earned_run_avg_mean',
    'home_pitching_earned_run_avg_std',
    'home_pitching_SO_batters_faced_mean',
    'home_pitching_SO_batters_faced_std',
    'home_pitching_H_batters_faced_mean',
    'home_pitching_H_batters_faced_std',
    'home_pitching_BB_batters_faced_mean',
    'home_pitching_BB_batters_faced_std',
    'home_pitching_leverage_index_avg_mean',
    'home_pitching_leverage_index_avg_std',
    'home_pitching_wpa_def_mean',
    'home_pitching_wpa_def_std',
    'away_pitching_earned_run_avg_mean',
    'away_pitching_earned_run_avg_std',
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_SO_batters_faced_std',
    'away_pitching_H_batters_faced_mean',
    'away_pitching_H_batters_faced_std',
    'away_pitching_BB_batters_faced_mean',
    'away_pitching_BB_batters_faced_std',
    #'away_pitching_leverage_index_avg_mean',
    'away_pitching_leverage_index_avg_std',
    'away_pitching_wpa_def_mean'
]

# 處理空值
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
y = train_df['home_team_win'].to_numpy().astype(int)

# 標準化特徵
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 訓練決策樹模型
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)  # 設定最大深度避免過度擬合
dt_model.fit(X_scaled, y)

# 查看模型的特徵重要性
importance = dt_model.feature_importances_

# 將特徵和重要性配對並按重要性排序
feature_importance = pd.DataFrame({
    'Feature': required_columns,
    'Importance': importance
})

# 按照重要性排序
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)


top_25_features = feature_importance.head(25)
for i in range(25):
    print(f"'{top_25_features.iloc[i]['Feature']}',")



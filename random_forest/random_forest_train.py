import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



# 去除訓練資料中包含空值的行
required_columns = [
       # 'season',
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
    #'away_batting_RBI_mean', 
  'away_batting_RBI_std',
     # 'away_batting_RBI_skew', 
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
    #  'away_pitcher_earned_run_avg_mean', 
      'away_pitcher_earned_run_avg_std', 
      'away_pitcher_earned_run_avg_skew', 
    'away_pitcher_SO_batters_faced_mean', 
     'away_pitcher_SO_batters_faced_std', 
    #'away_pitcher_SO_batters_faced_skew', 'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 
    #'away_pitcher_BB_batters_faced_mean',
    # 'away_pitcher_BB_batters_faced_std', 
    #'away_pitcher_BB_batters_faced_skew', 'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std', 
    #'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean', 'away_pitcher_wpa_def_std', 
    #'away_pitcher_wpa_def_skew', 
    'date_standardized'
]

# 讀取訓練資料
train_df = pd.read_csv('balanced_train_data.csv')

# 填補缺失值
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# 抽取特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 創建隨機森林和 XGBoost 模型
rf_model = RandomForestClassifier(n_estimators=12000, random_state=42)
xgb_model = XGBClassifier(n_estimators=12000, random_state=42)

# 創建投票分類器，將兩個模型組合在一起
voting_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], voting='soft')

# 訓練投票分類器
voting_model.fit(X_train, Y_train)


# 預測結果
predictions = voting_model.predict(X_val)

# 計算正確率
accuracy = accuracy_score(Y_val, predictions)

# 顯示驗證集的正確率
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 計算訓練集預測
train_predictions = voting_model.predict(X_train)
Ein = np.mean(train_predictions != Y_train)  # 計算錯誤率

# 顯示 Ein
print(f"In-sample Error (Ein): {Ein * 100:.2f}%")


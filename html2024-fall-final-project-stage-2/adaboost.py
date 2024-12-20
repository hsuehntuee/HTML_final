import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 讀取訓練資料
train_df = pd.read_csv('train_dataallwash.csv')

# 處理空值
required_columns = [
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
    'home_pitching_H_batters_faced_skew',
]
required_columns = ['home_team_spread_mean',
 'away_pitching_SO_batters_faced_10RA',
 'away_pitching_SO_batters_faced_mean',
 'home_team_wins_mean',
 'away_team_wins_mean',
 'away_pitching_earned_run_avg_mean',
 'away_pitching_earned_run_avg_10RA',
 'home_pitching_SO_batters_faced_mean',
 'home_pitching_earned_run_avg_std',
 'home_pitching_wpa_def_mean',
 'away_pitching_H_batters_faced_mean',
 'home_pitching_H_batters_faced_mean',
 'home_batting_onbase_plus_slugging_mean']

train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 多項式擴展與標準化
poly = PolynomialFeatures(degree=1)
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 設定 AdaBoost 模型
base_estimator = DecisionTreeClassifier(max_depth=2)  # Decision Stump
adaboost_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=700, learning_rate=0.01)

# 5-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"Training Fold {fold_idx + 1}...")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 訓練模型
    adaboost_model.fit(X_train, Y_train)

    # 驗證模型
    predictions = adaboost_model.predict(X_val)
    accuracy = accuracy_score(Y_val, predictions)
    fold_accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Accuracy: {accuracy * 100:.2f}%")


# 平均準確率
avg_accuracy = np.mean(fold_accuracies)
print(f"Average 5-Fold Validation Accuracy: {avg_accuracy * 100:.2f}%")

# 測試資料預測
test_df = pd.read_csv('2024_test_data.csv')
X_test = test_df[required_columns].fillna(train_df[required_columns].median()).to_numpy().astype(float)
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)
test_predictions = adaboost_model.predict(X_test_scaled)

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(test_predictions)),
    'home_team_win': test_predictions
})
result_df.to_csv('result_adaboost2.csv', index=False)

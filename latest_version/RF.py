

# 取得輸入所需欄位
def get_required_columns():
    # 根據需要設定 required_columns
    required_columns = [
    'is_night_game', 'home_team_rest', 'away_team_rest', 'home_pitcher_rest', 'away_pitcher_rest',
    'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 'home_batting_onbase_plus_slugging_10RA', 
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
    'home_batting_wpa_bat_std', 'home_batting_wpa_bat_skew', 'home_batting_RBI_mean', 'home_batting_RBI_std', 
    'home_batting_RBI_skew', 'away_batting_batting_avg_mean', 'away_batting_batting_avg_std', 
    'away_batting_batting_avg_skew', 'away_batting_onbase_perc_mean', 'away_batting_onbase_perc_std', 
    'away_batting_onbase_perc_skew', 'away_batting_onbase_plus_slugging_mean', 'away_batting_onbase_plus_slugging_std', 
    'away_batting_onbase_plus_slugging_skew', 'away_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_std',
    'away_batting_leverage_index_avg_skew', 'away_batting_wpa_bat_mean', 'away_batting_wpa_bat_std', 
    'away_batting_wpa_bat_skew', 'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew', 
    'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std', 'home_pitching_earned_run_avg_skew', 
    'home_pitching_SO_batters_faced_mean', 'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew', 
    'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std', 'home_pitching_H_batters_faced_skew', 
    'home_pitching_BB_batters_faced_mean', 'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std', 
    'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean', 'home_pitching_wpa_def_std', 
    'home_pitching_wpa_def_skew', 'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean', 'away_pitching_SO_batters_faced_std', 
    'away_pitching_SO_batters_faced_skew', 'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std', 
    'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean', 'away_pitching_BB_batters_faced_std', 
    'away_pitching_BB_batters_faced_skew', 'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std', 
    'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean', 'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 
    'date_standardized'
    ]
    return required_columns



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
import time

start = time.time()

# 載入資料
train_df = pd.read_csv('stage_train.csv')
val_df = pd.read_csv('stage_validation.csv')
test_df = pd.read_csv('stage1_test.csv')
test2_df = pd.read_csv('stage2_test.csv')
answer_df = pd.read_csv('stage1_label.csv')  # 載入預測解答
answer2_df = pd.read_csv('stage2_label.csv')
required_columns = get_required_columns()

# 填補缺失值與裁剪數據範圍
def preprocess_data(df, required_columns, lower_bounds=None, upper_bounds=None):
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
    if lower_bounds is not None and upper_bounds is not None:
        df[required_columns] = df[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    return df

# 選擇是否要進行數據裁剪
clip_choice = input("Do you want to clip data? (yes/no): ").strip().lower()
lower_bounds = None
upper_bounds = None
if clip_choice == 'yes':
    quantiles = train_df[required_columns].quantile([0.05, 0.95])
    lower_bounds = quantiles.loc[0.05]
    upper_bounds = quantiles.loc[0.95]

train_df = preprocess_data(train_df, required_columns, lower_bounds, upper_bounds)

# 特徵與標籤處理
X_train = train_df[required_columns].to_numpy().astype(float)
Y_train = train_df['home_team_win'].to_numpy().astype(int)
X_val = val_df[required_columns].to_numpy().astype(float)
Y_val = val_df['home_team_win'].to_numpy().astype(int)
X_test = test_df[required_columns].to_numpy().astype(float)
Y_test = answer_df['home_team_win'].to_numpy().astype(int)
X_test2 = test2_df[required_columns].to_numpy().astype(float)
Y_test2 = answer2_df['home_team_win'].to_numpy().astype(int)

# Scaling 選擇
scaler = None
scaling_choice = input("Choose scaling method (robust/standard/none): ").strip().lower()
if scaling_choice == 'robust':
    scaler = RobustScaler()
elif scaling_choice == 'standard':
    scaler = StandardScaler()
else: 
    print("Warning") 
if scaler:
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_test2 = scaler.transform(X_test2)

# Feature Selection 選擇
feature_selection_choice = input("Do you want to perform feature selection? (yes/no): ").strip().lower()

if feature_selection_choice == 'yes':
    # 訓練 RandomForestClassifier 來選擇重要的特徵
    rf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=10, n_jobs=-1)
    rf.fit(X_train, Y_train)

    # 取得每個特徵的重要性
    feature_importances = rf.feature_importances_

    # 根據特徵的重要性排序，選擇前 1/5 的特徵
    important_features = np.argsort(feature_importances)[-int(len(required_columns) / 5):]
    required_columns = [required_columns[i] for i in important_features]

    # 根據選出的特徵進行數據處理
    X_train = train_df[required_columns].to_numpy().astype(float)
    X_val = val_df[required_columns].to_numpy().astype(float)
    X_test = test_df[required_columns].to_numpy().astype(float)
    X_test2 = test2_df[required_columns].to_numpy().astype(float)

# 模型結構 - 隨機森林
model = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=10, n_jobs=-1)

# 訓練模型
history = {
    'Ein': [],
    'Evalidation': [],
    'Eout_test1': [],
    'Eout_test2': []
}

start_time = time.time()

# 訓練 Random Forest 模型
model.fit(X_train, Y_train)

# 預測與計算準確率
train_predictions = model.predict(X_train)
ein = accuracy_score(Y_train, train_predictions)
print(ein)

val_predictions = model.predict(X_val)
evalidation = accuracy_score(Y_val, val_predictions)
print(evalidation)

test_predictions = model.predict(X_test)
eout_test1 = accuracy_score(Y_test, test_predictions)
print(eout_test1)

test2_predictions = model.predict(X_test2)
eout_test2 = accuracy_score(Y_test2, test2_predictions)
print(eout_test2)

print("Training time: ", time.time() - start_time)

# 測試模型
predictions_test1 = model.predict(X_test)
result_df_test1 = pd.DataFrame({'id': np.arange(len(predictions_test1)), 'home_team_win': predictions_test1})
result_df_test1.to_csv('result_team_RF_test1.csv', index=False)
print("Test1 predictions saved to 'result_team_RF_test1.csv'.")

predictions_test2 = model.predict(X_test2)
result_df_test2 = pd.DataFrame({'id': np.arange(len(predictions_test2)), 'home_team_win': predictions_test2})
result_df_test2.to_csv('result_team_RF_test2.csv', index=False)
print("Test2 predictions saved to 'result_team_RF_test2.csv'.")

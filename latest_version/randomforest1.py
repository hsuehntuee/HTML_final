import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
train_df = pd.read_csv('stage12_wash_dateFormatted_train.csv')


# Truncate required columns to a smaller sample for demonstration
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
]


no = ['home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA']
required_columns = [col for col in required_columns if col not in no]
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())
# One-Hot Encoding for 'home_team_abbr'
#home_team_dummies = pd.get_dummies(train_df['home_team_abbr'], prefix='team')
#train_df = pd.concat([train_df, home_team_dummies], axis=1)


# 抽取特徵和目標變量
#X = train_df[required_columns + list(home_team_dummies.columns)].to_numpy().astype(float)
newo = pd.read_csv('filled_kaggle_test.csv')
X_test = newo[required_columns].fillna(newo[required_columns].mean()).to_numpy().astype(float)
quantiles = train_df[required_columns].quantile([0.05, 0.95])

# 分別取出上下界
lower_bounds = quantiles.loc[0.05]
upper_bounds = quantiles.loc[0.95]

# 對訓練和測試數據裁剪
train_df[required_columns] = train_df[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
newo[required_columns] = newo[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
'''
scaler = RobustScaler()
X = scaler.fit_transform(X)  # 在訓練集上擬合
X_test = scaler.transform(X_test)  # 測試集使用相同參數
'''
'''
import seaborn as sns
import matplotlib.pyplot as plt

for col in required_columns:
    sns.kdeplot(train_df[col], label="Train", shade=True)
    sns.kdeplot(newo[col], label="Test", shade=True)
    plt.title(f"Feature: {col}")
    plt.legend()
    plt.show()
'''
# 創建隨機森林分類模型
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型
model = RandomForestClassifier(n_estimators=500, random_state=1)

# 進行 3-fold 交叉驗證
kf = KFold(n_splits=7, shuffle=True, random_state=1)

accuracies = []
in_sample_errors = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")

    # 切分訓練集和驗證集
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 訓練模型
    model.fit(X_train, Y_train)

    # 驗證集預測
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_predictions)
    accuracies.append(accuracy)
    print(f"  Validation Accuracy: {accuracy * 100:.2f}%")

    # 訓練集預測
    train_predictions = model.predict(X_train)
    Ein = np.mean(train_predictions != Y_train)
    in_sample_errors.append(Ein)
    print(f"  In-sample Error (Ein): {Ein * 100:.2f}%")
    
    break

# 顯示平均結果
print("\nOverall Results:")
print(f"  Average Validation Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"  Average In-sample Error (Ein): {np.mean(in_sample_errors) * 100:.2f}%")

# 使用整個訓練集重新訓練模型
model.fit(X, Y)


predictions2 = model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),
    'home_team_win': predictions2
})

# 輸出為 CSV
result_df.to_csv('result_3fold2.csv', index=False)
feature_importances = model.feature_importances_

# 將特徵名稱和對應的重要性組合成一個字典
features_with_importance = list(zip(required_columns , feature_importances))

# 根據重要性進行排序（由高到低）
sorted_features = sorted(features_with_importance, key=lambda x: x[1], reverse=True)

# 輸出排序後的特徵及其重要性
print("Feature Importance:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")
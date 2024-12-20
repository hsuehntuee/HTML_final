import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# 讀取訓練資料
train_df = pd.read_csv('train_dataall.csv')
#team_name = "VJV"  # 替換為你想要的隊伍名稱
#train_df = train_df[train_df['home_team_abbr'] == team_name]
#print(len(train_df))
# 去除訓練資料中包含空值的行
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
# 填補缺失值
train_df[required_columns] = train_df[required_columns].fillna(0)

# One-Hot Encoding for 'home_team_abbr'
home_team_dummies = pd.get_dummies(train_df['home_team_abbr'], prefix='team')
train_df = pd.concat([train_df, home_team_dummies], axis=1)

# 抽取特徵和目標變量
X = train_df[required_columns + list(home_team_dummies.columns)].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型

# 創建隨機森林分類模型
model = RandomForestClassifier(n_estimators=7000, random_state=42)

# 進行 3-fold 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
    
    #break

# 顯示平均結果
print("\nOverall Results:")
print(f"  Average Validation Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"  Average In-sample Error (Ein): {np.mean(in_sample_errors) * 100:.2f}%")

# 使用整個訓練集重新訓練模型
model.fit(X, Y)

# 讀取測試資料並進行預測
newo = pd.read_csv('2024_test_data.csv')
newo_home_team_dummies = pd.get_dummies(newo['home_team_abbr'], prefix='team')
newo = pd.concat([newo, newo_home_team_dummies], axis=1)

# 確保測試集的 Dummy Columns 與訓練集一致
for col in home_team_dummies.columns:
    if col not in newo:
        newo[col] = 0  # 添加缺失的 Dummy 列，填入 0

X_test = newo[required_columns + list(home_team_dummies.columns)].fillna(newo[required_columns].mean()).to_numpy().astype(float)
predictions2 = model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),
    'home_team_win': predictions2
})

# 輸出為 CSV
result_df.to_csv('result_3fold.csv', index=False)

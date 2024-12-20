import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# 讀取訓練資料
train_df = pd.read_csv('4_10_balanced_train_data.csv')

# 選擇某一隊的資料（例：隊伍 VJV）
#team_name = "VJV"  # 替換為你想要的隊伍名稱
#team_data = train_df[train_df['home_team_abbr'] == team_name]
team_data = train_df
# 確保選取的資料非空
#if team_data.empty:
#    raise ValueError(f"No data found for team '{team_name}'!")

# 去除空值的行
required_columns = [
    'is_night_game',
    'season',
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
team_data[required_columns] = team_data[required_columns].fillna(0)
home_team_dummies = pd.get_dummies(train_df['home_team_abbr'], prefix='team')
team_data = pd.concat([team_data, home_team_dummies], axis=1)

# 抽取特徵和目標變量
X = team_data[required_columns].to_numpy().astype(float)
Y = team_data['home_team_win'].to_numpy().astype(float)  # 確保 Y 為 float 類型

# 創建 SVM 模型 (Soft Margin)
svm_model = SVC(C=0.1, kernel='linear', random_state=50)  # 調整 C 與 kernel 根據需求

# 進行 3-fold 交叉驗證
kf = KFold(n_splits=5, shuffle=True, random_state=50)

accuracies = []
in_sample_errors = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}:")

    # 切分訓練集和驗證集
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 訓練 SVM 模型
    svm_model.fit(X_train, Y_train)

    # 驗證集預測
    val_predictions = svm_model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_predictions)
    accuracies.append(accuracy)
    print(f"  Validation Accuracy: {accuracy * 100:.2f}%")

    # 訓練集預測
    train_predictions = svm_model.predict(X_train)
    Ein = np.mean(train_predictions != Y_train)
    in_sample_errors.append(Ein)
    print(f"  In-sample Error (Ein): {Ein * 100:.2f}%")
    
    #break

# 顯示平均結果
print("\nOverall Results:")
print(f"  Average Validation Accuracy: {np.mean(accuracies) * 100:.2f}%")
print(f"  Average In-sample Error (Ein): {np.mean(in_sample_errors) * 100:.2f}%")

# 使用整個訓練集重新訓練模型
svm_model.fit(X, Y)

# 測試資料範例（僅測試同隊的數據）
newo = pd.read_csv('same_season_test_data.csv')
#newo_team_data = newo[newo['home_team_abbr'] == team_name]

# 確保測試集非空
#if newo_team_data.empty:
#    raise ValueError(f"No test data found for team '{team_name}'!")

X_test = newo[required_columns].fillna(newo[required_columns].mean()).to_numpy().astype(float)
predictions2 = svm_model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),
    'home_team_win': predictions2
})

# 輸出為 CSV
result_df.to_csv('result_team_only_svm.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 讀取資料
train_df = pd.read_csv('train_dataall.csv')

# 設定需要的特徵欄位
required_columns = [
    'away_pitching_SO_batters_faced_mean',
    'home_team_spread_mean',
    'away_batting_wpa_bat_std',
    'away_batting_onbase_perc_mean',
    'away_batting_RBI_std',
    'home_pitching_H_batters_faced_10RA',
    'away_pitching_BB_batters_faced_mean',
    'home_batting_onbase_plus_slugging_mean',
    'home_pitching_wpa_def_mean',
    'away_pitching_SO_batters_faced_10RA',
    'home_team_wins_mean',
    'away_team_spread_std',
    'away_batting_onbase_perc_std',
    'away_batting_batting_avg_mean',
    'home_batting_wpa_bat_std',
    'home_batting_batting_avg_mean',
    'home_pitching_wpa_def_std',
    'away_batting_RBI_mean',
    'home_batting_wpa_bat_mean',
    'away_pitching_H_batters_faced_std',
    'away_team_wins_std',
    'home_pitching_H_batters_faced_std',
    'home_pitching_leverage_index_avg_mean',
    'home_pitching_SO_batters_faced_std',
    'home_pitching_SO_batters_faced_mean'
]

# 處理空值
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 特徵和目標變量
X = train_df[required_columns].to_numpy().astype(float)
y = train_df['home_team_win'].to_numpy().astype(int)

# 分割訓練和驗證資料
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 定義 SVM 模型
svm_model = SVC(probability=True)

# 設定超參數範圍
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正則化參數
    'gamma': [0.01, 0.1, 1, 10],  # 核函數的係數
    'kernel': ['linear', 'rbf', 'poly']  # 核函數類型
}

# 使用 GridSearchCV 進行超參數調整
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 顯示最佳參數
print(f"Best parameters: {grid_search.best_params_}")

# 使用最佳超參數訓練模型
best_svm_model = grid_search.best_estimator_
best_svm_model.fit(X_train_scaled, y_train)

# 預測與準確率計算
y_pred_val = best_svm_model.predict(X_val_scaled)
validation_accuracy = accuracy_score(y_val, y_pred_val)

print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# 可以進一步將測試資料讀入並預測結果
test_df = pd.read_csv('same_season_test_data.csv')
X_test = test_df[required_columns].fillna(train_df[required_columns].median()).to_numpy().astype(float)
X_test_scaled = scaler.transform(X_test)

# 預測測試資料
y_pred_test = best_svm_model.predict(X_test_scaled)

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(y_pred_test)),
    'home_team_win': y_pred_test
})
result_df.to_csv('haha.csv', index=False)


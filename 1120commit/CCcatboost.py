from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
# 使用 2016 的數據
train_df = pd.read_csv('XXX.csv')

required_columns =  [
           'away_team_spread_mean',
'away_team_wins_mean',
    'away_pitching_SO_batters_faced_10RA',
                    'home_team_wins_skew',
      'home_pitching_earned_run_avg_mean',
                    'away_team_wins_skew',
         'away_batting_onbase_perc_mean',
     'away_pitcher_SO_batters_faced_10RA',
      'away_pitching_earned_run_avg_10RA',
       'home_pitching_earned_run_avg_std',
                    'home_team_wins_mean',
                  'away_batting_RBI_mean',
              'home_batting_wpa_bat_mean',
                  'home_team_spread_mean',
    'home_pitching_BB_batters_faced_10RA',
    
]
# 處理空值
train_df[required_columns] = train_df[required_columns]

# 分割資料
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=50, stratify=Y)

# 訓練 CatBoost 模型
catboost_model = CatBoostClassifier(
    iterations=5000,          # 增加迭代次數，提升模型學習能力
    learning_rate=0.005,       # 控制每次更新步伐，降低 overfit 風險
    depth=6,                  # 樹深度，適中以防過擬合
    l2_leaf_reg=5,            # L2 正則化
    loss_function='Logloss',  # 適用於分類任務
    eval_metric='Accuracy',   # 驗證集評估指標
    random_seed=42,
    verbose=200               # 每 200 次迭代輸出結果
)
catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# 驗證集表現
y_pred = catboost_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

newo = pd.read_csv('same_season_test_data.csv')
X_test =newo[required_columns]
predictions2 = catboost_model.predict(X_test)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),  # id 從 0 到 6184
    'home_team_win': predictions2  # 預測結果
})

# 輸出為 CSV
result_df.to_csv('rrrrrd.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 讀取訓練資料
train_df = pd.read_csv('2016.csv')

# 處理空值
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
'home_pitching_H_batters_faced_skew']
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 多項式擴展與標準化
poly = PolynomialFeatures(degree=1)
X = train_df[required_columns].to_numpy().astype(float)[:1300]
Y = train_df['home_team_win'].to_numpy().astype(float)[:1300]
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 訓練 Adaboost 模型
base_estimator = DecisionTreeClassifier(max_depth=1)  # Decision Stump
adaboost_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=2000, learning_rate=0.01)
adaboost_model.fit(X_scaled, Y)

# 驗證資料
validation_df = pd.read_csv('2016.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].median()).to_numpy().astype(float)[1300:]
y_true = validation_df['home_team_win'].to_numpy()[1300:]
X_val_poly = poly.transform(X_val)
X_val_scaled = scaler.transform(X_val_poly)

# 預測與準確率計算
predictions = adaboost_model.predict(X_val_scaled)
accuracy = accuracy_score(y_true, predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 錯誤分析
errors = validation_df[predictions != y_true].copy()
errors['predicted_label'] = predictions[predictions != y_true]
errors.to_csv('error.csv', index=False)  # 將錯誤結果保存至 CSV 檔案

# 測試資料
test_df = pd.read_csv('same_season_test_data.csv')
X_test = test_df[required_columns].fillna(train_df[required_columns].median()).to_numpy().astype(float)
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)

# 測試資料預測
predictions = adaboost_model.predict(X_test_scaled)

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(predictions)),
    'home_team_win': predictions
})
result_df.to_csv('result_adaboost.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# 讀取訓練資料
train_df = pd.read_csv('2222.csv')

# 自定義線性回歸分類器
class LinearRegressionClassifier(LinearRegression):
    def predict(self, X):
        predictions = super().predict(X)
        return (predictions > 0.5).astype(int)

# 處理空值
required_columns = [
    'is_night_game',
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
#        'away_pitching_wpa_def_skew', 
   'home_pitcher_earned_run_avg_mean',
#     'home_pitcher_earned_run_avg_std', 
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
    #'away_pitcher_wpa_def_skew'
]
# 去除訓練資料中包含空值的行
required_columns = [
   
    'home_batting_onbase_perc_mean',
    'home_batting_onbase_plus_slugging_mean',
    'home_batting_RBI_mean',
    'away_batting_onbase_perc_mean',
    'away_batting_onbase_plus_slugging_mean',
    'away_batting_RBI_mean',
    'home_pitching_earned_run_avg_mean',
    'home_pitching_H_batters_faced_mean',
    'home_pitching_BB_batters_faced_mean',
    'away_pitching_earned_run_avg_mean',
    'away_pitching_SO_batters_faced_mean',
    'away_pitching_H_batters_faced_mean',
    
    'home_pitcher_earned_run_avg_mean',
    'home_pitcher_SO_batters_faced_mean',
    'away_pitcher_earned_run_avg_mean',
    'away_pitcher_SO_batters_faced_mean',
    'home_team_errors_mean',
    'away_team_errors_mean',
    'home_pitching_wpa_def_skew',
    'away_pitching_wpa_def_skew'
    

]
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].median())

# 多項式擴展與標準化
poly = PolynomialFeatures(degree=1)
X = train_df[required_columns].to_numpy().astype(float)[:6840]
Y = train_df['home_team_win'].to_numpy().astype(float)[:6840]
print(Y)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 設定多個模型
models = [
    ('log_reg', LogisticRegression(max_iter=1800)),
    ('svm', SVC(kernel='poly', probability=True)),
    ('rf', RandomForestClassifier(n_estimators=120)),
    ('knn', KNeighborsClassifier(n_neighbors=6)),

]

# 建立投票模型
voting_model = VotingClassifier(estimators=models, voting='soft')

# 訓練投票模型
voting_model.fit(X_scaled, Y)

# 驗證資料
validation_df = pd.read_csv('validation.csv')
X_val = validation_df[required_columns].fillna(validation_df[required_columns].median()).to_numpy().astype(float)
y_true = validation_df['home_team_win'].to_numpy()
X_val_poly = poly.transform(X_val)
X_val_scaled = scaler.transform(X_val_poly)

# 取得各模型的預測機率
probas = np.array([clf.predict_proba(X_val_scaled)[:, 1] for clf in voting_model.named_estimators_.values()])
avg_proba = (probas[0]*2+probas[1]+probas[2]+probas[0])/5  # 計算平均機率
predictions = (avg_proba > 0.5).astype(int)  # 平均後的預測結果

# 計算驗證準確率
accuracy = accuracy_score(y_true, predictions)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 錯誤分析
errors = validation_df[predictions != y_true].copy()
errors['predicted_probability'] = avg_proba[predictions != y_true]
errors['predicted_label'] = predictions[predictions != y_true]
errors.to_csv('error.csv', index=False)  # 將錯誤結果保存至 CSV 檔案

# 測試資料
test_df = pd.read_csv('same_season_test_data.csv')
X_test = test_df[required_columns].fillna(train_df[required_columns].median()).to_numpy().astype(float)
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)

probas = np.array([clf.predict_proba(X_test_scaled)[:, 1] for clf in voting_model.named_estimators_.values()])
avg_proba = (probas[0]*2+probas[1]+probas[2]+probas[0])/5  # 計算平均機率
predictions = (avg_proba > 0.5).astype(int)  # 平均後的預測結果

# 輸出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(predictions)),
    'home_team_win': predictions
})
result_df.to_csv('result_ensemble.csv', index=False)

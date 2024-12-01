import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# 讀取訓練資料
df = pd.read_csv('train_dataallwash.csv')

# 處理空值

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
df[required_columns] = df[required_columns].fillna(df[required_columns].median())
X = df[required_columns].to_numpy().astype(float)
Y = df['home_team_win'].to_numpy().astype(float)

# 多項式擴展與標準化
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

correlation = df[required_columns + ['home_team_win']].corr()

# 輸出每個欄位與 Y 的相關係數（特別關注與 home_team_win 的相關性）
correlation_with_y = correlation['home_team_win'].drop('home_team_win')

# 按照相關係數的絕對值排序
correlation_with_y_sorted = correlation_with_y.abs().sort_values(ascending=False)

# 輸出排序後的結果
print("每個特徵與 home_team_win 的相關係數（按絕對值排序）:")
print(correlation_with_y_sorted)

# 將結果保存至 CSV 檔案，並包含標題
correlation_with_y_sorted.to_csv('feature_correlation_with_Y_sorted.csv', header=True)



# SVM 模型初始化
model = SVC(kernel='linear', probability=True)

# 5-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
auc_list = []
accuracy_list = []

plt.figure(figsize=(8, 6))  # 建立繪圖區域

for train_index, test_index in kf.split(X_scaled, Y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # 訓練模型
    model.fit(X_train, y_train)
    
    # 預測
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    
    # 計算 ROC 與 AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    auc_list.append(auc)
    
    # 繪製 ROC 曲線
    plt.plot(fpr, tpr, label=f"Fold {fold} AUC = {auc:.3f}")
    fold += 1

# 繪製平均 ROC 曲線
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for SVM")
plt.legend(loc='lower right')
plt.grid()
plt.show()

# 平均 AUC 與準確率
mean_auc = np.mean(auc_list)
mean_accuracy = np.mean(accuracy_list)
print(f"Mean AUC across 5 folds: {mean_auc:.3f}")
print(f"Mean Accuracy across 5 folds: {mean_accuracy * 100:.2f}%")

# 測試資料
test_df = pd.read_csv('same_season_test_data.csv')
X_test = test_df[required_columns].fillna(df[required_columns].median()).to_numpy().astype(float)
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)

# 測試資料預測
predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test_scaled)[:, 1]

# 匯出結果至 CSV
result_df = pd.DataFrame({
    'id': np.arange(len(predictions)),
    'home_team_win': predictions,
    'probability': test_probabilities
})
result_df.to_csv('result_svm.csv', index=False)

print("Test predictions saved to 'result_svm.csv'")

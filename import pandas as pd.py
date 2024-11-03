import pandas as pd
import numpy as np

# 讀取訓練資料
train_df = pd.read_csv('train_data.csv')  # 替換成你的訓練 CSV 檔案名稱

# 去除訓練資料中包含空值的行
train_df = train_df.dropna(subset=[
    'home_team_wins_mean', 
    'home_batting_batting_avg_10RA', 
    'home_batting_onbase_perc_10RA', 
    'home_batting_onbase_plus_slugging_10RA', 
    'home_batting_leverage_index_avg_10RA', 
    'home_batting_RBI_10RA', 
    'away_batting_batting_avg_10RA', 
    'away_batting_onbase_perc_10RA', 
    'away_batting_onbase_plus_slugging_10RA', 
    'away_batting_leverage_index_avg_10RA', 
    'away_batting_RBI_10RA'
])  # 替換成需要檢查空值的欄位

# 提取目標變量（Y）
Y = train_df['home_team_wins_mean'].values  # 目標變量，根據你的需要替換

# 提取特徵變量（X），並轉換為數字
X = train_df[['home_batting_batting_avg_10RA', 
               'home_batting_onbase_perc_10RA', 
               'home_batting_onbase_plus_slugging_10RA', 
               'home_batting_leverage_index_avg_10RA', 
               'home_batting_RBI_10RA', 
               'away_batting_batting_avg_10RA', 
               'away_batting_onbase_perc_10RA', 
               'away_batting_onbase_plus_slugging_10RA', 
               'away_batting_leverage_index_avg_10RA', 
               'away_batting_RBI_10RA']].apply(pd.to_numeric, errors='coerce').values

# 在 X 中添加一列常數項（x0）
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 在最左側添加一列全為 1 的數組

# 計算最佳權重（w）使用伽瑪逆運算
w = np.linalg.pinv(X.T @ X) @ (X.T @ Y)

# 讀取測試資料
test_df = pd.read_csv('same_season_test_data.csv')  # 替換成你的測試 CSV 檔案名稱

# 將所有數值列轉換為數字，並填補空值為該列的平均值
test_df = test_df.apply(pd.to_numeric, errors='coerce')  # 將所有列轉換為數字
test_df.fillna(test_df.mean(), inplace=True)  # 用列的平均值填補空值

# 提取特徵變量（X_test）
X_test = test_df[['home_batting_batting_avg_10RA', 
                   'home_batting_onbase_perc_10RA', 
                   'home_batting_onbase_plus_slugging_10RA', 
                   'home_batting_leverage_index_avg_10RA', 
                   'home_batting_RBI_10RA', 
                   'away_batting_batting_avg_10RA', 
                   'away_batting_onbase_perc_10RA', 
                   'away_batting_onbase_plus_slugging_10RA', 
                   'away_batting_leverage_index_avg_10RA', 
                   'away_batting_RBI_10RA']].values

# 在 X_test 中添加一列常數項（x0）
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# 使用回歸模型預測
predictions = X_test @ w  # 計算預測值

# 將預測值轉換為 1 或 0，這裡假設閾值為 0.5
predictions_binary = (predictions >= 0.5).astype(int)

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions_binary)),  # id 從 0 到 6184
    'win': predictions_binary  # 預測結果
})

# 輸出為 CSV
result_df.to_csv('result.csv', index=False)

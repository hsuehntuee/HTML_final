import pandas as pd
import numpy as np

# 定義 SVM 類別
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 初始化權重和偏差
        self.w = np.zeros(n_features)
        self.b = 0

        # 轉換 y 以獲得 -1 和 1
        y_ = np.where(y <= 0, -1, 1)

        # 梯度下降
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # 正確分類
                    dw = self.lambda_param * self.w
                    db = 0
                else:
                    # 錯誤分類
                    dw = self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)  # 會輸出 1 或 -1

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

# 提取目標變量（Y）並轉換為 1 和 -1
Y = train_df['home_team_wins_mean'].values  # 目標變量，根據你的需要替換
Y = np.where(Y > 0, 1, -1)  # 將勝利轉換為 1，輸轉換為 -1

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

# 去除在 X 和 Y 中的空值
valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X = X[valid_indices]
Y = Y[valid_indices]

# 創建 SVM 模型
model = SVM()

# 訓練模型
model.fit(X, Y)

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

# 在 X_test 中去除空值
X_test = X_test[~np.isnan(X_test).any(axis=1)]

# 使用 SVM 模型進行預測
predictions = model.predict(X_test)

# 將預測結果轉換為 0（輸）和 1（贏）
predictions = np.where(predictions == 1, 1, 0)  # 將 1 轉為 1，將 -1 轉為 0

# 建立結果 DataFrame
result_df = pd.DataFrame({
    'id': np.arange(len(predictions)),  # id 從 0 到 6184
    'home_team_win': predictions  # 預測結果
})

# 輸出為 CSV
result_df.to_csv('result.csv', index=False)

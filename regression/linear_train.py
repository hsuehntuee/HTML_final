import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 定義線性回歸類別
class LinearRegression:
    def __init__(self):
        self.w = None
        
    def theta(self, x, y):
        z = self.w @ x
        if z > 10:
            z = 10
        elif z < -10:
            z = -10
        return 1 / (1 + np.exp(y * z))

    def fit(self, X, y, eta, epochs):
        # Compute weights w
        N, d = X.shape
        self.w = np.zeros(d)  # Initialize weights based on number of features
        y = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        
        for _ in range(epochs):  # Iterate over the number of epochs
            idx = np.random.randint(N)  # Randomly select an index

            prediction = X[idx] @ self.w
            
            gradient = 2 * (prediction - y[idx]) * X[idx]
            gradient = np.clip(gradient, -10, 10)
            self.w -= eta * gradient

    def predict(self, X):
        z = X @ self.w
        return np.where(z >= 0, 1, -1)



# 去除訓練資料中包含空值的行

required_columns = [
    'home_team_rest', 'away_team_rest', 'home_pitcher_rest', 'away_pitcher_rest',
    'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 'home_batting_onbase_plus_slugging_10RA', 
    'home_batting_leverage_index_avg_10RA', 'home_batting_RBI_10RA', 'away_batting_batting_avg_10RA', 
    'away_batting_onbase_perc_10RA', 'away_batting_onbase_plus_slugging_10RA', 'away_batting_leverage_index_avg_10RA', 
    'away_batting_RBI_10RA', 'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA', 
    'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA', 'away_pitching_earned_run_avg_10RA', 
    'away_pitching_SO_batters_faced_10RA', 'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA', 
    'home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 
    'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 
    'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA', 'home_team_errors_mean', 'home_team_errors_std', 'home_team_errors_skew', 
    'away_team_errors_mean', 'away_team_errors_std', 'away_team_errors_skew', 'home_team_spread_mean', 'home_team_spread_std', 
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
    'date_standardized'
]

data_type = "1_original_average_first"
# Read data
df1 = pd.read_csv(f'_SUPER_DATA/{data_type}/stage_train.csv')
df2 = pd.read_csv(f'_SUPER_DATA/{data_type}/stage_validation.csv')
df3 = pd.read_csv(f'_SUPER_DATA/{data_type}/stage1_test.csv')
df4 = pd.read_csv(f'_SUPER_DATA/{data_type}/stage1_label.csv')

# Handle missing values by filling with column means
#df1[required_columns] = df1[required_columns].fillna(df1[required_columns].mean())
#df2[required_columns] = df2[required_columns].fillna(df2[required_columns].mean())
#df3[required_columns] = df3[required_columns].fillna(df3[required_columns].mean())

# Convert to numpy arrays
X1 = df1[required_columns].to_numpy().astype(float)
X2 = df2[required_columns].to_numpy().astype(float)
X3 = df3[required_columns].to_numpy().astype(float)
Y1 = df1['home_team_win'].to_numpy().astype(int)
Y2 = df2['home_team_win'].to_numpy().astype(int)
Y3 = df4['home_team_win'].to_numpy().astype(int)

# Concatenate the data
X = np.vstack((X1, X2))
Y = np.concatenate((Y1, Y2), axis=0)
Y = np.where(Y == 0, -1, Y)
Y3 = np.where(Y3 == 0, -1, Y3)
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_test = np.hstack((np.ones((X3.shape[0], 1)), X3))
Y_test = Y3
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X, Y, 0.001, 600000)

predictions = model.predict(X_test)

accuracy = np.mean(predictions == Y_test)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

train_predictions = model.predict(X)

Ein = np.mean(train_predictions != Y)

print(f"In-sample Error (Ein): {Ein * 100:.2f}%")
print(np.mean(predictions))
print(predictions[0:10])
print(Y_test[0:10])


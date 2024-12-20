import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self):
        self.w = None
        
    def theta(self, x, y):
        z = self.w @ x
        z = np.clip(z, -10, 10)
        return 1 / (1 + np.exp(y * z))

    def fit(self, X, y, eta, epochs, lamb):
        # Compute weights w
        N, d = X.shape
        self.w = np.zeros(d)  # Initialize weights based on number of features
        y = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1
        
        for _ in range(epochs):  # Iterate over the number of epochs
            idx = np.random.randint(N)  # Randomly select an index
            # Update weights
            #gradient = eta * self.theta(X[idx], y[idx]) * y[idx] * X[idx]
            gradient = self.theta(X[idx], y[idx]) * y[idx] * X[idx] + lamb * self.w
            gradient = np.clip(gradient, -10, 10)
            self.w += eta * gradient
            #print(self.w)

    def predict(self, X):
        z = X @ self.w
        print(np.mean(z))
        z = np.clip(z, -10, 10)
        probabilities = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        print(np.mean(probabilities))
        return np.where(probabilities >= 0.5, 1, 0)
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

data_type = "4_reverse_average_first"
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
X = np.hstack((np.ones((X.shape[0], 1)), X))
X_test = np.hstack((np.ones((X3.shape[0], 1)), X3))
Y_test = Y3
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)




model = LogisticRegression()
model.fit(X, Y, 0.0001, 200000, 0)

# 讀取驗證資料並填補空值


# 預測結果
predictions = model.predict(X_test)

accuracy = np.mean(predictions == Y_test)

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

train_predictions = model.predict(X)

Ein = np.mean(train_predictions != Y)

print(f"In-sample Error (Ein): {Ein * 100:.2f}%")
print(np.mean(predictions))
print(predictions[0:10])
print(Y_test[0:10])
'''

weights_abs = np.abs(model.w[1:])  # Skip the bias term (w[0])

# 2. Create a dictionary of feature names and their corresponding absolute weights
feature_weights = dict(zip(required_columns, weights_abs))

# 3. Sort the dictionary by absolute weights in descending order
sorted_feature_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)

for i in sorted_feature_weights:
    print(i)
    '''
import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        self.w = np.linalg.pinv(X) @ y  # Compute the weights using the pseudo-inverse

    def predict(self, X):
        return X.dot(self.w)  # Return the continuous output

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
'home_pitching_H_batters_faced_skew'
]
'''
required_columns = [
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
    'home_batting_onbase_perc_skew', 'home_batting_onbase_plus_slugging_mean', 'home_batting_onbase_plus_slugging_std', 
    #'home_batting_onbase_plus_slugging_skew', 'home_batting_leverage_index_avg_mean', 
    #'home_batting_leverage_index_avg_std', 'home_batting_leverage_index_avg_skew', 'home_batting_wpa_bat_mean', 
    #'home_batting_wpa_bat_std', 'home_batting_wpa_bat_skew', 'home_batting_RBI_mean', 'home_batting_RBI_std', 
    #'home_batting_RBI_skew', 'away_batting_batting_avg_mean', 'away_batting_batting_avg_std', 
    #'away_batting_batting_avg_skew', 'away_batting_onbase_perc_mean', 'away_batting_onbase_perc_std', 
    #'away_batting_onbase_perc_skew', 'away_batting_onbase_plus_slugging_mean', 'away_batting_onbase_plus_slugging_std', 
    #'away_batting_onbase_plus_slugging_skew', 'away_batting_leverage_index_avg_mean', 'away_batting_leverage_index_avg_std',

    #'away_batting_leverage_index_avg_skew', 'away_batting_wpa_bat_mean', 'away_batting_wpa_bat_std', 
    'away_batting_wpa_bat_skew', 'away_batting_RBI_mean', 'away_batting_RBI_std', 'away_batting_RBI_skew', 
    'home_pitching_earned_run_avg_mean', 'home_pitching_earned_run_avg_std', 'home_pitching_earned_run_avg_skew', 
    #'home_pitching_SO_batters_faced_mean', 'home_pitching_SO_batters_faced_std', 'home_pitching_SO_batters_faced_skew', 
    'home_pitching_H_batters_faced_mean', 'home_pitching_H_batters_faced_std', 'home_pitching_H_batters_faced_skew', 
    'home_pitching_BB_batters_faced_mean', 'home_pitching_BB_batters_faced_std', 'home_pitching_BB_batters_faced_skew', 
    'home_pitching_leverage_index_avg_mean', 'home_pitching_leverage_index_avg_std', 
    #'home_pitching_leverage_index_avg_skew', 'home_pitching_wpa_def_mean', 'home_pitching_wpa_def_std', 
    'home_pitching_wpa_def_skew', 'away_pitching_earned_run_avg_mean', 'away_pitching_earned_run_avg_std', 
    'away_pitching_earned_run_avg_skew', 'away_pitching_SO_batters_faced_mean', 'away_pitching_SO_batters_faced_std', 
    'away_pitching_SO_batters_faced_skew', 'away_pitching_H_batters_faced_mean', 'away_pitching_H_batters_faced_std', 
    #'away_pitching_H_batters_faced_skew', 'away_pitching_BB_batters_faced_mean', 'away_pitching_BB_batters_faced_std', 
    #'away_pitching_BB_batters_faced_skew', 'away_pitching_leverage_index_avg_mean', 'away_pitching_leverage_index_avg_std', 
    'away_pitching_leverage_index_avg_skew', 'away_pitching_wpa_def_mean', 'away_pitching_wpa_def_std', 
    'away_pitching_wpa_def_skew', 'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std', 
    'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean', 'home_pitcher_SO_batters_faced_std', 
    'home_pitcher_SO_batters_faced_skew', 'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std', 
    #'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean', 'home_pitcher_BB_batters_faced_std', 
    'home_pitcher_BB_batters_faced_skew', 'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std', 
    'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean', 'home_pitcher_wpa_def_std', 
    'home_pitcher_wpa_def_skew', 'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std', 
    'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean', 'away_pitcher_SO_batters_faced_std', 
    #'away_pitcher_SO_batters_faced_skew', 'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean', 'away_pitcher_BB_batters_faced_std', 
    #'away_pitcher_BB_batters_faced_skew', 'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std', 
    #'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean', 'away_pitcher_wpa_def_std', 
    #'away_pitcher_wpa_def_skew'
]
'''
'''
required_columns = [
    'home_batting_batting_avg_10RA', 'home_batting_onbase_perc_10RA', 'home_batting_onbase_plus_slugging_10RA', 
    'home_batting_leverage_index_avg_10RA', 'home_batting_RBI_10RA', 'away_batting_batting_avg_10RA', 
    'away_batting_onbase_perc_10RA', 'away_batting_onbase_plus_slugging_10RA', 'away_batting_leverage_index_avg_10RA', 
    'away_batting_RBI_10RA', 'home_pitching_earned_run_avg_10RA', 'home_pitching_SO_batters_faced_10RA', 
    'home_pitching_H_batters_faced_10RA', 'home_pitching_BB_batters_faced_10RA', 'away_pitching_earned_run_avg_10RA', 
    'away_pitching_SO_batters_faced_10RA', 'away_pitching_H_batters_faced_10RA', 'away_pitching_BB_batters_faced_10RA', 
    'home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 
    'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 
    'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA', 'home_team_errors_mean', 'home_team_errors_std', 'home_team_errors_skew', 'away_team_errors_mean', 
    'away_team_errors_std', 'away_team_errors_skew', 'home_team_spread_mean', 'home_team_spread_std', 
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
    'away_pitching_wpa_def_skew', 'home_pitcher_earned_run_avg_mean', 'home_pitcher_earned_run_avg_std', 
    'home_pitcher_earned_run_avg_skew', 'home_pitcher_SO_batters_faced_mean', 'home_pitcher_SO_batters_faced_std', 
    'home_pitcher_SO_batters_faced_skew', 'home_pitcher_H_batters_faced_mean', 'home_pitcher_H_batters_faced_std', 
    'home_pitcher_H_batters_faced_skew', 'home_pitcher_BB_batters_faced_mean', 'home_pitcher_BB_batters_faced_std', 
    'home_pitcher_BB_batters_faced_skew', 'home_pitcher_leverage_index_avg_mean', 'home_pitcher_leverage_index_avg_std', 
    'home_pitcher_leverage_index_avg_skew', 'home_pitcher_wpa_def_mean', 'home_pitcher_wpa_def_std', 
    'home_pitcher_wpa_def_skew', 'away_pitcher_earned_run_avg_mean', 'away_pitcher_earned_run_avg_std', 
    'away_pitcher_earned_run_avg_skew', 'away_pitcher_SO_batters_faced_mean', 'away_pitcher_SO_batters_faced_std', 
    'away_pitcher_SO_batters_faced_skew', 'away_pitcher_H_batters_faced_mean', 'away_pitcher_H_batters_faced_std', 
    'away_pitcher_H_batters_faced_skew', 'away_pitcher_BB_batters_faced_mean', 'away_pitcher_BB_batters_faced_std', 
    'away_pitcher_BB_batters_faced_skew', 'away_pitcher_leverage_index_avg_mean', 'away_pitcher_leverage_index_avg_std', 
    'away_pitcher_leverage_index_avg_skew', 'away_pitcher_wpa_def_mean', 'away_pitcher_wpa_def_std', 
    'away_pitcher_wpa_def_skew'
]
'''

# Load training data
train_df = pd.read_csv('1120commit/2016.csv')
threshold = 0.0  # You can adjust this value based on your dataset. Here, it's 50% missing data.

# Calculate the number of non-missing values per row
non_missing_count = train_df[required_columns].notna().sum(axis=1)

# Calculate the number of total columns per row
total_columns = len(required_columns)

# Keep only the rows where the number of non-missing values is greater than the threshold
train_df = train_df[non_missing_count / total_columns >= threshold]


imputer = KNNImputer(n_neighbors=5)

# Handle missing data
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())
#train_df[required_columns] = imputer.fit_transform(train_df[required_columns])
# Prepare the feature matrix (X) and target vector (Y)

X = train_df[required_columns].to_numpy().astype(float)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias column
Y = train_df['home_team_win'].to_numpy().astype(float)  # Target variable


X, X_val, Y, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



# Bagging with 100 models
n_models = 1000
models = []
predictions_all = np.zeros(len(Y))

for _ in range(n_models):
    # Bootstrap sampling: Create a random subset of the data
    X_resampled, Y_resampled = resample(X, Y, replace=True)
    
    # Train a linear regression model on the resampled data
    model = LinearRegression()
    model.fit(X_resampled, Y_resampled)
    models.append(model)
    
    # Make predictions for this model
    predictions = (model.predict(X) >= 0.5).astype(int)
    predictions_all += predictions
    

predictions_all = ((predictions_all / n_models) >= 0.5).astype(int)


# Calculate accuracy for bagging model on training data
Ein = np.mean(predictions_all != Y)
print(f"Bagging In-sample Error (Ein): {Ein * 100:.2f}%")

# Make predictions on validation data using the bagging models
validation_df = pd.read_csv('validation.csv')

X_val = validation_df[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))  # Add bias column
y_true = validation_df['home_team_win'].to_numpy()

# Get predictions for each model on the validation data
predictions_val_all = np.zeros(len(y_true))
for model in models:
    predictions_val = ((model.predict(X_val)) >= 0.5).astype(int)
    predictions_val_all += predictions_val

predictions_val_all = ((predictions_val_all / n_models) >= 0.5).astype(int)


# Calculate accuracy on the validation set
validation_accuracy = np.mean(predictions_val_all == y_true)
print(f"Bagging Validation Accuracy: {validation_accuracy * 100:.2f}%")

'''
newo = pd.read_csv('../same_season_test_data.csv')
X_test =newo[required_columns].fillna(validation_df[required_columns].mean()).to_numpy().astype(float)
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
predictions2 = model.predict(X_test)


result_df = pd.DataFrame({
    'id': np.arange(len(predictions2)),
    'home_team_win': predictions2 
})

result_df.to_csv('result87.csv', index=False)
'''
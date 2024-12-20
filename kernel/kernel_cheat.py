from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# Load required data (the same as in your original code)
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
df3 = df3.sort_values(by='id')
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
# Perform grid search to tune the SVM parameters
'''
svm_model = SVC(kernel='rbf', random_state=42)
param_grid = {
    'C': [1000],  # Regularization parameter
    'gamma': [1000]  # Gamma values for RBF kernel
}
'''

svm_model = SVC(kernel='poly', random_state=42)
param_grid = {
    'C': [0.001],  # Regularization parameter, testing a wider range
    'degree': [3],     # Degree of the polynomial kernel, usually 2 to 4 is a good range
    'coef0': [500],  # Constant term, typically in the range of 0 to 10
}
#'''
grid_search = GridSearchCV(svm_model, param_grid, cv=5, verbose=2)
grid_search.fit(X, Y)

# Get the best SVM model after grid search
best_svm_model = grid_search.best_estimator_

# Fit the model with all data (it was already fitted during grid search)
best_svm_model.fit(X, Y)  # Ensure the model is fitted before feature selection

# Get all the coefficients (w_i) for all features
#coefficients = best_svm_model.coef_[0]  # Coefficients for all features

# Print all coefficients (w_i)
#print("All weights (w_i) for all features:")
#for feature, weight in zip(required_columns, coefficients):
#    print(f"{feature}: {weight}")

Y_pred = best_svm_model.predict(X3)  # Predict on the validation set X2

# Generate classification report
print("\nValidation Classification Report:")
print(classification_report(Y3, Y_pred))
print(np.mean(Y_pred))

print("Best hyperparameters:", grid_search.best_params_)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the training data
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
    'away_pitcher_wpa_def_skew',
    'date_standardized'
]

train_df = pd.read_csv('balanced_train_data.csv')
train_df = train_df[train_df['home_team_abbr'] == 'XFB']
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

# Prepare the feature matrix (X) and target vector (Y)
X = train_df[required_columns].to_numpy().astype(float)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias column
Y = train_df['home_team_win'].to_numpy().astype(int)  # Target variable (convert to int if needed)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the base model: a decision stump (max_depth=1 DecisionTreeClassifier)
decision_stump = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoost with the decision stump as the base estimator
adaboost_model = AdaBoostClassifier(estimator=decision_stump, n_estimators=10, random_state=42, algorithm='SAMME')


# Train the AdaBoost model
adaboost_model.fit(X_train, Y_train)

# Get predictions on the training set
predictions_train = adaboost_model.predict(X_train)
Ein = np.mean(predictions_train != Y_train)
print(f"AdaBoost In-sample Error (Ein): {Ein * 100:.2f}%")

# Get predictions on the validation set
predictions_val = adaboost_model.predict(X_val)
validation_accuracy = accuracy_score(Y_val, predictions_val)
print(f"AdaBoost Validation Accuracy: {validation_accuracy * 100:.2f}%")

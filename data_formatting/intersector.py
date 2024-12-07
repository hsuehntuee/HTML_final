# First list of features with their importance values (low importance)
low_importance_features = [
    ('away_pitching_SO_batters_faced_10RA', 0.0082),
    ('away_pitching_SO_batters_faced_mean', 0.0081),
    ('away_pitcher_SO_batters_faced_10RA', 0.0075),
    ('home_pitcher_wpa_def_mean', 0.0075),
    ('home_pitcher_leverage_index_avg_skew', 0.0073),
    ('away_pitcher_SO_batters_faced_mean', 0.0073),
    ('home_pitcher_SO_batters_faced_skew', 0.0072),
    ('home_pitching_BB_batters_faced_10RA', 0.0072),
    ('home_pitcher_SO_batters_faced_mean', 0.0071),
    ('home_pitcher_SO_batters_faced_10RA', 0.0071),
    ('away_pitcher_BB_batters_faced_skew', 0.0071),
    ('away_pitcher_leverage_index_avg_skew', 0.0071),
    ('away_pitching_earned_run_avg_10RA', 0.0070),
    ('home_pitcher_earned_run_avg_skew', 0.0070),
    ('home_team_spread_mean', 0.0070),
    ('home_pitcher_wpa_def_skew', 0.0070),
    ('away_pitcher_wpa_def_skew', 0.0070),
    ('home_pitcher_H_batters_faced_skew', 0.0069),
    ('away_pitcher_earned_run_avg_skew', 0.0069),
    ('home_pitcher_BB_batters_faced_skew', 0.0069),
    ('away_pitching_wpa_def_skew', 0.0069),
    ('away_pitcher_SO_batters_faced_skew', 0.0069),
    ('home_pitcher_leverage_index_avg_mean', 0.0068),
    ('away_pitcher_H_batters_faced_skew', 0.0068),
    ('away_pitcher_H_batters_faced_std', 0.0068),
    ('away_pitching_BB_batters_faced_10RA', 0.0068),
    ('home_pitching_earned_run_avg_mean', 0.0068),
    ('away_batting_onbase_perc_mean', 0.0068),
    ('home_pitcher_leverage_index_avg_std', 0.0068),
    ('home_pitcher_wpa_def_std', 0.0068),
    ('away_pitching_BB_batters_faced_mean', 0.0067)
]

# Second list of features with their importance values (higher importance)
high_importance_features = [
    ('away_batting_onbase_perc_mean', 0.1300),
    ('home_batting_onbase_plus_slugging_mean', 0.0991),
    ('home_pitcher_wpa_def_mean', 0.0927),
    ('away_team_rest', 0.0905),
    ('home_team_rest', 0.0885),
    ('away_batting_batting_avg_mean', 0.0829),
    ('home_pitching_earned_run_avg_std', 0.0781),
    ('away_pitching_SO_batters_faced_10RA', 0.0777),
    ('away_pitcher_earned_run_avg_std', 0.0771),
    ('away_pitching_SO_batters_faced_mean', 0.0755),
    ('home_pitcher_H_batters_faced_mean', 0.0678),
    ('away_batting_leverage_index_avg_10RA', 0.0674),
    ('home_team_spread_std', 0.0654),
    ('away_batting_onbase_perc_skew', 0.0627),
    ('home_pitching_H_batters_faced_mean', 0.0616),
    ('home_pitching_BB_batters_faced_10RA', 0.0603),
    ('home_batting_batting_avg_mean', 0.0602),
    ('home_batting_onbase_plus_slugging_10RA', 0.0599),
    ('home_team_wins_mean', 0.0542),
    ('away_batting_RBI_skew', 0.0538),
    ('away_pitching_earned_run_avg_mean', 0.0522),
    ('home_batting_wpa_bat_mean', 0.0503),
    ('away_pitching_wpa_def_mean', 0.0503),
    ('home_batting_batting_avg_skew', 0.0493),
    ('home_pitching_BB_batters_faced_skew', 0.0492),
    ('home_pitcher_SO_batters_faced_10RA', 0.0488),
    ('away_batting_RBI_std', 0.0476),
    ('home_pitching_leverage_index_avg_std', 0.0463),
    ('away_batting_onbase_plus_slugging_10RA', 0.0459),
    ('away_pitching_BB_batters_faced_mean', 0.0455),
    ('home_batting_onbase_plus_slugging_skew', 0.0455),
    ('away_pitcher_earned_run_avg_10RA', 0.0448)
]

# Extract feature names from both lists
low_feature_names = [feature[0] for feature in low_importance_features]
high_feature_names = [feature[0] for feature in high_importance_features]

# Find the intersection of feature names
intersecting_features = list(set(low_feature_names) & set(high_feature_names))

# Print the intersecting features
print("Intersecting Features:")
for feature in intersecting_features:
    print(f"'{feature}',")
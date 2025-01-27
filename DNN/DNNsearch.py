import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import keras_tuner as kt

# 載入與處理資料
train_df = pd.read_csv('filled_kaggle_train.csv')


# Truncate required columns to a smaller sample for demonstration
required_columns = [
    'is_night_game', 'home_team_rest', 'away_team_rest', 'home_pitcher_rest', 'away_pitcher_rest',
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
    'away_pitching_wpa_def_skew', 
    'date_standardized'

]
# 載入與處理資料
train_df = pd.read_csv('filled_kaggle_train.csv')

# 自動選擇所有數值型欄位
required_columns = train_df.select_dtypes(include=np.number).columns.tolist()
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(int)

# 測試集處理
newo = pd.read_csv('kaggle_test.csv')
#X_test = newo[required_columns].fillna(newo[required_columns].mean()).to_numpy().astype(float)

# 設定 K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 建立超參數調優函式
def create_model(hp):
    # Input layer
    inputs = Input(shape=(X.shape[1],))
    
    # First hidden layer with LeakyReLU activation
    x = Dense(hp.Int('units_1', min_value=512, max_value=4096, step=512), 
              kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    # Second hidden layer with LeakyReLU activation
    x = Dense(hp.Int('units_2', min_value=256, max_value=4096, step=256), 
              kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with Adam optimizer and binary crossentropy loss
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')), metrics=['accuracy'])
    
    return model

# 設定 Keras Tuner
tuner = kt.Hyperband(
    create_model,
    objective='val_accuracy', 
    max_epochs=300,
    factor=3,
    directory='kt_tuning',
    project_name='baseball_model_tuning'
)

# K-Fold Cross Validation + Hyperparameter Tuning
best_val_accuracy = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X, Y)):
    print(f"Training Fold {fold + 1}...")

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # 使用 Keras Tuner 進行調參
    tuner.search(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val), verbose=2)
    
    # 擷取最佳模型
    best_model_temp = tuner.get_best_models(num_models=1)[0]
    
    # 計算驗證集準確率
    y_val_pred = (best_model_temp.predict(X_val) > 0.5).astype(int)
    val_accuracy = accuracy_score(Y_val, y_val_pred)
    print(f"Validation Accuracy (Fold {fold + 1}): {val_accuracy * 100:.2f}%")
    
    # 儲存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = best_model_temp

# 測試新檔案並輸出結果
#predictions2 = (best_model.predict(X_test) > 0.5).astype(int)
#result_df = pd.DataFrame({'id': np.arange(len(predictions2)), 'home_team_win': predictions2.flatten()})
#result_df.to_csv('result_team_only_svm_tuned.csv', index=False)
#print("Test predictions saved to 'result_team_only_svm_tuned.csv'.")
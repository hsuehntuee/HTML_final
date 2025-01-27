

# 取得輸入所需欄位
def get_required_columns():
    # 根據需要設定 required_columns
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
    return required_columns
# 取得輸入所需欄位

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, log_loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import time
start = time.time()
# 載入資料
train_df = pd.read_csv('stage_train.csv')
val_df = pd.read_csv('stage_validation.csv')

test_df = pd.read_csv('stage1_test.csv')
test2_df = pd.read_csv('stage2_test.csv')
answer_df = pd.read_csv('stage1_label.csv')  # 載入預測解答
answer2_df = pd.read_csv('stage2_label.csv')
required_columns = get_required_columns()

# 填補缺失值與裁剪數據範圍
def preprocess_data(df, required_columns, lower_bounds=None, upper_bounds=None):
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())
    if lower_bounds is not None and upper_bounds is not None:
        df[required_columns] = df[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    return df

# 選擇是否要進行數據裁剪
clip_choice = input("Do you want to clip data? (yes/no): ").strip().lower()
lower_bounds = None
upper_bounds = None
if clip_choice == 'yes':
    quantiles = train_df[required_columns].quantile([0.15, 0.85])
    lower_bounds = quantiles.loc[0.15]
    upper_bounds = quantiles.loc[0.85]

train_df = preprocess_data(train_df, required_columns, lower_bounds, upper_bounds)


# 特徵與標籤處理
X_train = train_df[required_columns].to_numpy().astype(float)
Y_train = train_df['home_team_win'].to_numpy().astype(int)
X_val = val_df[required_columns].to_numpy().astype(float)
Y_val = val_df['home_team_win'].to_numpy().astype(int)
X_test = test_df[required_columns].to_numpy().astype(float)
Y_test = answer_df['home_team_win'].to_numpy().astype(int)
X_test2 = test2_df[required_columns].to_numpy().astype(float)
Y_test2 = answer2_df['home_team_win'].to_numpy().astype(int)

# Scaling 選擇
scaler = None
scaling_choice = input("Choose scaling method (robust/standard/none): ").strip().lower()
if scaling_choice == 'robust':
    scaler = RobustScaler()
elif scaling_choice == 'standard':
    scaler = StandardScaler()
else: 
    print("Warning") 
if scaler:
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_test2 = scaler.transform(X_test2)

# 模型結構
def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(1024, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(128, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=1.0694e-05), 
                  metrics=['accuracy'])
    return model

# 訓練模型
model = create_model(input_dim=X_train.shape[1])

# 詳細跑背訊
history = {
    'Ein': [],
    'Evalidation': [],
    'Eout_test1': [],
    'Eout_test2': []
}

start_time = time.time()
for epoch in range(20):
    model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=2, validation_data=(X_val, Y_val))

    # Ein
    train_predictions = (model.predict(X_train) > 0.5).astype(int)
    ein = accuracy_score(Y_train, train_predictions)
    history['Ein'].append(ein)

    # Evalidation
    val_predictions = (model.predict(X_val) > 0.5).astype(int)
    evalidation = accuracy_score(Y_val, val_predictions)
    history['Evalidation'].append(evalidation)

    # Eout_test1
    test_predictions = (model.predict(X_test) > 0.5).astype(int)
    eout_test1 = accuracy_score(Y_test, test_predictions)
    history['Eout_test1'].append(eout_test1)

    # Eout_test2
    test2_predictions = (model.predict(X_test2) > 0.5).astype(int)
    eout_test2 = accuracy_score(Y_test2, test2_predictions)
    history['Eout_test2'].append(eout_test2)

print("Training time: ", time.time() - start_time)

# 畫出跑背總結果
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), history['Ein'], label='InSampleAccuracy', marker='o')
plt.plot(range(1, 21), history['Evalidation'], label='ValidationAccuracy', marker='o')
plt.plot(range(1, 21), history['Eout_test1'], label='Test1Accuracy', marker='o')
plt.plot(range(1, 21), history['Eout_test2'], label='Test2Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('InAccuracy, ValidationAccuracy, Test1accuracy, and Test2accuracy over Epochs')
plt.legend()
plt.grid()
plt.savefig('training_progress.png')
plt.show()

# 測試模型
predictions_test1 = (model.predict(X_test) > 0.5).astype(int)
result_df_test1 = pd.DataFrame({'id': np.arange(len(predictions_test1)), 'home_team_win': predictions_test1.flatten()})
result_df_test1.to_csv('result_team_DNN2_test1.csv', index=False)
print("Test1 predictions saved to 'result_team_DNN2_test1.csv'.")

predictions_test2 = (model.predict(X_test2) > 0.5).astype(int)
result_df_test2 = pd.DataFrame({'id': np.arange(len(predictions_test2)), 'home_team_win': predictions_test2.flatten()})
result_df_test2.to_csv('result_team_DNN2_test2.csv', index=False)
print("Test2 predictions saved to 'result_team_DNN2_test2.csv'.")

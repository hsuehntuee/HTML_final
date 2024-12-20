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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 載入與處理資料
train_df = pd.read_csv('stage12_wash_dateFormatted_train.csv')
newo = pd.read_csv('filled_kaggle_test.csv')

# Truncate required columns to a smaller sample for demonstration
required_columns = [
'home_team_rest',
'away_team_rest',
'away_pitcher_rest',
'home_batting_batting_avg_10RA',
'home_batting_onbase_perc_10RA',
'home_batting_onbase_plus_slugging_10RA',
'home_batting_RBI_10RA',
'away_batting_batting_avg_10RA',
'away_batting_onbase_perc_10RA',
'away_batting_onbase_plus_slugging_10RA',
'away_batting_leverage_index_avg_10RA',
'away_batting_RBI_10RA',
'home_pitching_earned_run_avg_10RA',
'home_pitching_SO_batters_faced_10RA',
'home_pitching_H_batters_faced_10RA',
'home_pitching_BB_batters_faced_10RA',
'away_pitching_earned_run_avg_10RA',
'away_pitching_SO_batters_faced_10RA',
'away_pitching_H_batters_faced_10RA',
'away_pitching_BB_batters_faced_10RA',
'home_team_errors_mean',
'home_team_errors_skew',
'away_team_errors_mean',
'away_team_errors_skew',
'home_team_spread_mean',
'away_team_spread_mean',
'home_team_wins_mean',
'home_team_wins_skew',
'away_team_wins_mean',
'away_team_wins_skew',
'home_batting_batting_avg_mean',
'home_batting_batting_avg_skew',
'home_batting_onbase_perc_mean',
'home_batting_onbase_perc_skew',
'home_batting_onbase_plus_slugging_mean',
'home_batting_onbase_plus_slugging_std',
'home_batting_onbase_plus_slugging_skew',
'home_batting_wpa_bat_mean',
'home_batting_wpa_bat_std',
'home_batting_wpa_bat_skew',
'home_batting_RBI_mean',
'home_batting_RBI_std',
'home_batting_RBI_skew',
'away_batting_batting_avg_mean',
'away_batting_onbase_perc_mean',
'away_batting_onbase_plus_slugging_mean',
'away_batting_wpa_bat_mean',
'away_batting_wpa_bat_skew',
'away_batting_RBI_mean',
'away_batting_RBI_std',
'away_batting_RBI_skew',
'home_pitching_earned_run_avg_mean',
'home_pitching_earned_run_avg_std',
'home_pitching_SO_batters_faced_mean',
'home_pitching_SO_batters_faced_std',
'home_pitching_H_batters_faced_mean',
'home_pitching_BB_batters_faced_mean',
'home_pitching_leverage_index_avg_std',
'home_pitching_wpa_def_mean',
'home_pitching_wpa_def_skew',
'away_pitching_earned_run_avg_mean',
'away_pitching_earned_run_avg_std',
'away_pitching_SO_batters_faced_mean',
'away_pitching_SO_batters_faced_std',
'away_pitching_H_batters_faced_mean',
'away_pitching_BB_batters_faced_mean',
'away_pitching_BB_batters_faced_std',
'away_pitching_leverage_index_avg_mean',
'away_pitching_wpa_def_mean',
'away_pitching_wpa_def_skew',
]


no = ['home_pitcher_earned_run_avg_10RA', 'home_pitcher_SO_batters_faced_10RA', 'home_pitcher_H_batters_faced_10RA', 'home_pitcher_BB_batters_faced_10RA', 'away_pitcher_earned_run_avg_10RA', 'away_pitcher_SO_batters_faced_10RA', 'away_pitcher_H_batters_faced_10RA', 'away_pitcher_BB_batters_faced_10RA']
required_columns = [col for col in required_columns if col not in no]
train_df[required_columns] = train_df[required_columns].fillna(train_df[required_columns].mean())

quantiles = train_df[required_columns].quantile([0.1, 0.9])

# 分別取出上下界
lower_bounds = quantiles.loc[0.1]
upper_bounds = quantiles.loc[0.9]

# 對訓練和測試數據裁剪
train_df[required_columns] = train_df[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)
newo[required_columns] = newo[required_columns].clip(lower=lower_bounds, upper=upper_bounds, axis=1)

'''
for feature in required_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='home_team_win', y=feature, data=train_df)
    plt.xlabel('Home Team Win')
    plt.ylabel(feature)
    plt.title(f'{feature} Distribution by Home Team Win')
    plt.show()
'''
X = train_df[required_columns].to_numpy().astype(float)
Y = train_df['home_team_win'].to_numpy().astype(int)


# 測試集處理

X_test = newo[required_columns].fillna(newo[required_columns].mean()).to_numpy().astype(float)
scaler = RobustScaler()
X = scaler.fit_transform(X)  # 在訓練集上擬合
X_test = scaler.transform(X_test)  # 測試集使用相同參數
'''
print(train_df[required_columns].describe())
print(newo[required_columns].describe())

import seaborn as sns
import matplotlib.pyplot as plt

for col in required_columns:
    sns.kdeplot(train_df[col], label="Train", shade=True)
    sns.kdeplot(newo[col], label="Test", shade=True)
    plt.title(f"Feature: {col}")
    plt.legend()
    plt.show()

'''
# 設定 K-Fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []
fold_train_accuracies = []

# 建立模型函式
def create_model(input_dim):
    # Input layer
    inputs = Input(shape=(input_dim,))

    # First hidden layer with LeakyReLU activation
    x = Dense(4096, kernel_regularizer=l2(0.001))(inputs)  # 更新 units_1
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # 更新 dropout_1

    # Second hidden layer with LeakyReLU activation
    x = Dense(3584, kernel_regularizer=l2(0.001))(x)  # 更新 units_2
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # 更新 dropout_2

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model with Adam optimizer and updated learning rate
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=1.0694e-05),  # 更新 learning_rate
                  metrics=['accuracy'])

    return model
# K-Fold Cross Validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X, Y)):
    print(f"Training Fold {fold + 1}...")
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    
    model = create_model(input_dim=X_train.shape[1])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_val, Y_val), verbose=2)
    
    # 計算驗證集與訓練集準確率
    y_val_pred = (model.predict(X_val) > 0.5).astype(int)
    val_accuracy = accuracy_score(Y_val, y_val_pred)
    fold_accuracies.append(val_accuracy)
    print(f"Validation Accuracy (Fold {fold + 1}): {val_accuracy * 100:.2f}%")
    
    y_train_pred = (model.predict(X_train) > 0.5).astype(int)
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    fold_train_accuracies.append(train_accuracy)
    print(f"Training Accuracy (Fold {fold + 1}): {train_accuracy * 100:.2f}%")
    print(classification_report(Y_val, y_val_pred))
    break

# 平均準確率
mean_val_accuracy = np.mean(fold_accuracies)
mean_train_accuracy = np.mean(fold_train_accuracies)
print(f"Mean Validation Accuracy: {mean_val_accuracy * 100:.2f}%")
print(f"Mean Training Accuracy: {mean_train_accuracy * 100:.2f}%")

# 測試新檔案並輸出結果
predictions2 = (model.predict(X_test) > 0.5).astype(int)
result_df = pd.DataFrame({'id': np.arange(len(predictions2)), 'home_team_win': predictions2.flatten()})
result_df.to_csv('result_team_DNN2.csv', index=False)
print("Test predictions saved to 'result_team_only_svm.csv'.")

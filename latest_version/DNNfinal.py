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
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')
answer_df = pd.read_csv('answer.csv')  # 載入預測解答

# 取得輸入所需欄位
def get_required_columns():
    # 根據需要設定 required_columns
    required_columns = [
        # 在這裡添加所需的欄位名
    ]
    return required_columns

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
    quantiles = train_df[required_columns].quantile([0.05, 0.95])
    lower_bounds = quantiles.loc[0.05]
    upper_bounds = quantiles.loc[0.95]

train_df = preprocess_data(train_df, required_columns, lower_bounds, upper_bounds)
val_df = preprocess_data(val_df, required_columns, lower_bounds, upper_bounds)
test_df = preprocess_data(test_df, required_columns, lower_bounds, upper_bounds)

# 特徵與標籤處理
X_train = train_df[required_columns].to_numpy().astype(float)
Y_train = train_df['home_team_win'].to_numpy().astype(int)
X_val = val_df[required_columns].to_numpy().astype(float)
Y_val = val_df['home_team_win'].to_numpy().astype(int)
X_test = test_df[required_columns].to_numpy().astype(float)
Y_test = answer_df['home_team_win'].to_numpy().astype(int)  # 預測解答

# Scaling 選擇
scaler = None
scaling_choice = input("Choose scaling method (robust/standard/none): ").strip().lower()
if scaling_choice == 'robust':
    scaler = RobustScaler()
elif scaling_choice == 'standard':
    scaler = StandardScaler()

if scaler:
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

# 模型結構
def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(4096, kernel_regularizer=l2(0.001))(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(3584, kernel_regularizer=l2(0.001))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

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
    'Eout': []
}

for epoch in range(20):
    model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=2, validation_data=(X_val, Y_val))

    # Ein
    train_predictions = model.predict(X_train)
    ein = log_loss(Y_train, train_predictions)
    history['Ein'].append(ein)

    # Evalidation
    val_predictions = model.predict(X_val)
    evalidation = log_loss(Y_val, val_predictions)
    history['Evalidation'].append(evalidation)

    # Eout
    test_predictions = model.predict(X_test)
    eout = log_loss(Y_test, test_predictions)
    history['Eout'].append(eout)

print(time.time()-start)
# 畫出跑背總結果
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), history['Ein'], label='Ein', marker='o')
plt.plot(range(1, 21), history['Evalidation'], label='Evalidation', marker='o')
plt.plot(range(1, 21), history['Eout'], label='Eout', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Ein, Evalidation, and Eout over Epochs')
plt.legend()
plt.grid()
plt.savefig('training_progress.png')
plt.show()

# 測試模型
predictions = (model.predict(X_test) > 0.5).astype(int)
result_df = pd.DataFrame({'id': np.arange(len(predictions)), 'home_team_win': predictions.flatten()})
result_df.to_csv('result_team_DNN2.csv', index=False)
print("Test predictions saved to 'result_team_DNN2.csv'.")

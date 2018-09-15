"""
6.7.1 ステートフルLSTMで電力消費の予測

要は初期隠れ状態も周期性のあるデータの場合うまく活用できるよって話らしい？
ただその周期性って人間が適当に決めた固定のやつでうまくいくの？と思う
逆を言えば、ほとんどはステートフルじゃなくてもできるし、周期性がぱっとわからない時系列データに対しては使えない


ステートフルじゃない場合
Epoch 5/5
98179/98179 [==============================] - 21s 211us/step - loss: 0.0043 - mean_squared_error: 0.0043 - val_loss: 0.0038 - val_mean_squared_error: 0.0038
42077/42077 [==============================] - 2s 36us/step

MSE: 0.004, RMSE: 0.061

ステートフルな場合
Epoch 5/5
Train on 98112 samples, validate on 42048 samples
Epoch 1/1
98112/98112 [==============================] - 13s 134us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.0043 - val_mean_squared_error: 0.0043
42048/42048 [==============================] - 2s 36us/step

MSE: 0.004, RMSE: 0.066

ほとんどかわらん

"""
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import math
import os

import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = "./data"

data = np.load(os.path.join(DATA_DIR, "LD_250.npy"))

# STATELESS = True
STATELESS = False

NUM_TIMESTEPS = 20
HIDDEN_SIZE = 10
BATCH_SIZE = 96  # 24 hours (15 min intervals)
NUM_EPOCHS = 5

# 前処理（0-1に正規化)
# scale the data to be in the range (0, 1)
data = data.reshape(-1, 1)  # データを縦ベクトルに
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
data = scaler.fit_transform(data)

# 入力の系列長はNUM_TIMESTEPS(20)が良かったらしい
# 入力は(None, 20)であり、出力は(None, 1)である(None)はバッチサイズに対応
# transform to 4 inputs -> 1 label format
X = np.zeros((data.shape[0], NUM_TIMESTEPS))
Y = np.zeros((data.shape[0], 1))
for i in range(len(data) - NUM_TIMESTEPS - 1):
    # 1ずつ後ろにずらしてデータを作っている
    X[i] = data[i:i + NUM_TIMESTEPS].T  # データは縦ベクトルだったため転置して横ベクトルにしてる
    Y[i] = data[i + NUM_TIMESTEPS + 1]
#>>> X.shape
#(140256, 20)

# reshape X to three dimensions (samples, timesteps, features)
# ただし、kerasに投げるときは一工夫いる、なぜなら、(samples, timesteps, features)の形になっている必要があるからだ。
X = np.expand_dims(X, axis=2)
# >>> X.shape
#(140256, 20, 1)


# split into training and test sets (add the extra offsets so
# we can use batch size of 5)
# 後ろの3割をtestとして分割
sp = int(0.7 * len(data))
Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# モデルは1時刻先予測
if STATELESS:
    # stateless
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(NUM_TIMESTEPS, 1),
                   return_sequences=False))
    model.add(Dense(1))
else:
    # stateful
    model = Sequential()
    model.add(LSTM(HIDDEN_SIZE, stateful=True,  # ここでステートフルであると明示的に書く
                   batch_input_shape=(BATCH_SIZE, NUM_TIMESTEPS, 1),
                   return_sequences=False))
    model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam",  # 回帰問題なので二乗誤差で定義する
              metrics=["mean_squared_error"])

model.summary()

if STATELESS:
    # stateless
    model.fit(Xtrain, Ytrain, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(Xtest, Ytest),
              shuffle=False)  # 順番が大事なのでシャッフルしてはいけない
else:
    # stateful
    # need to make training and test data to multiple of BATCH_SIZE
    # 注意すべき点
    # データの周期性を反映するバッチサイズが必要(学習データも評価データもバッチサイズの倍数にする必要がある)
    # 手動で1エポックずつ回すひつようがある。なぜなら1エポックまわしたら内部状態をリセットしないと行けないから
    train_size = (Xtrain.shape[0] // BATCH_SIZE) * BATCH_SIZE
    # 後ろの数個を切り捨てることになるがしかたない
    test_size = (Xtest.shape[0] // BATCH_SIZE) * BATCH_SIZE
    Xtrain, Ytrain = Xtrain[0:train_size], Ytrain[0:train_size]
    Xtest, Ytest = Xtest[0:test_size], Ytest[0:test_size]
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    for i in range(NUM_EPOCHS):
        print("Epoch {:d}/{:d}".format(i+1, NUM_EPOCHS))
        model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=1,
                  validation_data=(Xtest, Ytest),
                  shuffle=False)
        model.reset_states()
        #　ここでステータスをリセット！

score, _ = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
rmse = math.sqrt(score)
print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))

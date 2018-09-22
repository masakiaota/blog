"""
gruを用いたオートエンコーダー

オートエンコーダーで電力消費の再現
"""
import math
import os

import numpy as np
from keras.layers import Dense, RepeatVector
from keras.layers import GRU
# from keras.layers import CuDNNGRU as GRU #GPU用
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard, EarlyStopping

# データの準備


def preaprete_data():
    """
    戻り値
      規格化された
      X_train, X_test
    """
    DATA_DIR = "./data/"
    data = np.load(DATA_DIR + "LD_250.npy")
    global NUM_TIMESTEPS
    NUM_TIMESTEPS = 192  # 2日間に対応

    # 正規化
    data = data.reshape(-1, 1)  # データを縦ベクトルに
    scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    data = scaler.fit_transform(data)
    # 本当はテストデータもひっくるめて正規化してしまうのは少しズルなのだが横着

    X = np.zeros((data.shape[0]-NUM_TIMESTEPS, NUM_TIMESTEPS))
    for i in range(data.shape[0] - NUM_TIMESTEPS):
        # 1ずつ後ろにずらしてデータを作っている
        X[i] = data[i:i + NUM_TIMESTEPS].T  # データは縦ベクトルだったため転置して横ベクトルにしてる
    #(140064, 192)(サンプル数, 時間幅)

    # ただし、kerasに投げるときは一工夫いる、なぜなら、(samples, timesteps, features)の形になっている必要があるからだ。
    X = np.expand_dims(X, axis=2)
    #(140064, 192, 1)(サンプル数, 時間幅, 次元数)

    # データの後半の1割をtestデータとして分割
    sep = int(0.9 * X.shape[0])
    X_train, X_test = X[:sep], X[sep:]
    print("train", X_train.shape, "test", X_test.shape)
    np.save("./data/X_train.npy", X_train)
    np.save("./data/X_test.npy", X_test)
    return X_train, X_test


def seq_autoencoder():
    """
    入力
    ↓
    GRU(encoder)
    ↓
    内部状態 (latent)
    ↓
    GRU(decoder)
    ↓
    全結合層(出力)

    のようなシンプルなネットワーク

    戻り値
     model
    """
    LATENT_DIM = 20
    model = Sequential()
    # (None(batchsize), NUM_TIMESTEMPS, 1)
    model.add(GRU(LATENT_DIM, input_shape=(NUM_TIMESTEPS, 1)))
    # (None, LATENT_DIM)
    model.add(RepeatVector(NUM_TIMESTEPS))
    # (None, NUM_TIMESTEPS, LATENT_DIM)
    model.add(GRU(LATENT_DIM, return_sequences=True))
    model.add(Dense(1))  # 本当はtimedistributedのラッパーで包まないと行けないらしいが、しなくてもできてしまった。

    model.summary()

    return model


if __name__ == "__main__":

    print("="*20+"preparating the data..."+"="*20)

    X_train, X_test = preaprete_data()
    print("="*20+"summary of this model"+"="*20)
    seq_ae = seq_autoencoder()

    seq_ae.compile(loss="mean_squared_error", optimizer="adam",
                   metrics=["mean_squared_error"])
    # 学習
    seq_ae.fit(X_train, X_train,
               epochs=30,
               batch_size=128,
               shuffle=False,
               validation_split=0.1,
               callbacks=[TensorBoard(log_dir="./gru_ae/"), EarlyStopping(patience=2)])
    seq_ae.save('./gru_ae/gru_ae.h5')

    # 推論
    X_pred = seq_ae.predict(X_test)

    import matplotlib.pyplot as plt
    # データをプロット
    for true, pred in zip(X_test[::1000], X_pred[::1000]):
        plt.plot(range(true.shape[0]), true)
        plt.plot(range(pred.shape[0]), pred)
        plt.ylabel("electricity consumption")
        plt.xlabel("time (1pt = 15 mins)")
        plt.ylim([0.2, 0.8])
        plt.show()


"""
====================preparating the data...====================
train (126057, 192, 1) test (14007, 192, 1)
====================summary of this model====================
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cu_dnngru_9 (CuDNNGRU)       (None, 20)                1380      
_________________________________________________________________
repeat_vector_6 (RepeatVecto (None, 192, 20)           0         
_________________________________________________________________
cu_dnngru_10 (CuDNNGRU)      (None, 192, 20)           2520      
_________________________________________________________________
dense_1 (Dense)              (None, 192, 1)            21        
=================================================================
Total params: 3,921
Trainable params: 3,921
Non-trainable params: 0
_________________________________________________________________
Train on 113451 samples, validate on 12606 samples
Epoch 1/30
113451/113451 [==============================] - 53s 464us/step - loss: 0.0301 - mean_squared_error: 0.0301 - val_loss: 0.0227 - val_mean_squared_error: 0.0227
Epoch 2/30
113451/113451 [==============================] - 40s 356us/step - loss: 0.0247 - mean_squared_error: 0.0247 - val_loss: 0.0224 - val_mean_squared_error: 0.0224
Epoch 3/30
113451/113451 [==============================] - 40s 356us/step - loss: 0.0245 - mean_squared_error: 0.0245 - val_loss: 0.0219 - val_mean_squared_error: 0.0219
Epoch 4/30
113451/113451 [==============================] - 40s 357us/step - loss: 0.0241 - mean_squared_error: 0.0241 - val_loss: 0.0213 - val_mean_squared_error: 0.0213
Epoch 5/30
113451/113451 [==============================] - 40s 356us/step - loss: 0.0229 - mean_squared_error: 0.0229 - val_loss: 0.0173 - val_mean_squared_error: 0.0173
Epoch 6/30
113451/113451 [==============================] - 40s 353us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0054 - val_mean_squared_error: 0.0054
Epoch 7/30
113451/113451 [==============================] - 40s 352us/step - loss: 0.0057 - mean_squared_error: 0.0057 - val_loss: 0.0046 - val_mean_squared_error: 0.0046
Epoch 8/30
113451/113451 [==============================] - 40s 350us/step - loss: 0.0053 - mean_squared_error: 0.0053 - val_loss: 0.0043 - val_mean_squared_error: 0.0043
Epoch 9/30
113451/113451 [==============================] - 40s 350us/step - loss: 0.0051 - mean_squared_error: 0.0051 - val_loss: 0.0042 - val_mean_squared_error: 0.0042
Epoch 10/30
113451/113451 [==============================] - 40s 350us/step - loss: 0.0049 - mean_squared_error: 0.0049 - val_loss: 0.0041 - val_mean_squared_error: 0.0041
Epoch 11/30
113451/113451 [==============================] - 39s 346us/step - loss: 0.0047 - mean_squared_error: 0.0047 - val_loss: 0.0040 - val_mean_squared_error: 0.0040
Epoch 12/30
113451/113451 [==============================] - 40s 350us/step - loss: 0.0045 - mean_squared_error: 0.0045 - val_loss: 0.0039 - val_mean_squared_error: 0.0039
Epoch 13/30
113451/113451 [==============================] - 39s 347us/step - loss: 0.0042 - mean_squared_error: 0.0042 - val_loss: 0.0034 - val_mean_squared_error: 0.0034
Epoch 14/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0040 - mean_squared_error: 0.0040 - val_loss: 0.0028 - val_mean_squared_error: 0.0028
Epoch 15/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0039 - mean_squared_error: 0.0039 - val_loss: 0.0027 - val_mean_squared_error: 0.0027
Epoch 16/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0026 - val_mean_squared_error: 0.0026
Epoch 17/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0038 - mean_squared_error: 0.0038 - val_loss: 0.0026 - val_mean_squared_error: 0.0026
Epoch 18/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0026 - val_mean_squared_error: 0.0026
Epoch 19/30
113451/113451 [==============================] - 39s 347us/step - loss: 0.0037 - mean_squared_error: 0.0037 - val_loss: 0.0026 - val_mean_squared_error: 0.0026
Epoch 20/30
113451/113451 [==============================] - 40s 351us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0026 - val_mean_squared_error: 0.0026
Epoch 21/30
113451/113451 [==============================] - 39s 346us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 22/30
113451/113451 [==============================] - 39s 348us/step - loss: 0.0036 - mean_squared_error: 0.0036 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 23/30
113451/113451 [==============================] - 39s 347us/step - loss: 0.0035 - mean_squared_error: 0.0035 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 24/30
113451/113451 [==============================] - 39s 347us/step - loss: 0.0035 - mean_squared_error: 0.0035 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 25/30
113451/113451 [==============================] - 39s 347us/step - loss: 0.0035 - mean_squared_error: 0.0035 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 26/30
113451/113451 [==============================] - 39s 345us/step - loss: 0.0035 - mean_squared_error: 0.0035 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 27/30
113451/113451 [==============================] - 39s 346us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 28/30
113451/113451 [==============================] - 40s 351us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 29/30
113451/113451 [==============================] - 39s 346us/step - loss: 0.0034 - mean_squared_error: 0.0034 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
Epoch 30/30
113451/113451 [==============================] - 39s 346us/step - loss: 0.0033 - mean_squared_error: 0.0033 - val_loss: 0.0025 - val_mean_squared_error: 0.0025
"""

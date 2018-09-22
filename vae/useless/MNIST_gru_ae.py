"""
gruを用いたオートエンコーダー

オートエンコーダーで電力消費の再現
"""
import math
import os

import numpy as np
from keras.layers import Dense, RepeatVector
# from keras.layers import GRU
from keras.layers import CuDNNGRU as GRU  # GPU用
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import adam

# データの準備


def preaprete_data():
    """
    戻り値
      規格化された
      x_train, x_test
    """
    from keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = x_train/255, x_test/255
    # >>> x_train.shape, x_test.shape
    # ((60000, 28, 28), (10000, 28, 28))
    # これを(None, NUM_TIMESTEPS, NUM_INPUT_DIM)と対応させる
    global NUM_TIMESTEPS, NUM_INPUT_DIM
    _, NUM_TIMESTEPS, NUM_INPUT_DIM = x_train.shape
    return x_train, x_test


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
    # (None(batchsize), NUM_TIMESTEMPS, NUM_INPUT_DIM)
    model.add(GRU(
        LATENT_DIM,
        input_shape=(NUM_TIMESTEPS, NUM_INPUT_DIM),
    ))
    # (None, LATENT_DIM)
    model.add(RepeatVector(NUM_TIMESTEPS))
    # (None, NUM_TIMESTEPS, LATENT_DIM)
    model.add(GRU(LATENT_DIM, return_sequences=True,
                  ))
    # 本当はtimedistributedのラッパーで包まないと行けないらしいが、しなくてもできてしまった。
    model.add(Dense(NUM_INPUT_DIM))

    model.summary()

    return model


if __name__ == "__main__":

    print("="*20+"preparating the data..."+"="*20)

    x_train, x_test = preaprete_data()
    print("="*20+"summary of this model"+"="*20)
    seq_ae = seq_autoencoder()

    seq_ae.compile(loss="mean_squared_error", optimizer="nadam",
                   )
    # 学習
    seq_ae.fit(x_train, x_train,
               epochs=100,
               batch_size=64,
               shuffle=True,
               validation_split=0.1,
               callbacks=[TensorBoard(log_dir="./MNIST_gru_ae/"), EarlyStopping(patience=3)])
    seq_ae.save('./MNIST_gru_ae/MNIST_gru_ae.h5')

    # 推論
    x_pred = seq_ae.predict(x_test)

    import matplotlib.pyplot as plt
    # データをプロット
    for true, pred in zip(x_test[::1000], x_pred[::1000]):
        plt.matshow(true)

        plt.matshow(pred)
        plt.show()

"""
====================preparating the data...====================
====================summary of this model====================
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cu_dnngru_13 (CuDNNGRU)      (None, 20)                3000      
_________________________________________________________________
repeat_vector_13 (RepeatVect (None, 28, 20)            0         
_________________________________________________________________
cu_dnngru_14 (CuDNNGRU)      (None, 28, 20)            2520      
_________________________________________________________________
dense_13 (Dense)             (None, 28, 28)            588       
=================================================================
Total params: 6,108
Trainable params: 6,108
Non-trainable params: 0
_________________________________________________________________
Train on 54000 samples, validate on 6000 samples
Epoch 1/100
54000/54000 [==============================] - 16s 301us/step - loss: 0.0599 - val_loss: 0.0499
Epoch 2/100
54000/54000 [==============================] - 14s 268us/step - loss: 0.0452 - val_loss: 0.0407
Epoch 3/100
54000/54000 [==============================] - 14s 263us/step - loss: 0.0387 - val_loss: 0.0373
Epoch 4/100
54000/54000 [==============================] - 14s 267us/step - loss: 0.0348 - val_loss: 0.0329
Epoch 5/100
54000/54000 [==============================] - 15s 282us/step - loss: 0.0320 - val_loss: 0.0304
Epoch 6/100
54000/54000 [==============================] - 15s 280us/step - loss: 0.0300 - val_loss: 0.0289
Epoch 7/100
54000/54000 [==============================] - 15s 278us/step - loss: 0.0286 - val_loss: 0.0277
Epoch 8/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0276 - val_loss: 0.0268
Epoch 9/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0268 - val_loss: 0.0263
Epoch 10/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0262 - val_loss: 0.0254
Epoch 11/100
54000/54000 [==============================] - 15s 278us/step - loss: 0.0256 - val_loss: 0.0250
Epoch 12/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0251 - val_loss: 0.0245
Epoch 13/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0247 - val_loss: 0.0243
Epoch 14/100
54000/54000 [==============================] - 15s 281us/step - loss: 0.0243 - val_loss: 0.0239
Epoch 15/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0239 - val_loss: 0.0237
Epoch 16/100
54000/54000 [==============================] - 15s 269us/step - loss: 0.0236 - val_loss: 0.0230
Epoch 17/100
54000/54000 [==============================] - 15s 272us/step - loss: 0.0232 - val_loss: 0.0229
Epoch 18/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0229 - val_loss: 0.0233
Epoch 19/100
54000/54000 [==============================] - 15s 272us/step - loss: 0.0227 - val_loss: 0.0222
Epoch 20/100
54000/54000 [==============================] - 14s 263us/step - loss: 0.0224 - val_loss: 0.0221
Epoch 21/100
54000/54000 [==============================] - 14s 263us/step - loss: 0.0222 - val_loss: 0.0219
Epoch 22/100
54000/54000 [==============================] - 16s 295us/step - loss: 0.0219 - val_loss: 0.0215
Epoch 23/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0217 - val_loss: 0.0214
Epoch 24/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0215 - val_loss: 0.0214
Epoch 25/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0213 - val_loss: 0.0213
Epoch 26/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0212 - val_loss: 0.0208
Epoch 27/100
54000/54000 [==============================] - 16s 291us/step - loss: 0.0210 - val_loss: 0.0207
Epoch 28/100
54000/54000 [==============================] - 15s 278us/step - loss: 0.0208 - val_loss: 0.0205
Epoch 29/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0207 - val_loss: 0.0205
Epoch 30/100
54000/54000 [==============================] - 15s 278us/step - loss: 0.0206 - val_loss: 0.0206
Epoch 31/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0204 - val_loss: 0.0202
Epoch 32/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0203 - val_loss: 0.0201
Epoch 33/100
54000/54000 [==============================] - 15s 273us/step - loss: 0.0202 - val_loss: 0.0200
Epoch 34/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0201 - val_loss: 0.0200
Epoch 35/100
54000/54000 [==============================] - 15s 279us/step - loss: 0.0200 - val_loss: 0.0197
Epoch 36/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0199 - val_loss: 0.0198
Epoch 37/100
54000/54000 [==============================] - 15s 276us/step - loss: 0.0198 - val_loss: 0.0197
Epoch 38/100
54000/54000 [==============================] - 15s 272us/step - loss: 0.0198 - val_loss: 0.0197
Epoch 39/100
54000/54000 [==============================] - 15s 274us/step - loss: 0.0197 - val_loss: 0.0196
Epoch 40/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0196 - val_loss: 0.0194
Epoch 41/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0196 - val_loss: 0.0193
Epoch 42/100
54000/54000 [==============================] - 15s 272us/step - loss: 0.0195 - val_loss: 0.0193
Epoch 43/100
54000/54000 [==============================] - 15s 270us/step - loss: 0.0194 - val_loss: 0.0197
Epoch 44/100
54000/54000 [==============================] - 14s 268us/step - loss: 0.0194 - val_loss: 0.0191
Epoch 45/100
54000/54000 [==============================] - 15s 269us/step - loss: 0.0193 - val_loss: 0.0193
Epoch 46/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0193 - val_loss: 0.0190
Epoch 47/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0192 - val_loss: 0.0191
Epoch 48/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0192 - val_loss: 0.0190
Epoch 49/100
54000/54000 [==============================] - 15s 277us/step - loss: 0.0191 - val_loss: 0.0190
Epoch 50/100
54000/54000 [==============================] - 15s 272us/step - loss: 0.0191 - val_loss: 0.0189
Epoch 51/100
54000/54000 [==============================] - 15s 273us/step - loss: 0.0190 - val_loss: 0.0188
Epoch 52/100
54000/54000 [==============================] - 15s 274us/step - loss: 0.0190 - val_loss: 0.0187
Epoch 53/100
54000/54000 [==============================] - 15s 274us/step - loss: 0.0189 - val_loss: 0.0188
Epoch 54/100
54000/54000 [==============================] - 15s 275us/step - loss: 0.0189 - val_loss: 0.0189
Epoch 55/100
54000/54000 [==============================] - 15s 278us/step - loss: 0.0189 - val_loss: 0.0192
"""

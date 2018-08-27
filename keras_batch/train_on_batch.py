import os
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def lenet(input_shape, num_classes):
    """
    http://tecmemo.wpblog.jp/wp-content/uploads/2017/03/dl_lenet-01.png この表を参考に一部活性化関数を変更してLenetを定義
    """
    model = Sequential()

    # フィルターを6枚用意, 小窓のサイズ5×5, paddingによって入力と出力の画像サイズは同じ
    model.add(Conv2D(
        20, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu",
    ))
    # 2, 2でマックスプーリング
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 再度畳み込み、深い層ほどフィルターを増やすのはテクニック
    model.add(Conv2D(50, kernel_size=5, padding="same",
                     activation="relu", ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten()はマトリックスを1次元ベクトルに変換する層
    # FCにつなぐために必要
    model.add(Flatten())
    # model.add(Dense(120, activation="relu", init="he_normal"))
    model.add(Dense(500, activation="relu", init="he_normal"))
    model.add(Dense(num_classes, init="he_normal"))
    model.add(Activation("softmax"))
    model.summary()
    return model


# acquire the .csv name
TRAINS = list(Path("./data/train/").glob("*.csv"))
y = pd.read_csv("./data/target.csv", header=None).iloc[:, 1].values
# split test
X_train, X_test, y_train, y_test = train_test_split(
    TRAINS, y, test_size=0.1, random_state=42)
# onehot
Y_train, Y_test = keras.utils.to_categorical(
    y_train, 10), keras.utils.to_categorical(y_test, 10)


def get_batch(batch_size):
    """
    batchを取得する関数
    """
    global X_train, Y_train
    SIZE = len(X_train)
    # n_batchs
    n_batchs = SIZE//batch_size
    # for でyield
    i = 0
    while (i < n_batchs):
        print("doing", i, "/", n_batchs, end="")
        Y_batch = Y_train[(i * n_batchs):(i * n_batchs + batch_size)]
        X_batch_name = X_train[(i * n_batchs):(i * n_batchs + batch_size)]
        X_batch = np.array([np.loadtxt(file)
                            for file in X_batch_name]).reshape(-1, 28, 28, 1)
        print(" X_batch.shape",X_batch.shape)
        # これで(batch_size, 28, 28, 1)のtrainのテンソルが作られる
        i += 1
        yield X_batch, Y_batch


# またtestに関してはデータが十分少ない(メモリに乗る)と仮定して、取得しておく
print("loading X_test...")
X_test = np.array([np.loadtxt(file)
                   for file in X_test]).reshape(-1, 28, 28, 1)

model = lenet((28, 28, 1), 10)
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(),
              metrics=["accuracy"])
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    print("=" * 50)
    print(epoch, "/", N_EPOCHS)
    acc = []
    for X_batch, Y_batch in get_batch(256):
        model.train_on_batch(X_batch, Y_batch)
        score = model.evaluate(X_batch, Y_batch)
        print("batch accuracy:", score[1])
        acc.append(score[1])
    print("Train accuracy", np.mean(acc))
    score = model.evaluate(X_test, Y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


"""
result
Train accuracy 0.9967049319727891
4200/4200 [==============================] - 2s 506us/step
Test loss: 0.06669212961836907
Test accuracy: 0.9816666666666667
"""

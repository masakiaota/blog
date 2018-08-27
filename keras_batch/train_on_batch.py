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
        print("doing", i, "/", n_batchs)
        Y_batch = Y_train[(i * n_batchs):(i * n_batchs + batch_size)]
        X_batch_name = X_train[(i * n_batchs):(i * n_batchs + batch_size)]
        X_batch = np.array([np.loadtxt(file)
                            for file in X_batch_name]).reshape(batch_size, 28, 28, 1)
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
    for X_batch, Y_batch in get_batch(512):
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

Train accuracy 0.9124304366438356
4200/4200 [==============================] - 2s 554us/step
Test loss: 0.16485723899943489
Test accuracy: 0.9530952380952381
==================================================
1 / 5
doing 0 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.9609375
doing 1 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 0.96484375
doing 2 / 73
512/512 [==============================] - 0s 541us/step
batch accuracy: 0.96484375
doing 3 / 73
512/512 [==============================] - 0s 556us/step
batch accuracy: 0.966796875
doing 4 / 73
512/512 [==============================] - 0s 538us/step
batch accuracy: 0.962890625
doing 5 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 0.962890625
doing 6 / 73
512/512 [==============================] - 0s 614us/step
batch accuracy: 0.970703125
doing 7 / 73
512/512 [==============================] - 0s 566us/step
batch accuracy: 0.96875
doing 8 / 73
512/512 [==============================] - 0s 579us/step
batch accuracy: 0.970703125
doing 9 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 0.98046875
doing 10 / 73
512/512 [==============================] - 0s 633us/step
batch accuracy: 0.970703125
doing 11 / 73
512/512 [==============================] - 0s 540us/step
batch accuracy: 0.970703125
doing 12 / 73
512/512 [==============================] - 0s 601us/step
batch accuracy: 0.97265625
doing 13 / 73
512/512 [==============================] - 0s 566us/step
batch accuracy: 0.97265625
doing 14 / 73
512/512 [==============================] - 0s 566us/step
batch accuracy: 0.974609375
doing 15 / 73
512/512 [==============================] - 0s 584us/step
batch accuracy: 0.984375
doing 16 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 0.984375
doing 17 / 73
512/512 [==============================] - 0s 582us/step
batch accuracy: 0.984375
doing 18 / 73
512/512 [==============================] - 0s 581us/step
batch accuracy: 0.978515625
doing 19 / 73
512/512 [==============================] - 0s 612us/step
batch accuracy: 0.9765625
doing 20 / 73
512/512 [==============================] - 0s 564us/step
batch accuracy: 0.98046875
doing 21 / 73
512/512 [==============================] - 0s 536us/step
batch accuracy: 0.98046875
doing 22 / 73
512/512 [==============================] - 0s 562us/step
batch accuracy: 0.978515625
doing 23 / 73
512/512 [==============================] - 0s 570us/step
batch accuracy: 0.98046875
doing 24 / 73
512/512 [==============================] - 0s 638us/step
batch accuracy: 0.974609375
doing 25 / 73
512/512 [==============================] - 0s 585us/step
batch accuracy: 0.9765625
doing 26 / 73
512/512 [==============================] - 0s 579us/step
batch accuracy: 0.978515625
doing 27 / 73
512/512 [==============================] - 0s 624us/step
batch accuracy: 0.98046875
doing 28 / 73
512/512 [==============================] - 0s 552us/step
batch accuracy: 0.982421875
doing 29 / 73
512/512 [==============================] - 0s 743us/step
batch accuracy: 0.990234375
doing 30 / 73
512/512 [==============================] - 0s 806us/step
batch accuracy: 0.994140625
doing 31 / 73
512/512 [==============================] - 0s 589us/step
batch accuracy: 0.990234375
doing 32 / 73
512/512 [==============================] - 0s 605us/step
batch accuracy: 0.984375
doing 33 / 73
512/512 [==============================] - 0s 622us/step
batch accuracy: 0.9921875
doing 34 / 73
512/512 [==============================] - 0s 624us/step
batch accuracy: 0.986328125
doing 35 / 73
512/512 [==============================] - 0s 693us/step
batch accuracy: 0.98828125
doing 36 / 73
512/512 [==============================] - 0s 585us/step
batch accuracy: 0.984375
doing 37 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 0.986328125
doing 38 / 73
512/512 [==============================] - 0s 559us/step
batch accuracy: 0.98828125
doing 39 / 73
512/512 [==============================] - 0s 628us/step
batch accuracy: 0.984375
doing 40 / 73
512/512 [==============================] - 0s 577us/step
batch accuracy: 0.984375
doing 41 / 73
512/512 [==============================] - 0s 558us/step
batch accuracy: 0.994140625
doing 42 / 73
512/512 [==============================] - 0s 641us/step
batch accuracy: 0.99609375
doing 43 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 0.99609375
doing 44 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 0.9921875
doing 45 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 0.990234375
doing 46 / 73
512/512 [==============================] - 0s 547us/step
batch accuracy: 0.98828125
doing 47 / 73
512/512 [==============================] - 0s 549us/step
batch accuracy: 0.994140625
doing 48 / 73
512/512 [==============================] - 0s 642us/step
batch accuracy: 0.994140625
doing 49 / 73
512/512 [==============================] - 0s 597us/step
batch accuracy: 0.986328125
doing 50 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.984375
doing 51 / 73
512/512 [==============================] - 0s 704us/step
batch accuracy: 0.98828125
doing 52 / 73
512/512 [==============================] - 0s 536us/step
batch accuracy: 0.98828125
doing 53 / 73
512/512 [==============================] - 0s 580us/step
batch accuracy: 0.984375
doing 54 / 73
512/512 [==============================] - 0s 542us/step
batch accuracy: 0.98828125
doing 55 / 73
512/512 [==============================] - 0s 570us/step
batch accuracy: 0.986328125
doing 56 / 73
512/512 [==============================] - 0s 622us/step
batch accuracy: 0.986328125
doing 57 / 73
512/512 [==============================] - 0s 562us/step
batch accuracy: 0.98828125
doing 58 / 73
512/512 [==============================] - 0s 606us/step
batch accuracy: 0.98828125
doing 59 / 73
512/512 [==============================] - 0s 606us/step
batch accuracy: 0.9921875
doing 60 / 73
512/512 [==============================] - 0s 680us/step
batch accuracy: 0.99609375
doing 61 / 73
512/512 [==============================] - 0s 649us/step
batch accuracy: 0.994140625
doing 62 / 73
512/512 [==============================] - 0s 645us/step
batch accuracy: 0.9921875
doing 63 / 73
512/512 [==============================] - 0s 607us/step
batch accuracy: 0.9921875
doing 64 / 73
512/512 [==============================] - 0s 614us/step
batch accuracy: 0.9921875
doing 65 / 73
512/512 [==============================] - 0s 607us/step
batch accuracy: 0.994140625
doing 66 / 73
512/512 [==============================] - 0s 543us/step
batch accuracy: 0.99609375
doing 67 / 73
512/512 [==============================] - 0s 609us/step
batch accuracy: 0.994140625
doing 68 / 73
512/512 [==============================] - 0s 594us/step
batch accuracy: 0.9921875
doing 69 / 73
512/512 [==============================] - 0s 631us/step
batch accuracy: 0.99609375
doing 70 / 73
512/512 [==============================] - 0s 658us/step
batch accuracy: 1.0
doing 71 / 73
512/512 [==============================] - 0s 615us/step
batch accuracy: 1.0
doing 72 / 73
512/512 [==============================] - 0s 568us/step
batch accuracy: 0.998046875
Train accuracy 0.984294734589041
4200/4200 [==============================] - 3s 596us/step
Test loss: 0.11887204998749352
Test accuracy: 0.9676190476190476
==================================================
2 / 5
doing 0 / 73
512/512 [==============================] - 0s 528us/step
batch accuracy: 0.982421875
doing 1 / 73
512/512 [==============================] - 0s 676us/step
batch accuracy: 0.98046875
doing 2 / 73
512/512 [==============================] - 0s 534us/step
batch accuracy: 0.9765625
doing 3 / 73
512/512 [==============================] - 0s 540us/step
batch accuracy: 0.98046875
doing 4 / 73
512/512 [==============================] - 0s 561us/step
batch accuracy: 0.984375
doing 5 / 73
512/512 [==============================] - 0s 607us/step
batch accuracy: 0.98828125
doing 6 / 73
512/512 [==============================] - 0s 609us/step
batch accuracy: 0.98828125
doing 7 / 73
512/512 [==============================] - 0s 585us/step
batch accuracy: 0.99609375
doing 8 / 73
512/512 [==============================] - 0s 596us/step
batch accuracy: 0.9921875
doing 9 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 0.9921875
doing 10 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.9921875
doing 11 / 73
512/512 [==============================] - 0s 538us/step
batch accuracy: 0.986328125
doing 12 / 73
512/512 [==============================] - 0s 606us/step
batch accuracy: 0.9921875
doing 13 / 73
512/512 [==============================] - 0s 598us/step
batch accuracy: 0.998046875
doing 14 / 73
512/512 [==============================] - 0s 578us/step
batch accuracy: 1.0
doing 15 / 73
512/512 [==============================] - 0s 617us/step
batch accuracy: 0.9921875
doing 16 / 73
512/512 [==============================] - 0s 564us/step
batch accuracy: 0.9921875
doing 17 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 0.9921875
doing 18 / 73
512/512 [==============================] - 0s 657us/step
batch accuracy: 0.990234375
doing 19 / 73
512/512 [==============================] - 0s 601us/step
batch accuracy: 0.994140625
doing 20 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 0.994140625
doing 21 / 73
512/512 [==============================] - 0s 647us/step
batch accuracy: 0.9921875
doing 22 / 73
512/512 [==============================] - 0s 543us/step
batch accuracy: 0.990234375
doing 23 / 73
512/512 [==============================] - 0s 576us/step
batch accuracy: 0.99609375
doing 24 / 73
512/512 [==============================] - 0s 590us/step
batch accuracy: 0.99609375
doing 25 / 73
512/512 [==============================] - 0s 588us/step
batch accuracy: 0.994140625
doing 26 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.994140625
doing 27 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.99609375
doing 28 / 73
512/512 [==============================] - 0s 574us/step
batch accuracy: 0.998046875
doing 29 / 73
512/512 [==============================] - 0s 563us/step
batch accuracy: 0.99609375
doing 30 / 73
512/512 [==============================] - 0s 569us/step
batch accuracy: 0.99609375
doing 31 / 73
512/512 [==============================] - 0s 593us/step
batch accuracy: 1.0
doing 32 / 73
512/512 [==============================] - 0s 680us/step
batch accuracy: 0.99609375
doing 33 / 73
512/512 [==============================] - 0s 545us/step
batch accuracy: 0.99609375
doing 34 / 73
512/512 [==============================] - 0s 575us/step
batch accuracy: 0.99609375
doing 35 / 73
512/512 [==============================] - 0s 696us/step
batch accuracy: 0.99609375
doing 36 / 73
512/512 [==============================] - 0s 589us/step
batch accuracy: 0.994140625
doing 37 / 73
512/512 [==============================] - 0s 597us/step
batch accuracy: 0.99609375
doing 38 / 73
512/512 [==============================] - 0s 560us/step
batch accuracy: 0.994140625
doing 39 / 73
512/512 [==============================] - 0s 610us/step
batch accuracy: 0.99609375
doing 40 / 73
512/512 [==============================] - 0s 800us/step
batch accuracy: 0.998046875
doing 41 / 73
512/512 [==============================] - 0s 529us/step
batch accuracy: 0.998046875
doing 42 / 73
512/512 [==============================] - 0s 551us/step
batch accuracy: 1.0
doing 43 / 73
512/512 [==============================] - 0s 590us/step
batch accuracy: 0.998046875
doing 44 / 73
512/512 [==============================] - 0s 550us/step
batch accuracy: 0.998046875
doing 45 / 73
512/512 [==============================] - 0s 588us/step
batch accuracy: 0.99609375
doing 46 / 73
512/512 [==============================] - 0s 560us/step
batch accuracy: 0.99609375
doing 47 / 73
512/512 [==============================] - 0s 649us/step
batch accuracy: 1.0
doing 48 / 73
512/512 [==============================] - 0s 726us/step
batch accuracy: 0.998046875
doing 49 / 73
512/512 [==============================] - 0s 652us/step
batch accuracy: 0.998046875
doing 50 / 73
512/512 [==============================] - 0s 571us/step
batch accuracy: 0.998046875
doing 51 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 1.0
doing 52 / 73
512/512 [==============================] - 0s 559us/step
batch accuracy: 0.998046875
doing 53 / 73
512/512 [==============================] - 0s 639us/step
batch accuracy: 0.998046875
doing 54 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 0.99609375
doing 55 / 73
512/512 [==============================] - 0s 583us/step
batch accuracy: 0.99609375
doing 56 / 73
512/512 [==============================] - 0s 648us/step
batch accuracy: 0.998046875
doing 57 / 73
512/512 [==============================] - 0s 601us/step
batch accuracy: 1.0
doing 58 / 73
512/512 [==============================] - 0s 562us/step
batch accuracy: 0.99609375
doing 59 / 73
512/512 [==============================] - 0s 556us/step
batch accuracy: 0.99609375
doing 60 / 73
512/512 [==============================] - 0s 551us/step
batch accuracy: 0.998046875
doing 61 / 73
512/512 [==============================] - 0s 555us/step
batch accuracy: 1.0
doing 62 / 73
512/512 [==============================] - 0s 582us/step
batch accuracy: 0.998046875
doing 63 / 73
512/512 [==============================] - 0s 623us/step
batch accuracy: 0.99609375
doing 64 / 73
512/512 [==============================] - 0s 540us/step
batch accuracy: 0.99609375
doing 65 / 73
512/512 [==============================] - 0s 633us/step
batch accuracy: 1.0
doing 66 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 0.998046875
doing 67 / 73
512/512 [==============================] - 0s 595us/step
batch accuracy: 1.0
doing 68 / 73
512/512 [==============================] - 0s 685us/step
batch accuracy: 1.0
doing 69 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 1.0
doing 70 / 73
512/512 [==============================] - 0s 562us/step
batch accuracy: 1.0
doing 71 / 73
512/512 [==============================] - 0s 602us/step
batch accuracy: 1.0
doing 72 / 73
512/512 [==============================] - 0s 747us/step
batch accuracy: 0.998046875
Train accuracy 0.9950770547945206
4200/4200 [==============================] - 3s 630us/step
Test loss: 0.11261171695709761
Test accuracy: 0.9742857142857143
==================================================
3 / 5
doing 0 / 73
512/512 [==============================] - 0s 552us/step
batch accuracy: 0.98828125
doing 1 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 0.98828125
doing 2 / 73
512/512 [==============================] - 0s 606us/step
batch accuracy: 0.98828125
doing 3 / 73
512/512 [==============================] - 0s 542us/step
batch accuracy: 0.99609375
doing 4 / 73
512/512 [==============================] - 0s 597us/step
batch accuracy: 1.0
doing 5 / 73
512/512 [==============================] - 0s 541us/step
batch accuracy: 0.998046875
doing 6 / 73
512/512 [==============================] - 0s 561us/step
batch accuracy: 0.998046875
doing 7 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 0.998046875
doing 8 / 73
512/512 [==============================] - 0s 743us/step
batch accuracy: 0.998046875
doing 9 / 73
512/512 [==============================] - 0s 599us/step
batch accuracy: 0.998046875
doing 10 / 73
512/512 [==============================] - 0s 538us/step
batch accuracy: 0.99609375
doing 11 / 73
512/512 [==============================] - 0s 639us/step
batch accuracy: 1.0
doing 12 / 73
512/512 [==============================] - 0s 658us/step
batch accuracy: 1.0
doing 13 / 73
512/512 [==============================] - 0s 583us/step
batch accuracy: 0.998046875
doing 14 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 0.99609375
doing 15 / 73
512/512 [==============================] - 0s 598us/step
batch accuracy: 0.998046875
doing 16 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 0.998046875
doing 17 / 73
512/512 [==============================] - 0s 577us/step
batch accuracy: 0.99609375
doing 18 / 73
512/512 [==============================] - 0s 605us/step
batch accuracy: 0.998046875
doing 19 / 73
512/512 [==============================] - 0s 533us/step
batch accuracy: 0.998046875
doing 20 / 73
512/512 [==============================] - 0s 609us/step
batch accuracy: 0.998046875
doing 21 / 73
512/512 [==============================] - 0s 560us/step
batch accuracy: 1.0
doing 22 / 73
512/512 [==============================] - 0s 561us/step
batch accuracy: 0.994140625
doing 23 / 73
512/512 [==============================] - 0s 562us/step
batch accuracy: 0.994140625
doing 24 / 73
512/512 [==============================] - 0s 549us/step
batch accuracy: 1.0
doing 25 / 73
512/512 [==============================] - 0s 640us/step
batch accuracy: 0.99609375
doing 26 / 73
512/512 [==============================] - 0s 644us/step
batch accuracy: 0.99609375
doing 27 / 73
512/512 [==============================] - 0s 554us/step
batch accuracy: 0.99609375
doing 28 / 73
512/512 [==============================] - 0s 668us/step
batch accuracy: 0.998046875
doing 29 / 73
512/512 [==============================] - 0s 735us/step
batch accuracy: 1.0
doing 30 / 73
512/512 [==============================] - 0s 539us/step
batch accuracy: 1.0
doing 31 / 73
512/512 [==============================] - 0s 774us/step
batch accuracy: 1.0
doing 32 / 73
512/512 [==============================] - 0s 543us/step
batch accuracy: 0.9921875
doing 33 / 73
512/512 [==============================] - 0s 720us/step
batch accuracy: 0.998046875
doing 34 / 73
512/512 [==============================] - 0s 681us/step
batch accuracy: 0.99609375
doing 35 / 73
512/512 [==============================] - 0s 567us/step
batch accuracy: 0.99609375
doing 36 / 73
512/512 [==============================] - 0s 767us/step
batch accuracy: 0.99609375
doing 37 / 73
512/512 [==============================] - 0s 567us/step
batch accuracy: 0.994140625
doing 38 / 73
512/512 [==============================] - 0s 543us/step
batch accuracy: 1.0
doing 39 / 73
512/512 [==============================] - 0s 626us/step
batch accuracy: 0.998046875
doing 40 / 73
512/512 [==============================] - 0s 563us/step
batch accuracy: 1.0
doing 41 / 73
512/512 [==============================] - 0s 626us/step
batch accuracy: 1.0
doing 42 / 73
512/512 [==============================] - 0s 570us/step
batch accuracy: 1.0
doing 43 / 73
512/512 [==============================] - 0s 590us/step
batch accuracy: 1.0
doing 44 / 73
512/512 [==============================] - 0s 586us/step
batch accuracy: 1.0
doing 45 / 73
512/512 [==============================] - 0s 550us/step
batch accuracy: 0.99609375
doing 46 / 73
512/512 [==============================] - 0s 539us/step
batch accuracy: 1.0
doing 47 / 73
512/512 [==============================] - 0s 702us/step
batch accuracy: 1.0
doing 48 / 73
512/512 [==============================] - 0s 618us/step
batch accuracy: 1.0
doing 49 / 73
512/512 [==============================] - 0s 632us/step
batch accuracy: 0.998046875
doing 50 / 73
512/512 [==============================] - 0s 611us/step
batch accuracy: 1.0
doing 51 / 73
512/512 [==============================] - 0s 546us/step
batch accuracy: 0.998046875
doing 52 / 73
512/512 [==============================] - 0s 859us/step
batch accuracy: 1.0
doing 53 / 73
512/512 [==============================] - 0s 877us/step
batch accuracy: 0.998046875
doing 54 / 73
512/512 [==============================] - 1s 1ms/step
batch accuracy: 1.0
doing 55 / 73
512/512 [==============================] - 0s 551us/step
batch accuracy: 0.998046875
doing 56 / 73
512/512 [==============================] - 0s 634us/step
batch accuracy: 0.99609375
doing 57 / 73
512/512 [==============================] - 0s 583us/step
batch accuracy: 1.0
doing 58 / 73
512/512 [==============================] - 0s 629us/step
batch accuracy: 1.0
doing 59 / 73
512/512 [==============================] - 0s 601us/step
batch accuracy: 1.0
doing 60 / 73
512/512 [==============================] - 0s 561us/step
batch accuracy: 1.0
doing 61 / 73
512/512 [==============================] - 0s 537us/step
batch accuracy: 1.0
doing 62 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 0.99609375
doing 63 / 73
512/512 [==============================] - 0s 563us/step
batch accuracy: 1.0
doing 64 / 73
512/512 [==============================] - 0s 661us/step
batch accuracy: 1.0
doing 65 / 73
512/512 [==============================] - 0s 564us/step
batch accuracy: 1.0
doing 66 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 1.0
doing 67 / 73
512/512 [==============================] - 0s 637us/step
batch accuracy: 1.0
doing 68 / 73
512/512 [==============================] - 0s 579us/step
batch accuracy: 1.0
doing 69 / 73
512/512 [==============================] - 0s 573us/step
batch accuracy: 1.0
doing 70 / 73
512/512 [==============================] - 0s 557us/step
batch accuracy: 1.0
doing 71 / 73
512/512 [==============================] - 0s 545us/step
batch accuracy: 1.0
doing 72 / 73
512/512 [==============================] - 0s 551us/step
batch accuracy: 0.998046875
Train accuracy 0.997966609589041
4200/4200 [==============================] - 2s 588us/step
Test loss: 0.11035395232710428
Test accuracy: 0.9723809523809523
==================================================
4 / 5
doing 0 / 73
512/512 [==============================] - 0s 534us/step
batch accuracy: 0.9921875
doing 1 / 73
512/512 [==============================] - 0s 534us/step
batch accuracy: 0.994140625
doing 2 / 73
512/512 [==============================] - 0s 596us/step
batch accuracy: 0.99609375
doing 3 / 73
512/512 [==============================] - 0s 547us/step
batch accuracy: 1.0
doing 4 / 73
512/512 [==============================] - 0s 529us/step
batch accuracy: 1.0
doing 5 / 73
512/512 [==============================] - 0s 538us/step
batch accuracy: 1.0
doing 6 / 73
512/512 [==============================] - 0s 538us/step
batch accuracy: 0.998046875
doing 7 / 73
512/512 [==============================] - 0s 539us/step
batch accuracy: 1.0
doing 8 / 73
512/512 [==============================] - 0s 574us/step
batch accuracy: 1.0
doing 9 / 73
512/512 [==============================] - 0s 540us/step
batch accuracy: 0.998046875
doing 10 / 73
512/512 [==============================] - 0s 575us/step
batch accuracy: 0.994140625
doing 11 / 73
512/512 [==============================] - 0s 580us/step
batch accuracy: 0.998046875
doing 12 / 73
512/512 [==============================] - 0s 719us/step
batch accuracy: 1.0
doing 13 / 73
512/512 [==============================] - 0s 872us/step
batch accuracy: 1.0
doing 14 / 73
512/512 [==============================] - 0s 669us/step
batch accuracy: 1.0
doing 15 / 73
512/512 [==============================] - 0s 605us/step
batch accuracy: 1.0
doing 16 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 1.0
doing 17 / 73
512/512 [==============================] - 0s 597us/step
batch accuracy: 0.998046875
doing 18 / 73
512/512 [==============================] - 0s 565us/step
batch accuracy: 0.998046875
doing 19 / 73
512/512 [==============================] - 0s 596us/step
batch accuracy: 0.998046875
doing 20 / 73
512/512 [==============================] - 0s 622us/step
batch accuracy: 0.998046875
doing 21 / 73
512/512 [==============================] - 0s 560us/step
batch accuracy: 1.0
doing 22 / 73
512/512 [==============================] - 0s 568us/step
batch accuracy: 1.0
doing 23 / 73
512/512 [==============================] - 0s 601us/step
batch accuracy: 0.99609375
doing 24 / 73
512/512 [==============================] - 0s 566us/step
batch accuracy: 0.998046875
doing 25 / 73
512/512 [==============================] - 0s 548us/step
batch accuracy: 1.0
doing 26 / 73
512/512 [==============================] - 0s 539us/step
batch accuracy: 0.99609375
doing 27 / 73
512/512 [==============================] - 0s 558us/step
batch accuracy: 0.998046875
doing 28 / 73
512/512 [==============================] - 0s 588us/step
batch accuracy: 0.998046875
doing 29 / 73
512/512 [==============================] - 0s 561us/step
batch accuracy: 0.998046875
doing 30 / 73
512/512 [==============================] - 0s 646us/step
batch accuracy: 0.998046875
doing 31 / 73
512/512 [==============================] - 0s 645us/step
batch accuracy: 1.0
doing 32 / 73
512/512 [==============================] - 0s 808us/step
batch accuracy: 0.998046875
doing 33 / 73
512/512 [==============================] - 0s 545us/step
batch accuracy: 0.998046875
doing 34 / 73
512/512 [==============================] - 0s 572us/step
batch accuracy: 0.99609375
doing 35 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 1.0
doing 36 / 73
512/512 [==============================] - 0s 639us/step
batch accuracy: 1.0
doing 37 / 73
512/512 [==============================] - 0s 690us/step
batch accuracy: 1.0
doing 38 / 73
512/512 [==============================] - 0s 600us/step
batch accuracy: 1.0
doing 39 / 73
512/512 [==============================] - 0s 579us/step
batch accuracy: 1.0
doing 40 / 73
512/512 [==============================] - 0s 660us/step
batch accuracy: 1.0
doing 41 / 73
512/512 [==============================] - 0s 572us/step
batch accuracy: 1.0
doing 42 / 73
512/512 [==============================] - 0s 570us/step
batch accuracy: 1.0
doing 43 / 73
512/512 [==============================] - 0s 808us/step
batch accuracy: 1.0
doing 44 / 73
512/512 [==============================] - 0s 699us/step
batch accuracy: 1.0
doing 45 / 73
512/512 [==============================] - 0s 897us/step
batch accuracy: 0.998046875
doing 46 / 73
512/512 [==============================] - 0s 746us/step
batch accuracy: 0.998046875
doing 47 / 73
512/512 [==============================] - 0s 851us/step
batch accuracy: 1.0
doing 48 / 73
512/512 [==============================] - 0s 603us/step
batch accuracy: 1.0
doing 49 / 73
512/512 [==============================] - 0s 553us/step
batch accuracy: 1.0
doing 50 / 73
512/512 [==============================] - 0s 588us/step
batch accuracy: 1.0
doing 51 / 73
512/512 [==============================] - 0s 550us/step
batch accuracy: 1.0
doing 52 / 73
512/512 [==============================] - 0s 586us/step
batch accuracy: 1.0
doing 53 / 73
512/512 [==============================] - 0s 542us/step
batch accuracy: 0.998046875
doing 54 / 73
512/512 [==============================] - 0s 577us/step
batch accuracy: 1.0
doing 55 / 73
512/512 [==============================] - 0s 548us/step
batch accuracy: 0.998046875
doing 56 / 73
512/512 [==============================] - 0s 571us/step
batch accuracy: 1.0
doing 57 / 73
512/512 [==============================] - 0s 548us/step
batch accuracy: 1.0
doing 58 / 73
512/512 [==============================] - 0s 574us/step
batch accuracy: 1.0
doing 59 / 73
512/512 [==============================] - 0s 549us/step
batch accuracy: 1.0
doing 60 / 73
512/512 [==============================] - 0s 571us/step
batch accuracy: 1.0
doing 61 / 73
512/512 [==============================] - 0s 555us/step
batch accuracy: 1.0
doing 62 / 73
512/512 [==============================] - 0s 567us/step
batch accuracy: 1.0
doing 63 / 73
512/512 [==============================] - 0s 547us/step
batch accuracy: 1.0
doing 64 / 73
512/512 [==============================] - 0s 545us/step
batch accuracy: 1.0
doing 65 / 73
512/512 [==============================] - 0s 550us/step
batch accuracy: 1.0
doing 66 / 73
512/512 [==============================] - 0s 567us/step
batch accuracy: 1.0
doing 67 / 73
512/512 [==============================] - 0s 554us/step
batch accuracy: 1.0
doing 68 / 73
512/512 [==============================] - 0s 552us/step
batch accuracy: 1.0
doing 69 / 73
512/512 [==============================] - 0s 548us/step
batch accuracy: 1.0
doing 70 / 73
512/512 [==============================] - 0s 544us/step
batch accuracy: 1.0
doing 71 / 73
512/512 [==============================] - 0s 541us/step
batch accuracy: 1.0
doing 72 / 73
512/512 [==============================] - 0s 542us/step
batch accuracy: 0.998046875
Train accuracy 0.9990100599315068
4200/4200 [==============================] - 2s 556us/step
Test loss: 0.1043071129803069
Test accuracy: 0.9769047619047619
"""

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


def lenet(input_shape, num_classes):
    """
    http://tecmemo.wpblog.jp/wp-content/uploads/2017/03/dl_lenet-01.png この表を参考に一部活性化関数を変更してLenetを定義
    """
    model = Sequential()

    # フィルターを6枚用意, 小窓のサイズ5×5, paddingによって入力と出力の画像サイズは同じ
    model.add(Conv2D(
        6, kernel_size=5, padding="same",
        input_shape=input_shape, activation="relu",
        init="he_uniform"
    ))
    # 2, 2でマックスプーリング
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 再度畳み込み、深い層ほどフィルターを増やすのはテクニック
    model.add(Conv2D(16, kernel_size=5, padding="same",
                     activation="relu", init="he_uniform"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten()はマトリックスを1次元ベクトルに変換する層
    # FCにつなぐために必要
    model.add(Flatten())
    model.add(Dense(120, activation="relu", init="he_normal"))
    model.add(Dense(64, activation="relu", init="he_normal"))
    model.add(Dense(num_classes, init="he_normal"))
    model.add(Activation("softmax"))
    return model


# わかりやすさを考慮してこれはやめとくか
# def get_batch(batch_size):
#     """
#     batchを取得する関数
#     """
#     # n_batchs
#     # for でyield
#         yield X_train, Y_train, X_val, Y_val
model = lenet((28, 28), 10)

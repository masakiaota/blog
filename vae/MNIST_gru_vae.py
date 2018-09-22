"""
gruを用いたvae

vaeで時系列に見立てたMNISTを生成
"""
import math
import os
import numpy as np
from keras.layers import Input, InputLayer, Dense, RepeatVector, Lambda, TimeDistributed
from keras.layers import GRU
# from keras.layers import CuDNNGRU as GRU  # GPU用
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import adam
from keras import backend as K
# データの準備


def prepare_data():
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


def seq_vae():
    """
    入力(input)
    ↓
    GRU(encoder)
    ↓
    内部状態
    ↓   ↓
    mean, log_var
    ↓
    zをサンプリング(ここまでencoder)
    ↓（このzを復元された内部状態だとして）
    GRU(decoder)
    ↓
    全結合層(出力)


    戻り値
     model
    """
    LATENT_DIM = 20

    def sampling(args):
        """
        z_mean, z_log_var=argsからzをサンプリングする関数
        戻り値
            z (tf.tensor):サンプリングされた潜在変数
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        # K.exp(0.5 * z_log_var)が分散に標準偏差になっている
        # いきなり標準偏差を求めてしまっても構わないが、負を許容してしまうのでこのようなトリックを用いている
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # encoderの定義
    inputs = Input(shape=(NUM_TIMESTEPS, NUM_INPUT_DIM))
    # (None, NUM_TIMESTEPS,NUM_INPUT_DIM)
    x = GRU(LATENT_DIM)(inputs)
    # (None, LATENT_DIM)
    z_mean = Dense(LATENT_DIM, name='z_mean')(x)  # z_meanを出力
    # (None,LATENT_DIM)
    z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)  # z_sigmaを出力
    # (None, LATENT_DIM)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')(
        [z_mean, z_log_var])  # 2つの変数を受け取ってランダムにサンプリング
    # (None, NATENT_DIM)
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
    print("encoderの構成")
    encoder.summary()
    # encoder部分は入力を受けて平均、分散、そこからランダムサンプリングしたものの3つを返す

    # decoderの定義
    latent_inputs = RepeatVector(
        NUM_TIMESTEPS)(z)
    # (None, NUM_TIMESTEPS, LATENT_DIM)
    x = GRU(LATENT_DIM, return_sequences=True)(latent_inputs)
    # (None, NUM_TIMESTEPS, LATENT_DIM)
    outputs = TimeDistributed(
        Dense(NUM_INPUT_DIM, activation='sigmoid'))(x)
    # (None, NUM_TIMESTEPS, NUM_INPUT_DIM)

    # instantiate decoder model
    # decoder = Model(z, outputs, name='decoder')
    # print("decoderの構成")
    # decoder.summary()

    # デコーダーとエンコーダーの結合
    # encoderの出力の3つめ、つまりzを入力として、decoderを実行する
    vae = Model(inputs, outputs, name='seq_vae')

    # 損失関数をこのモデルに加える
    def loss(inputs, outputs):
        """
        損失関数の定義
        """
        from keras.losses import binary_crossentropy
        z_mean, z_log_var, _ = encoder(inputs)
        reconstruction_loss = binary_crossentropy(
            K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= NUM_INPUT_DIM*NUM_TIMESTEPS
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(loss(inputs, outputs))
    print("seq_vaeの構成")
    vae.summary()
    return vae


if __name__ == "__main__":

    print("="*20+"preparating the data..."+"="*20)

    x_train, x_test = prepare_data()
    print("="*20+"summary of this model"+"="*20)
    seq_vae = seq_vae()

    seq_vae.compile(optimizer="adam")
    # 学習
    seq_vae.fit(x_train,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_split=0.1,
                callbacks=[TensorBoard(log_dir="./MNIST_gru_vae/"), EarlyStopping(patience=3)])
    seq_vae.save('./MNIST_gru_vae/MNIST_gru_vae.h5')

    # 推論
    x_pred = seq_vae.predict(x_test)

    import matplotlib.pyplot as plt
    # データをプロット
    for true, pred in zip(x_test[::1000], x_pred[::1000]):
        plt.matshow(true)

        plt.matshow(pred)
        plt.show()

    encoder = K.function([seq_vae.input], [seq_vae.get_layer("z").output])
    decoder = K.function([seq_vae.get_layer("z").output],
                         [seq_vae.layers[-1].output])
    # 値の確認
    coded = encoder([x_test])[0]
    # zは毎回サンプリングされているので、そこそこ変動する
    print(np.max(coded[2]), np.min(coded[2]), np.mean(coded[2]))
    # ここから生成
    # seq_vae.layers[2].summary()
    gen = decoder([np.random.normal(0, 0.7, size=(20, 20))])[0]
    for g in gen:
        plt.matshow(g)
        plt.show()
"""
"""

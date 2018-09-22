"""
理解して自分で実装
"""
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """
    args(平均と分散のlog)を受け取って正規分布からランダムに返す。
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # K.exp(0.5 * z_log_var)が分散に標準偏差になっている
    # いきなり標準偏差を求めてしまっても構わないが、負を許容してしまうのでこのようなトリックを用いている
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def prepare_data():
    """
    MNISTのデータセットを返す正規化して
    """
    # MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = x_train.reshape(-1, original_dim)
    x_test = x_test.reshape(-1, original_dim)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test


# network parameters
original_dim = 28*28
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

# モデルの構築！！！！
# 分岐するからSequentialでは書けないかな？
# VAE model = encoder + decoder


def vae():
    # まずはencoderとdecoderの構築
    def encoder():
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)  # z_meanを出力
        z_log_var = Dense(latent_dim, name='z_log_var')(x)  # z_sigmaを出力

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')(
            [z_mean, z_log_var])  # 2つの変数を受け取ってランダムにサンプリング
        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        # encoder部分は入力を受けて平均、分散、そこからランダムサンプリングしたものの3つを返す
        return encoder

    def decoder():
        # build decoder model
        # とりあえずモデルを組み立てるだけなので入力を自明に書かなくて良い
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder

    encoder = encoder()
    decoder = decoder()

    inputs = Input(shape=input_shape, name='encoder_input')
    # encoderの出力の3つめ、つまりzを入力として、decoderを実行する
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    def loss(inputs, outputs):
        """
        損失関数の定義
        """
        z_mean, z_log_var, _ = encoder(inputs)
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(loss(inputs, outputs))
    vae.summary()
    return vae


if __name__ == '__main__':
    x_train, x_test = prepare_data()
    vae = vae()

    vae.compile(optimizer='adam')
    vae.summary()

    # train the autoencoder
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')

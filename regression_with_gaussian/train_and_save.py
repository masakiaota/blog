import os
import pandas as pd
import numpy as np
from datetime import datetime

from keras.layers import Dense, RepeatVector, Input, Lambda
from keras.layers import GRU
# from keras.layers import CuDNNGRU as GRU #GPU用
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import adam
from keras import backend as K
from keras.models import Model


def load_data():
    """
    return X_train, X_test, y_train, y_test
    """
    print("loading the data...")
    DATA_DIR="./data/"
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    print("X train",X_train.shape)
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    print("X test",X_test.shape)
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    print("y train",y_train.shape)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print("y test",y_test.shape)
    
    # shapeをglobal変数に
    global NUM_timesteps, NUM_input_dim, NUM_output_dim
    _, NUM_timesteps, NUM_input_dim = X_train.shape
    _, NUM_output_dim = y_train.shape
    
    
    return X_train, X_test, y_train, y_test


def ret_model(est="mu"):
    """
    GRU
    FC
    FC
    mu
    """
    
    
    # hyper parameter
    LATENT = 20
    FC=10
    
    # ネットワークの定義
    inputs = Input(shape=(NUM_timesteps, NUM_input_dim))
    # (, NUM_timesteps, NUM_input_dim)
    gru=GRU(LATENT)(inputs)
    # (, LATENT)
    fc=Dense(FC,activation="relu")(gru)
    # (, FC)
    if est is not "mu":
        output=Dense(NUM_output_dim,activation = "sigmoid")(fc)
    else:
        output=Dense(NUM_output_dim)(fc)
    
    model = Model(inputs,output)
    
    model.summary()
    
    return model

if __name__=="__main__":
    
    X_train, X_test, y_train, y_test = load_data()
    
    mu_predictor = ret_model()
    mu_predictor.compile(optimizer=adam(lr=0.001),loss="mean_squared_error")
    logdir="./"+str(datetime.now().strftime('%s'))+"mu/"
    mu_predictor.fit(X_train, y_train,
             epochs=20,
             batch_size=128,
             shuffle=True,
             validation_split=0.1,
             callbacks=[TensorBoard(log_dir=logdir), 
                       EarlyStopping(patience=2),
                       ModelCheckpoint(filepath = logdir+'model_epoch.{epoch:02d}-los{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')])
    print("mu_predictor is save to", logdir)
    
    # 二乗誤差系列を取っている
    print("calculating squared error ...")
    temp=mu_predictor.predict(X_train)
    err_var=np.square(temp-y_train)
    
    
    var_predictor=ret_model(est="var")
    var_predictor.compile(optimizer=adam(lr=0.001),loss="mean_squared_error")
    logdir="./"+str(datetime.now().strftime('%s'))+"var/"
    var_predictor.fit(X_train, err_var,
             epochs=20,
             batch_size=128,
             shuffle=True,
             validation_split=0.1,
             callbacks=[TensorBoard(log_dir=logdir), 
                       EarlyStopping(patience=2),
                       ModelCheckpoint(filepath = logdir+'model_epoch.{epoch:02d}-los{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')])
    print("var_predictor is save to",logdir)

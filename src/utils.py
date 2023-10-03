from keras.applications.mobilenet import MobileNet 
from keras.datasets import cifar10
from keras.utils import to_categorical
import random
import numpy as np

def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return (X_train, Y_train), (X_test, Y_test)

def sampling_data(num_sampling):
    (x_train, y_train), (x_test, y_test) = load_dataset()
    num_of_each_dataset = num_sampling
    split_data_index = []
    while len(split_data_index) < num_of_each_dataset:
        item = random.choice(range(x_train.shape[0]))
        if item not in split_data_index:
            split_data_index.append(item)
    new_x_train = np.asarray([x_train[k] for k in split_data_index])
    new_y_train = np.asarray([y_train[k] for k in split_data_index])
    return new_x_train, new_y_train

from keras.layers import Input, Dense, BatchNormalization, Flatten
from keras import Model
from keras.optimizers import SGD

def model_init():
    model = MobileNet(include_top=False,input_tensor=Input(shape=(32,32,3)))
    x = model.output
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10,activation='softmax')(x)
    model = Model(model.input,x)
    return model

def fedAVG(server_weight):
    avg_weight = np.array(server_weight[0])
    if len(server_weight) > 1:
        for i in range(1, len(server_weight)):
            avg_weight += server_weight[i]
    avg_weight = avg_weight / len(server_weight)
    return avg_weight
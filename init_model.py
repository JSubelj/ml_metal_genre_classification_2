# -*- coding: utf-8 -*-

import numpy
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from config import no_of_samples_per_split, no_of_coef_per_sample, batch_size, genres

def create_model(no_of_layers=1, num_of_nodes=(1024,)):
    print("[+] Creating model...")
    convnet = input_data(shape=[None, no_of_samples_per_split, no_of_coef_per_sample, 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    for i in range(no_of_layers):
        convnet = fully_connected(convnet, num_of_nodes[i], activation='elu')
        convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, len(genres), activation='softmax')
    convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

    model = tflearn.DNN(convnet)
    print("Model created!")

    return model


if __name__ == "__main__":
    create_model(3, (1024,1024,1024))

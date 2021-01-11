# -*- coding: utf-8 -*-
"""tf13GAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uKd2kw4qwOfyaMlDFfAxZrgBC3e1fe3k
"""

# GAN(생성적 적대 신경망)
# DCGAN(Deep Convolutional GAN)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, \
        BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

if not os.path.exists("./genimages"):
    os.makedirs("./genimages")

np.random.seed(3)
tf.random.set_seed(3)

generator = Sequential()

generator.add(Dense(128 * 7 * 7, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())  # 안정적 학습 유도
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D())  # 2배 차원 확대
generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization()) 
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D()) 
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
print(generator.summary())

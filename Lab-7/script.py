import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train,_),(x_test,_) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

noise_factor = 0.5
x_train_noisy = np.clip(x_train + noise_factor * np.random.randn(*x_train.shape), 0,1)
x_test_noisy  = np.clip(x_test  + noise_factor * np.random.randn(*x_test.shape), 0,1)

input_img = layers.Input(shape=(28,28,1))
x = layers.Conv2D(32,3,activation='relu',padding='same')(input_img)
x = layers.MaxPool2D(2,padding='same')(x)
x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(1,3,activation='sigmoid',padding='same')(x)

auto = models.Model(input_img, decoded)
auto.compile(optimizer='adam', loss='mse')
auto.fit(x_train_noisy, x_train, epochs=5, batch_size=128, validation_data=(x_test_noisy, x_test))

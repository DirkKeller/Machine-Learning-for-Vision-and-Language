# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
from skimage.util.shape import view_as_windows
import numpy as np
import layers as ls


# def print_hi(name):
# Use a breakpoint in the code line below to debug your script.

# print(device_lib.list_local_devices())

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# with tf.device("gpu:0"):
#    first_models(x_train, y_train, x_test, y_test)

#    dnn_model(x_train, y_train, x_test, y_test)

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# with tf.device("gpu:0"):

#    ex2_models(x_train, y_train, x_test, y_test)


def make_plot(history, lim, lim2):
    plt.figure(figsize=(10, 20))
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('model accuracy')
    axs[0].set(ylabel='accuracy')

    axs[0].set(xlabel='epoch')
    axs[0].set_ylim(lim)
    axs[0].legend(['training', 'validation'], loc='upper left')

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set(ylabel='loss')
    axs[1].set(xlabel='epoch')
    axs[1].set_ylim(lim2)
    axs[1].legend(['training', 'validation'], loc='upper left')
    fig.set_size_inches(10.5, 6.5)
    fig.show()


def first_models(x_train, y_train, x_test, y_test):
    X_train = x_train.reshape(60000, 784) / 255
    X_test = x_test.reshape(10000, 784) / 255
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_shape=(784,)))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics='accuracy')
    history = model.fit(X_train, Y_train, batch_size=128, epochs=12, verbose=1, validation_split=0.2)
    model.summary()

    make_plot(history, [0.8, 1.0], [0.0, .4])

    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(loss)
    print(accuracy)

    model2 = keras.Sequential()
    model2.add(keras.layers.Dense(256, input_shape=(784,), activation='relu'))
    model2.add(keras.layers.Dense(10, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics='accuracy')
    history = model2.fit(X_train, Y_train, batch_size=128, epochs=12, verbose=1, validation_split=0.2)
    model2.summary()

    make_plot(history, [0.8, 1.0], [0.0, 0.4])
    loss, accuracy = model2.evaluate(X_test, Y_test, verbose=0)
    print(loss)
    print(accuracy)


def dnn_model(x_train, y_train, x_test, y_test):
    X_train_2 = x_train.reshape(60000, 28, 28, 1) / 255
    X_test_2 = x_test.reshape(10000, 28, 28, 1) / 255
    Y_train_2 = keras.utils.to_categorical(y_train, 10)
    Y_test_2 = keras.utils.to_categorical(y_test, 10)

    dnn = keras.Sequential()
    dnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                activation="relu", input_shape=(28, 28, 1)))
    dnn.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                activation="relu"))
    dnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    dnn.add(keras.layers.Flatten())
    dnn.add(keras.layers.Dense(128, activation="relu"))
    dnn.add(keras.layers.Dense(10, activation="softmax"))
    dnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(learning_rate=1),
                metrics='accuracy')

    history = dnn.fit(X_train_2, Y_train_2, batch_size=512, epochs=6, verbose=1, validation_split=0.2)
    dnn.summary()

    make_plot(history, [0.8, 1.0], [0.0, 0.6])

    loss, accuracy = dnn.evaluate(X_test_2, Y_test_2, verbose=0)
    print(loss)
    print(accuracy)

    dnn_2 = keras.Sequential()
    dnn_2.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu", input_shape=(28, 28, 1)))
    dnn_2.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
    dnn_2.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    dnn_2.add(keras.layers.Dropout(rate=0.25))
    dnn_2.add(keras.layers.Flatten())
    dnn_2.add(keras.layers.Dense(128, activation="relu"))
    dnn_2.add(keras.layers.Dropout(rate=0.50))
    dnn_2.add(keras.layers.Dense(10, activation="softmax"))
    dnn_2.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(learning_rate=1),
                  metrics='accuracy')

    history = dnn_2.fit(X_train_2, Y_train_2, batch_size=512, epochs=6, verbose=1, validation_split=0.2)
    dnn_2.summary()

    make_plot(history, [0.8, 1.0], [0.0, 0.6])

    loss, accuracy = dnn_2.evaluate(X_test_2, Y_test_2, verbose=0)
    print(loss)
    print(accuracy)


def ex2_models(x_train, y_train, x_test, y_test):
    X_train = x_train / 255
    X_test = x_test / 255
    Y_train = keras.utils.to_categorical(y_train, 10)
    Y_test = keras.utils.to_categorical(y_test, 10)

    dnn = keras.Sequential()
    dnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                activation="relu", input_shape=(32, 32, 3), padding="same"))
    dnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                activation="relu"))
    dnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    dnn.add(keras.layers.Dropout(rate=0.25))
    dnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                activation="relu", padding="same"))
    dnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                activation="relu"))
    dnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    dnn.add(keras.layers.Dropout(rate=0.25))
    dnn.add(keras.layers.Flatten())
    dnn.add(keras.layers.Dense(512, activation="relu"))
    dnn.add(keras.layers.Dropout(rate=0.50))
    dnn.add(keras.layers.Dense(10, activation="softmax"))
    dnn.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=0.0001, decay=1e-6),
                metrics='accuracy')

    history = dnn.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, Y_test)
                      , shuffle=True)
    dnn.summary()
    print(history)
    make_plot(history, [0.25, 0.8], [0.7, 1.8])


def build_Network():
    layers = [ls.Conv_Layer(np.random.rand(5, 2, 2, 9)),
              ls.MaxPool_Layer([2, 2]),
              ls.Relu_Layer(),
              ls.Conv_Layer(np.random.rand(5, 2, 2, 5)),
              ls.MaxPool_Layer([2, 2]),
              ls.Normalize_Layer(),
              ls.Fully_Connected_Layer(10),
              ls.Softmax_Layer()]
    return ls.NN_Model(layers)


from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_X, train_y), (test_X, test_y) = mnist.load_data()

img_rgb = np.random.rand(32, 32, 9)
nn = build_Network()
predictions = nn.predicts(train_X[0:50])
print(nn.accuracy(predictions,train_y[0:50]))

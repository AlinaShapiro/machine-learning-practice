
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, MaxPooling2D, LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    batch_size = 128
    num_classes = 10
    epochs = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train_orig), (x_test, y_test_orig) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train_orig, num_classes)
    y_test = to_categorical(y_test_orig, num_classes)

    indexes = np.random.permutation(len(x_train))
    train_data = x_train[indexes]
    train_labels = y_train[indexes]

    val_count = int(0.1 * len(x_train))
    x_val = train_data[:val_count, :]
    y_val = train_labels[:val_count, :]

    # leave rest in training set
    part_x_train = train_data[val_count:, :]
    part_y_train = train_labels[val_count:, :]

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='tanh'))
    tf.keras.layers.Conv2D(64, (3, 3), activation='softmax')
    model.add(Dense(216, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(part_x_train, part_y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


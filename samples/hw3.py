from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from losses import softmax_cross_entropy
from session import Session
from activations import relu
from samples.download_mnist import load
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_metrics(epochs, train_losses, train_accuracies, test_losses, test_accuracies, png_name):
    plot_x = np.array(range(epochs)) + 1
    plt.subplot(2, 1, 1)
    p1, = plt.plot(plot_x, train_losses, color='blue', marker='o')
    p2, = plt.plot(plot_x, test_losses, color='red', marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([p1, p2], ['train', 'test'])
    plt.subplot(2, 1, 2)
    p3, = plt.plot(plot_x, train_accuracies, color='blue', marker='o')
    p4, = plt.plot(plot_x, test_accuracies, color='red', marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend([p3, p4], ['train', 'test'])
    plt.savefig('{}.png'.format(png_name))
    plt.show()


def hw3_a(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float)
    y_train = y_train.reshape([-1, 1])
    x_test = x_test.astype(float)
    y_test = y_test.reshape([-1, 1])
    il = Input(784)
    dl1 = Dense([il], 200, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 50, use_bias=True, activation=relu)
    dl3 = Dense([dl2], 10, use_bias=True)
    ol = Output([dl3], 10, loss_function=softmax_cross_entropy, learning_rate=lr)
    sess = Session([ol], x_train, y_train, x_test, y_test)
    history = sess.train(epochs, batch_size)
    train_losses, train_accuracies, test_losses, test_accuracies = history['train_losses'], \
        history['train_accuracies'], \
        history['test_losses'], \
        history['test_accuracies']
    plot_metrics(epochs, train_losses, train_accuracies, test_losses, test_accuracies, 'hw3_a')
    return sess, history


def hw3_b(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float) / 255
    y_train = y_train.reshape([-1, 1])
    x_test = x_test.astype(float) / 255
    y_test = y_test.reshape([-1, 1])
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, use_bias=True, activation='relu'),
        tf.keras.layers.Dense(50, use_bias=True, activation='relu'),
        tf.keras.layers.Dense(10, use_bias=True, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)
    train_losses, train_accuracies, test_losses, test_accuracies = history.history['loss'], \
        history.history['acc'], \
        history.history['val_loss'], \
        history.history['val_acc']
    train_losses = train_losses[1:]
    train_accuracies = train_accuracies[1:]
    eval_his = model.evaluate(x_train, y_train)
    train_losses.append(eval_his[0])
    train_accuracies.append(eval_his[1])
    plot_metrics(epochs, train_losses, train_accuracies, test_losses, test_accuracies, 'hw3_b')
    return model, history

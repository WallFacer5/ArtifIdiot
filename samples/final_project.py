from samples.download_mnist import load
from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from layers.conv2d import Conv2d
from layers.max_pool2d import MaxPool2d
from layers.flatten import Flatten
from losses import softmax_cross_entropy
from session import Session
from activations import relu
import numpy as np
import matplotlib.pyplot as plt


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


def lenet(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float) / 255
    x_train = x_train.reshape([-1, 28, 28, 1])[:200]
    y_train = y_train.reshape([-1, 1])[:200]
    x_test = x_test.astype(float) / 255
    x_test = x_test.reshape([-1, 28, 28, 1])[:80]
    y_test = y_test.reshape([-1, 1])[:80]

    il = Input([28, 28, 1])
    cl1 = Conv2d([il], 6, kernel_size=[5, 5], strides=[1, 1], use_bias=True, activation=relu)
    pl1 = MaxPool2d([cl1], [2, 2])
    cl2 = Conv2d([pl1], 16, kernel_size=[5, 5], strides=[1, 1], use_bias=True, activation=relu)
    fl = Flatten([cl2])
    dl1 = Dense([fl], 120, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 84, use_bias=True, activation=relu)
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

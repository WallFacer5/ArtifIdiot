from constants import Directions
import numpy as np
import matplotlib.pyplot as plt
import time


def simple_run(pool, direction):
    def forward(f_pool):
        node_to_run = f_pool[0]
        f_pool = f_pool[1:]
        node_to_run.forward()
        f_pool.extend(list(filter(lambda ol: ol.can_forward, node_to_run.get_output_layers())))
        return f_pool

    def backward(b_pool):
        node_to_run = b_pool[0]
        b_pool = b_pool[1:]
        node_to_run.backward()
        b_pool.extend(list(filter(lambda il: il.can_backward, node_to_run.get_input_layers())))
        return b_pool

    return forward(pool) if direction == Directions.forward else backward(pool)


class Session:
    def __init__(self, ends, train_x, train_y, test_x, test_y, run_algo=simple_run):
        self.ends = ends
        self.starts = set()
        list(map(lambda e: self.starts.update(e.get_starts()), ends))
        self.pool = []
        self.run_algo = run_algo
        self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y
        self.cur_pred = []

    def get_pool(self):
        return self.pool

    def run_batch(self, batch_x, batch_y, need_backward=True):
        self.pool = list(self.starts)
        list(map(lambda il: il.set_cur_input(batch_x), self.starts))
        list(map(lambda ol: ol.set_cur_y_true(batch_y), self.ends))
        while self.pool:
            self.pool = self.run_algo(self.pool, direction=Directions.forward)
        self.cur_pred.extend(self.ends[0].get_cur_outputs())
        if need_backward:
            self.pool = self.ends
            while self.pool:
                self.pool = self.run_algo(self.pool, direction=Directions.backward)

    def train_epoch(self, batch_size=1, need_backward=True):
        perm = np.random.permutation(self.train_x.shape[0])
        epoch_x = self.train_x[perm]
        epoch_y = self.train_y[perm]
        count = len(self.train_x)
        start_time = time.time()
        self.cur_pred = []
        i = 0
        while i < count:
            # self.run_batch(self.train_x[perm[i:i+batch_size]], self.train_y[perm[i:i+batch_size]], need_backward)
            self.run_batch(epoch_x[i:i + batch_size], epoch_y[i:i + batch_size], need_backward)
            i += batch_size
        # print(self.cur_pred)
        time_spent = time.time() - start_time
        self.cur_pred = []
        self.run_batch(self.train_x, self.train_y, need_backward=False)
        train_loss, _, train_accuracy = self.ends[0].loss_function(self.train_y, self.cur_pred, 0)
        self.cur_pred = []
        self.run_batch(self.test_x, self.test_y, need_backward=False)
        test_loss, _, test_accuracy = self.ends[0].loss_function(self.test_y, self.cur_pred, 0)
        # print(self.train_y, self.cur_pred)
        return train_loss, train_accuracy, test_loss, test_accuracy, time_spent

    def train(self, epochs, batch_size):
        plot_x = np.array(range(epochs)) + 1
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        for i in range(epochs):
            train_loss, train_accuracy, test_loss, test_accuracy, time_spent = self.train_epoch(batch_size=batch_size)
            print('Epoch {}:\ttime spent: {}s\nTrain loss: {};\taccuracy: {};\nTest loss: {};\taccuracy: {}.'.format(
                i+1, '%.4f' % time_spent, train_loss, train_accuracy, test_loss, test_accuracy))
            print()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
        train_loss, train_accuracy, test_loss, test_accuracy, time_spent = self.train_epoch(batch_size=batch_size,
                                                                                            need_backward=False)
        print('Final evaluate:\ttime spent: {}s\nTrain loss: {};\taccuracy: {};\nTest loss: {};\taccuracy: {}.'.format(
            '%.4f' % time_spent, train_loss, train_accuracy, test_loss, test_accuracy))

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
        plt.savefig('hw3_a.png')
        plt.show()

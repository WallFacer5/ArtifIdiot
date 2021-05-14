import numpy as np
from layers.base import Layer
from numba import jit


class MaxPool2d(Layer):
    def __init__(self, input_layers, pool_size=[2, 2]):
        super().__init__(input_layers, None, False, None, None)
        input_shape = self.input_shapes[0]
        self.pool_size = pool_size
        self.output_shape = [input_shape[0] // pool_size[0], input_shape[1] // pool_size[1], input_shape[2]]

    @staticmethod
    @jit(nopython=True)
    def in_forward(output_shape, cur_outputs, max_flags, pool_size, cur_input):
        def get_max(_cur_value):
            max_val = np.max(_cur_value)
            for i in range(_cur_value.shape[0]):
                for j in range(_cur_value.shape[1]):
                    if _cur_value[i, j] == max_val:
                        return max_val, i, j

        # cur_outputs = np.zeros([cur_input.shape[0]] + output_shape)
        # max_flags = np.zeros([cur_input.shape[0]] + input_shapes[0])
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                for s in range(cur_input.shape[0]):
                    for k in range(output_shape[2]):
                        cur_value = cur_input[s,
                                    i * pool_size[0]:i * pool_size[0] + pool_size[0],
                                    j * pool_size[1]:j * pool_size[1] + pool_size[1], k]
                        max_val, max_i, max_j = get_max(cur_value)
                        cur_outputs[s, i, j, k] = max_val
                        max_flags[s, i * pool_size[0] + max_i, j * pool_size[1] + max_j, k] = 1
        return cur_outputs, max_flags

    def forward(self):
        '''
        self.cur_outputs = np.zeros([self.cur_inputs[0].shape[0]] + self.output_shape)
        self.max_flags = np.zeros([self.cur_inputs[0].shape[0]] + self.input_shapes[0])
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for s in range(self.cur_inputs[0].shape[0]):
                    for k in range(self.output_shape[2]):
                        cur_value = self.cur_inputs[0][s,
                                    i * self.pool_size[0]:i * self.pool_size[0] + self.pool_size[0],
                                    j * self.pool_size[1]:j * self.pool_size[1] + self.pool_size[1], k]
                        max_val, max_i, max_j = self.get_max(cur_value)
                        self.cur_outputs[s, i, j, k] = max_val
                        self.max_flags[s, i * self.pool_size[0] + max_i, j * self.pool_size[1] + max_j, k] = 1
        '''
        cur_outputs = np.zeros([self.cur_inputs[0].shape[0]] + self.output_shape)
        max_flags = np.zeros([self.cur_inputs[0].shape[0]] + self.input_shapes[0])
        self.cur_outputs, self.max_flags = self.in_forward(self.output_shape, cur_outputs, max_flags, self.pool_size,
                                                           self.cur_inputs[0])
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))
        self.clear_cur_inputs_flags()

    @staticmethod
    @jit(nopython=True)
    def in_backward(delta, cur_inputs, output_shape, pool_size, max_flags, cur_deltas):
        # delta = np.zeros_like(cur_inputs[0], dtype='float64')
        # print(np.array(self.cur_deltas).shape)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                for s in range(cur_inputs[0].shape[0]):
                    for k in range(output_shape[2]):
                        delta[s, i * pool_size[0]:i * pool_size[0] + pool_size[0],
                        j * pool_size[1]:j * pool_size[1] + pool_size[1], k] += cur_deltas[0][
                            s][i][j][k]
        return delta * max_flags

    def backward(self):
        '''
        delta = np.zeros_like(self.cur_inputs[0], dtype='float64')
        # print(np.array(self.cur_deltas).shape)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for s in range(self.cur_inputs[0].shape[0]):
                    for k in range(self.output_shape[2]):
                        delta[s, i * self.pool_size[0]:i * self.pool_size[0] + self.pool_size[0],
                        j * self.pool_size[1]:j * self.pool_size[1] + self.pool_size[1], k] += self.cur_deltas[0][
                            s][i][j][k]
        # print(delta * self.max_flags)
        '''
        delta = np.zeros_like(self.cur_inputs[0], dtype='float64')
        delta = self.in_backward(delta, self.cur_inputs, self.output_shape, self.pool_size, self.max_flags,
                                 self.cur_deltas)
        # list(map(lambda layer: layer.append_cur_delta(self, delta * self.max_flags), self.input_layers))
        list(map(lambda layer: layer.append_cur_delta(self, delta), self.input_layers))
        self.clear_cur_deltas_flags()

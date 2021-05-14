import numpy as np
from layers.base import Layer


class MaxPool2d(Layer):
    def __init__(self, input_layers, pool_size=[2, 2]):
        super().__init__(input_layers, None, False, None, None)
        input_shape = self.input_shapes[0]
        self.pool_size = pool_size
        self.output_shape = [input_shape[0] // pool_size[0], input_shape[1] // pool_size[1], input_shape[2]]

    @staticmethod
    def get_max(cur_value):
        max_val = np.max(cur_value)
        for i in range(cur_value.shape[0]):
            for j in range(cur_value.shape[1]):
                if cur_value[i, j] == max_val:
                    return max_val, i, j

    @jit(nopython=True)
    def forward(self):
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
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))
        self.clear_cur_inputs_flags()

    @jit(nopython=True)
    def backward(self):
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
        list(map(lambda layer: layer.append_cur_delta(self, delta * self.max_flags), self.input_layers))
        self.clear_cur_deltas_flags()

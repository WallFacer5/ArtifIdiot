import numpy as np
from layers.base import Layer
from constants import Directions


class Flatten(Layer):
    def __init__(self, input_layers):
        super().__init__(input_layers, None, False, None, None)
        self.output_shape = np.prod(self.input_shapes[0])

    def forward(self):
        # print(self.cur_inputs)
        self.cur_outputs = self.cur_inputs[0].reshape([self.cur_inputs[0].shape[0], self.output_shape])
        # print(self.cur_outputs)
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))
        self.clear_cur_inputs_flags()

    def backward(self):
        # print(self.cur_deltas)
        delta = np.array(self.cur_deltas[0]).reshape([self.cur_inputs[0].shape[0]] + self.input_shapes[0])
        # print(delta)
        list(map(lambda layer: layer.append_cur_delta(self, delta), self.input_layers))
        self.clear_cur_deltas_flags()

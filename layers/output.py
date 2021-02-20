import numpy as np
from layers.base import Layer


class Output(Layer):
    def __init__(self, input_layers, output_shape, loss_function=None):
        super().__init__(input_layers, output_shape)

    def get_cur_outputs(self):
        return self.cur_outputs

    def forward(self):
        self.cur_outputs = self.cur_inputs[0]

    def backward(self):
        list(map(lambda layer: layer.append_cur_delta(self, self.delta), self.input_layers))
        self.delta = np.zeros((self.input_shapes[0], self.output_shape))
        self.clear_cur_deltas_flags()

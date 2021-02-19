import numpy as np
from layers.base import Layer


class Output(Layer):
    def __init__(self, input_layers, output_shape, loss_function=None):
        super().__init__(input_layers, output_shape)

    def get_cur_outputs(self):
        return self.cur_outputs

    def forward(self):
        self.cur_outputs = self.cur_inputs[0]

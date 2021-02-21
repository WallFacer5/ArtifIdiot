import numpy as np
from layers.base import Layer


class Input(Layer):
    def __init__(self, output_shape):
        super().__init__([], output_shape)

    def set_cur_input(self, values):
        self.cur_inputs = values
        self.cur_outputs = values

    def append_cur_delta(self, _layer_ref, values):
        self.cur_deltas_ready_flags.add(self.output_layers[_layer_ref])

    def forward(self):
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))

    def backward(self):
        pass

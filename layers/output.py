import numpy as np
from layers.base import Layer


class Output(Layer):
    def __init__(self, input_layers, output_shape, loss_function=None, learning_rate=0.1):
        super().__init__(input_layers, output_shape)
        self.loss_function = loss_function
        self.cur_y_true = None
        self.learning_rate = learning_rate
        self.delta = np.zeros([0, self.output_shape])
        self.cur_loss = None

    def get_cur_outputs(self):
        return self.cur_outputs

    def forward(self):
        self.cur_outputs = self.cur_inputs[0]
        self.clear_cur_inputs_flags()

    def backward(self):
        self.cur_loss, self.delta = self.loss_function(self.cur_y_true, self.cur_outputs, self.learning_rate)
        list(map(lambda layer: layer.append_cur_delta(self, np.array(self.delta)), self.input_layers))
        self.clear_cur_deltas_flags()

    def set_cur_y_true(self, y_values):
        self.cur_y_true = y_values


import torch.optim
from torch.nn import Module, Linear
from torch.nn.functional import relu


class Net(Module):
    def __init__(self, n_input: int = 2, n_output: int = 2, n_hidden: int = 20, n_layer: int = 2):
        super(Net, self).__init__()
        self.n_input = n_input
        self.n_outputs = n_output
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.layers = [Linear(n_input, n_hidden)]
        for _ in range(n_layer - 2):
            self.layers.append(Linear(n_hidden, n_hidden))
        self.layers.append(Linear(n_hidden, n_output))

    def forward(self, inputs):
        pass

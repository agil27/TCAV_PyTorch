from copy import deepcopy
from torch.autograd import grad


class ModelWrapper(object):
    def __init__(self, model, layers):
        self.model = deepcopy(model)
        self.intermediate_activations = {}
        for name, module in self.model._modules.items():
            if name in layers.keys():
                # define the hook function
                def save_activation(name, input, output):
                    self.intermediate_activations[name] = output
                # register the hook
                module.register_forward_hook(save_activation)

    def generate_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        return grad(self.output[:, c], activation)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, x):
        self.output = self.model(x)
        return self.output

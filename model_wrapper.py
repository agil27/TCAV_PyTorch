from copy import deepcopy
from torch.autograd import grad
import torch
import numpy as np


class ModelWrapper(object):
    def __init__(self, model, layers):
        self.model = deepcopy(model)
        self.intermediate_activations = {}

        def save_activation(name):
            '''create specific hook by module name'''
            def hook(module, input, output):
                self.intermediate_activations[name] = output
            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(save_activation(name))

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        activation.register_hook(self.save_gradient)
        logit = self.output[:, c]
        logit.backward()
        # gradients = grad(logit, activation, retain_graph=True)[0]
        # gradients = gradients.cpu().detach().numpy()
        gradients = self.gradients.cpu().detach().numpy()
        return gradients

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

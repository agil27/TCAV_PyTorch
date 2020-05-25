import numpy as np
from cav import CAV
import os
from utils import get_activations, load_activations


def directional_derivative(model, x, cav, layer_name, class_name):
    _ = model(x)
    gradient = model.generate_gradient(class_name, layer_name).reshape(-1)
    return np.dot(gradient, cav) < 0


def tcav_score(model, data_loader, class_name, cav, layer_name):
    derivatives = [directional_derivative(model, x, cav, layer_name, class_name) for x in data_loader]
    score = np.array(derivatives).astype(np.int).sum(axis=0) / len(derivatives)
    return score


class TCAV(object):
    def __init__(self, model, input_dataloader, concept_dataloaders, num_classes, max_samples):
        self.model = model
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = concept_dataloaders.keys()
        self.output_dir = 'output'
        self.max_samples = max_samples
        self.lr = 1e-3
        self.model_type = 'logistic'
        self.num_classes = num_classes

    def generate_activations(self, layer_names):
        for concept_name, data_loader in self.concept_dataloaders.items():
            get_activations(self.model, self.output_dir, data_loader, concept_name, layer_names, self.max_samples)

    def load_activations(self):
        self.activations = {}
        for concept_name in self.concepts:
            self.activations[concept_name] = load_activations(
                os.path.join(self.output_dir, 'activations_%s.h5' % concept_name))

    def generate_cavs(self, layer_name):
        cav_trainer = CAV(self.concepts, layer_name, self.lr, self.model_type)
        cav_trainer.train(self.activations)
        self.cavs = cav_trainer.get_cav()

    def calculate_tcav_score(self, layer_name):
        self.scores = [{} for _ in range(self.num_classes)]
        for k in self.num_classes:
            for i, cav in enumerate(self.cavs):
                self.scores[k][self.concepts[i]] = tcav_score(self.model, self.input_dataloader, k, cav, layer_name)

    def get_tcav_score(self):
        return self.scores

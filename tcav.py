import numpy as np
from cav import CAV
import os
from utils import get_activations, load_activations
import torch
from tqdm import tqdm

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def directional_derivative(model, cav, layer_name, class_name):
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    return np.dot(gradient, cav) < 0


def tcav_score(model, data_loader, cav, layer_name, num_classes, concept):
    model.eval()
    derivatives = [[] for _ in range(num_classes)]
    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description('Calculating tcav score for %s' % concept)
    for x, _ in tcav_bar:
        x = x.to(device)
        # x.requires_grad_(True)
        outputs = model(x)
        k = outputs.max(dim=1)[1]
        if k < num_classes:
            derivatives[k].append(directional_derivative(model, cav, layer_name, k))
    score = np.zeros(num_classes)
    for k in range(num_classes):
        score[k] = np.array(derivatives[k]).astype(np.int).sum(axis=0) / len(derivatives[k])
    return score


class TCAV(object):
    def __init__(self, model, input_dataloader, concept_dataloaders, num_classes, max_samples):
        self.model = model
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = list(concept_dataloaders.keys())
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
        self.scores = np.zeros((self.cavs.shape[0], self.num_classes))
        for i, cav in enumerate(self.cavs):
            self.scores[i] = tcav_score(self.model, self.input_dataloader, cav, layer_name, self.num_classes, self.concepts[i])
        self.scores = self.scores.T.tolist()

    def get_tcav_score(self):
        return self.scores

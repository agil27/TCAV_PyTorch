import numpy as np
from cav import CAV
import os
from utils import get_activations, load_activations
import torch
from tqdm import tqdm
from copy import deepcopy


use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def directional_derivative(model, cav, layer_name, class_name):
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    return np.dot(gradient, cav) < 0


def tcav_score(model, data_loader, cav, layer_name, class_list, concept):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []

    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description('Calculating tcav score for %s' % concept)
    for x, _ in tcav_bar:
        model.eval()
        x = x.to(device)
        outputs = model(x)
        k = int(outputs.max(dim=1)[1].cpu().detach().numpy())
        if k in class_list:
            derivatives[k].append(directional_derivative(model, cav, layer_name, k))

    score = np.zeros(len(class_list))
    for i, k in enumerate(class_list):
        score[i] = np.array(derivatives[k]).astype(np.int).sum(axis=0) / len(derivatives[k])
    return score


class TCAV(object):
    def __init__(self, model, input_dataloader, concept_dataloaders, class_list, max_samples):
        self.model = model
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = list(concept_dataloaders.keys())
        self.output_dir = 'output'
        self.max_samples = max_samples
        self.lr = 1e-3
        self.model_type = 'logistic'
        self.class_list = class_list

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

    def calculate_tcav_score(self, layer_name, output_path):
        self.scores = np.zeros((self.cavs.shape[0], len(self.class_list)))
        for i, cav in enumerate(self.cavs):
            self.scores[i] = tcav_score(self.model, self.input_dataloader, cav, layer_name, self.class_list,
                                        self.concepts[i])
        # print(self.scores)
        np.save(output_path, self.scores)

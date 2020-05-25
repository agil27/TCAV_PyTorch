import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import h5py

device = 'gpu' if torch.cuda.is_available() else 'cpu'


def get_activations(model, output_dir, data_loader, concept_name, layer_names, max_samples):
    '''
    The function to generate the activations of all layers for ONE concept only
    :param model:
    :param output_dir:
    :param data_loader: the dataloader for the input of ONE concept
    :param layer_names:
    :param max_samples:
    :return:
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model = model.eval()
    activations = {}
    for l in layer_names:
        activations[l] = []

    for i, data in enumerate(data_loader):
        if i == max_samples:
            break
        data = data.to(device)
        outputs = model(data)
        for l in layer_names:
            z = model.intermediate_activations[l].cpu().detach().numpy()
            activations[l].append(z)

    for l in layer_names:
        activations[l] = np.stack(activations[l], axis=0)

    with h5py.File(os.path.join(output_dir, 'activations_%.h5' % concept_name), 'w') as f:
        for l in layer_names:
            f.create_dataset(l, data=activations[l])


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations

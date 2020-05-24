import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import h5py

device = 'gpu' if torch.cuda.is_available() else 'cpu'


def get_activations(model, output_dir, data_loader, layer_names, max_samples):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model = model.eval()

    for i, data in enumerate(data_loader):
        if i == max_samples:
            break
        data = data.to(device)
        activations = {}
        outputs = model(data)
        for l in layer_names:
            z = model.intermediates[l].cpu().detach().numpy()
            if l not in activations.keys():
                activations[l] = z
            else:
                activations[l] = np.append(activations[l], z, axis=0)

        with h5py.File(os.path.join(output_dir, 'activations.h5'), 'w') as f:
            for l in layer_names:
                f.create_dataset(name=l, data=activations[l])


def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations

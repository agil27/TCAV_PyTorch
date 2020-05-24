import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split


def flatten_activations_and_get_labels(concepts, layer_name, activations):
    # in case of different activation shapes
    min_size = np.min([activations[c][layer_name].size(0) for c in concepts])
    # flatten the activations and acquire the concept dictionary
    data = []
    concept_labels = np.zeros((len(concepts), min_size))
    for i, c in enumerate(concepts):
        data.extend(activations[c][layer_name][:min_size].reshape(min_size, -1))
        concept_labels[i * min_size, (i + 1) * min_size] = i
    data = np.array(data)
    return data, concept_labels


class CAV(object):
    def __init__(self, concepts, layer_name, lr, model_type):
        self.concepts = concepts
        self.layer_name = layer_name
        self.lr = lr
        self.model_type = model_type

    def train(self, activations):
        data, labels = flatten_activations_and_get_labels(self.concepts, self.layer_name, activations)
        assert self.model_type in ['linear', 'logistic']
        if self.model_type == 'linear':
            model = SGDClassifier(alpha=self.lr)
        else:
            model = LogisticRegression()

        x_train, x_test, y_train, y_test, _ = train_test_split(data, labels, test_size=0.2, stratify=labels)
        model.fit(x_train, y_train)

        if len(model.coef_) == 1:
            self.cav = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cav = np.array(model.coef_)

    def get_cav(self):
        return self.cav

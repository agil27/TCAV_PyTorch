import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Resnet18
from tcav import TCAV
from model_wrapper import ModelWrapper
from mydata import MyDataset
import os
import prettytable as pt

def data_loader(base_path):
    image_dataset_train = MyDataset(base_path, transform=data_transforms)
    train_loader = DataLoader(image_dataset_train, batch_size=1)
    return train_loader


def train():
    best_weights = model.state_dict()
    best_acc = 0.0
    for epoch in range(200):
        # test phase
        total = 0
        score = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.max(dim=1)[1]
                total += labels.size(0)
                score += predicted.eq(labels).sum().item()
        acc = score / total
        print("epoch: {}\tacc: {}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict()

        # train phase
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # save model parameters
    torch.save(best_weights, 'resnet18_office.pth')


def validate(model):
    model.eval()
    weights = torch.load('../resnet18_office.pth')
    model.load_state_dict(weights)
    model = ModelWrapper(model, ['feature_layers'])

    num_classes = 2
    scorer = TCAV(model, validloader, concept_dict, num_classes, 20)
    print('Generating concepts...')
    # scorer.generate_activations(['feature_layers'])
    scorer.load_activations()
    print('Concepts successfully generated and loaded!')
    scorer.generate_cavs('feature_layers')
    scorer.calculate_tcav_score('feature_layers')
    scores = scorer.get_tcav_score()
    table = pt.PrettyTable()
    table.field_names = list(concept_dict.keys())
    for i in range(num_classes):
        table.add_row(scores[i])
    print(table)


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_transforms = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    image_dataset = datasets.ImageFolder('../data/amazon', data_transforms)
    train_size = int(len(image_dataset) * 0.8)
    train_data, test_data = torch.utils.data.random_split(image_dataset, [train_size, len(image_dataset) - train_size])
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=8)
    validloader = DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4)

    concept_dict = {}
    for dirname in os.listdir('../data/concepts'):
        fullpath = os.path.join('../data/concepts', dirname)
        if os.path.isdir(fullpath):
            concept_dict[dirname] = data_loader(fullpath)

    model = Resnet18(output_num=31)
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    # train()
    validate(model)

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18
import os
from tcav import TCAV

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

model = resnet18()
model = model.cuda()
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())


def train():
    best_weights = model.state_dict()
    best_acc = 0.0
    for epoch in range(1, 201):
        model.train()
        # train phase
        for inputs, targets in trainloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # test phase
        total, score = 0
        with torch.no_grad():
            model.eval()
            for inputs, targets in testloader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                predicted = outputs.max(dim=1)[1]
                total += targets.size(0)
                score += predicted.eq(targets).sum().item()
        acc = score / total
        if acc > best_acc:
            best_acc = acc
            best_weights = model.state_dict()

    # save model parameters
    torch.save(best_weights, 'resnet18_cifar10.pth')


def validate():
    model.eval()
    model.load_state_dict(torch.load('resnet18_cifar10.pth'))
    # TODO: Create DataLoaders for Broden concepts and train TCAV

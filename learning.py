import os

import torch
import matplotlib.pyplot as plt
import torchvision.io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from torchvision.transforms import Lambda
import torchvision.models as models
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image

def device_check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def get_data():

    training_data = datasets.MNIST(
        root= "data",
        train = True,
        download=True,
        transform=transforms.ToTensor()

    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    return training_data, test_data

def Hyperparameters():
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    return learning_rate, batch_size, epochs

def Loss_Function(pred= None, y = None):
    loss_fn = nn.CrossEntropyLoss(pred, y)

    return loss_fn

def save(model):
    torch.save(model, 'model.pth')

def load():
    return torch.load('model.pth')

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current}/{size}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f"prediction: {pred.argmax(1)}")
            print(f"actuall: {y}")

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def custom_check(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            print(y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(f"{correct} = correct")
    print(f"{test_loss} = test_loss")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])

        self.img_dir = img_dir
        if transform:
            self.transform = transform
        if target_transform:
            self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        image = self.transform(img)
        image = image.squeeze()
        label = self.img_labels.iloc[idx, 1]

        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def run_model():

    training_data, test_data = get_data()
    device = device_check()
    model = load()
    # model = NeuralNetwork()

    learning_rate, batch_size, epochs = Hyperparameters()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    print(device, "\n")


    loss_fn = Loss_Function()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    params = list(model.parameters())
    print(len(params))
    print(params[0].size())

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    save(model)


    print("Done")

def custom_Image_run():
    test_data = CustomImageDataset(
        img_dir="data/test",
        annotations_file='data/test/labels.csv',
        transform= transforms.Compose([transforms.ToTensor(),
                                       transforms.Grayscale()])
    )

    train_data, tst = get_data()

    test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = NeuralNetwork()
    learning_rate, batch_size, epochs = Hyperparameters()
    loss_fn = Loss_Function()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)


def main():
    # run_model()
    custom_Image_run()




if __name__ == "__main__":
    main()

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import nn


training_data = datasets.MNIST(
    root="data",
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

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",

}

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

def runc():
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3,3
    for i in range(1, cols*rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows,cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

def tesr():
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    print(f"Label: {label}")
    plt.show()

def device_check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device



def main():
    device = device_check()
    model = NeuralNetwork().to(device)
    print(model)

    X = torch.rand(1, 28,28, device= device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    input_image = torch.rand(3, 28, 28)
    flatten = nn.Flatten()
    flat_image = flatten(input_image)

    layer1 = nn.Linear(in_features=28*28, out_features= 20)
    hidden1 = layer1(flat_image)
    hidden1 = nn.ReLU()(hidden1)

if __name__ == "__main__":
    main()
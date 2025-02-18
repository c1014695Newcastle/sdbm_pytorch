import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import os
from tqdm import tqdm

encoder_epochs = 100
classes = [0, 1]

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

encoder_path = '../models/encoder.pth'

class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder, self).__init__()
        # Encoding from 784 (32*32) to an arbitary number of features
        # self.encoder = nn.Linear(784, encoding_dim)

        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

        # Encoding to 784 (32*32) to an arbitary number of features
        # self.decoder = nn.Linear(encoding_dim, 784)

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784)
        )

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.encoder(x))
        x = self.decoder(x)
        return x


def train_autoencoder(epoch, train_loader, device, optimizer, encoder, criterion):
    train_loss = 0.0

    for data in train_loader:
        images, _ = data

        images = images.view(images.size(0), -1)
        images = images.to(device)

        optimizer.zero_grad()

        outputs = encoder(images)

        loss = criterion(outputs, images)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))


def make_encoded_dataset(num_dimensions, device, batch_size=64):
    encoder = AutoEncoder(num_dimensions).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder.parameters())

    mnist_trainset = datasets.MNIST(
        "../data/Dataset",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    mnist_testset = datasets.MNIST(
        "../data/Dataset",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_indices = [i for i, label in enumerate(mnist_trainset.targets) if label in classes]
    test_indices = [i for i, label in enumerate(mnist_testset.targets) if label in classes]

    train_subset = Subset(mnist_trainset, train_indices)
    test_subset = Subset(mnist_testset, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False
    )

    if os.path.exists(encoder_path):
        print('Pre-trained Encoder Exists!')
        encoder.load_state_dict(torch.load(encoder_path))
    else:
        # Run the training and testing for defined epochs
        for epoch in range(encoder_epochs):
            train_autoencoder(epoch, train_loader, device, optimizer, encoder, criterion)
        torch.save(encoder.state_dict(), encoder_path)
    encoder.eval()

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    with torch.no_grad():
        for data, label in tqdm(train_loader, desc="Processing training data", unit="data"):
            # Get the encoded features (latent space representation)
            encoded = encoder.encoder(data.view(-1, 28 * 28).to(device))
            train_features.append(encoded.to('cpu'))
            train_labels.append(label)

        for data, label in tqdm(test_loader, desc="Processing training data", unit="data"):
            # Get the encoded features (latent space representation)
            encoded = encoder.encoder(data.view(-1, 28 * 28).to(device))
            test_features.append(encoded.to('cpu'))
            test_labels.append(label)

    train_features =  torch.cat(train_features, dim=0)
    test_features = torch.cat(test_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    full = np.concat([train_features.numpy(), test_features.numpy()])
    min_val_X = full.min(axis=0)
    max_val_X = full.max(axis=0)

    train_scaled = (train_features - min_val_X) / (max_val_X - min_val_X)
    test_scaled = (test_features - min_val_X) / (max_val_X - min_val_X)

    return train_scaled, test_scaled, train_labels, test_labels


if __name__ == '__main__':
    encoded_trainloader, encoded_testloader = make_encoded_dataset(30, torch.device('cpu'), 64)

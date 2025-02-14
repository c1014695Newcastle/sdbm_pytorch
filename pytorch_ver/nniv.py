import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NNInv(nn.Module):
    def __init__(self, X, X_2d, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.epochs = 5
        self.loss = nn.MSELoss()
        self.device = device
        self.m = nn.Sequential(
            nn.Linear(2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, X.shape[1]),
            nn.Sigmoid(),
        )
        self.optimiser = optim.Adam(self.m.parameters(), lr=0.001)
        self.m.to(self.device)
        self.train_model(X, X_2d)

    def forward(self, x):
        return self.m(x)

    def train_model(self, X, X_2d):
        # Convert numpy arrays to PyTorch tensors
        plt.scatter(X_2d[:, 0], X_2d[:, 1])
        plt.xticks([])
        plt.yticks([])
        plt.show()


        min_val_X = X.min(axis=0)
        max_val_X = X.max(axis=0)
        min_val_X2d = X_2d.min(axis=0)
        max_val_X2d = X_2d.max(axis=0)
        X_scaled = (X - min_val_X) / (max_val_X - min_val_X)
        X_2d_scaled = (X_2d - min_val_X2d) / (max_val_X2d - min_val_X2d)

        plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1])
        plt.xticks([])
        plt.yticks([])
        plt.show()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        X_2d_tensor = torch.tensor(X_2d_scaled, dtype=torch.float32)
        # Create a DataLoader
        dataset = TensorDataset(X_2d_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        print('Training Model...\n')
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimiser.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.m(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(dataloader)}")

    def inverse(self, X_2d):
        self.eval()
        with torch.no_grad():
            X_2d_tensor = torch.tensor(X_2d, dtype=torch.float32)
            predictions = self.m(X_2d_tensor)
        return predictions.numpy()


if __name__ == '__main__':
    print('=== TESTING NNInv ===')
    X = np.load('../data/encoded_mnist_full.npy')
    X_2d = np.load('../data/dimensionally_reduced_points_full.npy')
    full = np.concatenate((X, X_2d), axis=1)
    test = NNInv(X, X_2d)

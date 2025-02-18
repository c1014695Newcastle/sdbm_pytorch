import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    def __init__(self, X):
        super().__init__()
        self.epochs = 5
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

    def forward(self, x):
        return self.m(x)

    def inverse(self, X_2d):
        self.eval()
        with torch.no_grad():
            X_2d_tensor = torch.tensor(X_2d, dtype=torch.float32)
            predictions = self(X_2d_tensor)
        return predictions.numpy()

def train_model(model, X, X_2d, device, optimiser, epochs, loss_fn):
    best_loss = float('inf')
    best_model_weights = None
    patience = 20

    min_val_X = X.min(axis=0)
    max_val_X = X.max(axis=0)
    min_val_X2d = X_2d.min(axis=0)
    max_val_X2d = X_2d.max(axis=0)
    X_scaled = (X - min_val_X) / (max_val_X - min_val_X)
    X_2d_scaled = (X_2d - min_val_X2d) / (max_val_X2d - min_val_X2d)

    plt.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1])
    plt.title('Scaled Points')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X_2d_scaled, X_scaled, test_size=0.2, random_state=SEED)

    # Create a DataLoader
    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    print('Training Model...\n')
    training_loss = []
    testing_loss = []
    final_epoch = 0
    for epoch in range(epochs):
        running_loss = 0.0

        model.train()
        for inputs, labels in trainloader:
            optimiser.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f'TRAINING: Epoch: {epoch + 1:2}/{epochs}... Loss:{running_loss / len(trainloader):10.7f}')
        training_loss.append(running_loss / len(trainloader))
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        avg_loss = val_loss / len(testloader)
        testing_loss.append(avg_loss)
        print(f'TESTING: Epoch: {epoch + 1:2}/{epochs}... Loss: {avg_loss:10.7f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = model.state_dict()
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                final_epoch = epoch
                break

    epochs = range(1, final_epoch + 2)
    plt.plot(epochs, training_loss, 'r', epochs, testing_loss, 'b')
    plt.title('NNInv Test/Train Loss')
    plt.show()
    model.load_state_dict(best_model_weights)


if __name__ == '__main__':
    print('=== TESTING NNInv ===')
    X = np.load('../data/encoded_mnist_full.npy')
    X_2d = np.load('../data/dimensionally_reduced_points_full.npy')
    full = np.concatenate((X, X_2d), axis=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = NNInv(X).to(device)
    optimiser = optim.Adam(test.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_model(test, X, X_2d, device, optimiser, epochs=300, loss_fn=loss_fn)
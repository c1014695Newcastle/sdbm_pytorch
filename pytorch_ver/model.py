import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

from encoded_dataset import make_encoded_dataset


class Net(nn.Module):
    def __init__(self, num_dimensions=30, neuron_number=100):
        super().__init__()
        self.loss_fn = nn.BCELoss()
        self.fc_layers = nn.Sequential(
            nn.Linear(num_dimensions, neuron_number),
            nn.ReLU(),

            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(neuron_number, 1),
        )
        self.final_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_layers(x)
        return self.final_layer(x)

    def compute_confidence(self, x):
        x = torch.Tensor(x)
        x = self.fc_layers(x)
        return self.final_layer(x).detach().numpy()[0]

    def compute_class(self, x):
        x = torch.Tensor(x)
        x = self.fc_layers(x)
        return int(self.final_layer(x).detach().numpy()[0] > 0.5)

def train_net(model, device, train_loader, optimizer, epoch):
    model.to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.unsqueeze(1).float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


def test_net(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.unsqueeze(1).float().to(device)
            output = model(data)
            test_loss += model.loss_fn(output, target).item()  # sum up batch loss
            pred = (output > 0.5).float()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.3f}%)\n"
    )

if __name__ == '__main__':
    print('=== Testing Model ===')
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset, train_labels, test_labels = make_encoded_dataset(30, torch.device('cpu'), 64)

    train_dataset_tensor = torch.utils.data.TensorDataset(train_dataset, train_labels)
    test_dataset_tensor = torch.utils.data.TensorDataset(test_dataset, test_labels)
    encoded_trainloader = torch.utils.data.DataLoader(train_dataset_tensor, batch_size=64, shuffle=False)
    encoded_testloader = torch.utils.data.DataLoader(test_dataset_tensor, batch_size=64, shuffle=False)
    print('Training classifier...')
    classifier = Net()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(1, num_epochs + 1):
        train_net(classifier, device, encoded_trainloader, optimizer, epoch)
        test_net(classifier, device, encoded_testloader)
    classifier.to('cpu')

    data = next(iter(encoded_testloader))[0]
    print(data.shape)

    print('Class Confidence:')
    print(classifier.compute_confidence(data))

    print('\nClasses:')
    print(classifier.compute_class(data))



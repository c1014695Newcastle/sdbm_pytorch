import numpy as np
import os

from pathlib import Path

from sympy import Inverse

script_path = Path(__file__).parent
os.chdir(script_path)

import pickle
from time import time
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import cartesian
from tqdm import tqdm
import umap
import random
import torch
import torch.optim as optim

from model import Net, train_net, test_net
from encoded_dataset import make_encoded_dataset
from ssnp_pytorch import PtSSNP
import matplotlib.pyplot as plt

from nniv import NNInv

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

output_dir = '../data/results_pytorch'

classifier = Net(30, 100)

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [5, 5]
    patience = 5
    num_epochs = 10
    grid_size = 100

    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_delta = 0.05

    verbose = False
    results = []

    #os.makedirs(output_dir, exist_ok=True)
    print('\nMaking dataset...')
    train_dataset, test_dataset, train_labels, test_labels = make_encoded_dataset(30, torch.device('cpu'), 64)

    train_dataset_tensor = torch.utils.data.TensorDataset(train_dataset, train_labels)
    test_dataset_tensor = torch.utils.data.TensorDataset(test_dataset, test_labels)
    encoded_trainloader = torch.utils.data.DataLoader(train_dataset_tensor, batch_size=64, shuffle=False)
    encoded_testloader = torch.utils.data.DataLoader(test_dataset_tensor, batch_size=64, shuffle=False)

    umap_reducer = umap.UMAP()
    reduced_points = umap_reducer.fit_transform(train_dataset)

    print('\nTraining classifier...')
    classifier = Net()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(1, num_epochs + 1):
        train_net(classifier, device, encoded_trainloader, optimizer, epoch)
        test_net(classifier, device, encoded_testloader)
    classifier.to('cpu')

    # Create the axes intervals
    print('\nMaking Map...')

    x_intrvls_umap = np.linspace(0, 1, num=grid_size)
    y_intrvls_umap = np.linspace(0, 1, num=grid_size)
    pts_umap = cartesian((x_intrvls_umap, y_intrvls_umap))

    print('X Shape:',x_intrvls_umap.shape)
    print('Y Shape:', y_intrvls_umap.shape)
    print('Map Points Shape', pts_umap.shape)

    print('\nMaking Inverse Points...')
    print(type(train_dataset))
    inverse_function = NNInv(train_dataset.numpy(), reduced_points).to('cpu')
    classification_grid = np.zeros(((grid_size, grid_size)))
    confidence_grid = np.zeros(((grid_size, grid_size)))

    row = 0
    column = 0
    for point in tqdm(pts_umap, desc="Processing Grid Points", unit="point"):
        point_inverse = inverse_function.inverse(X_2d=point)
        point_class = classifier.compute_class(point_inverse)
        point_conf = classifier.compute_confidence(point_inverse)
        classification_grid[row, column] = point_class
        confidence_grid[row, column] = point_conf
        row += 1
        if row == grid_size:
            column += 1
            row = 0

    plt.imshow(classification_grid, cmap="Set1")
    plt.title("Classification Grid")
    plt.axis("off")
    plt.show()

    plt.imshow(confidence_grid, cmap="plasma")
    plt.title("Confidence Grid")
    plt.xticks([])
    plt.yticks([])
    plt.show()


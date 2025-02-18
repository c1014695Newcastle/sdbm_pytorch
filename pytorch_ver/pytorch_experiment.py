import numpy as np
import os
import pandas as pd
from pathlib import Path

from numba.core.ir import Print
from sympy import Inverse

script_path = Path(__file__).parent
os.chdir(script_path)

import pickle
import torch.nn as nn
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
import argparse, warnings

from nniv import NNInv, train_model
from sklearn.manifold import TSNE

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
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Config for making SDBMs")
parser.add_argument("-r", "--reduction", default='umap', choices=['umap','tsne'], help="dimensionality reduction technique to use")
parser.add_argument("-s", "--size", default=300, type=int, help="size of the image to generate")


if __name__ == "__main__":
    args = parser.parse_args()
    classifer_path = '../models/classifier.pth'
    changed_classifier_path = '../models/classifier_changed.pth'

    if args.reduction == 'umap':
        nninv_path = '../models/nninv_umap.pth'
        reducer = umap.UMAP(random_state=SEED)
    else:
        reducer = TSNE(n_components=2, random_state=SEED, metric='cosine')
        nninv_path = '../models/nninv_tsne.pth'

    grid_size = args.size

    plt.rcParams["figure.figsize"] = [10, 10]
    patience = 5
    num_epochs = 100

    device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_delta = 0.05

    verbose = False
    results = []

    print('\nMaking dataset...')
    train_dataset, test_dataset, train_labels, test_labels = make_encoded_dataset(30, device, 16)

    train_dataset_tensor = torch.utils.data.TensorDataset(train_dataset, train_labels)
    test_dataset_tensor = torch.utils.data.TensorDataset(test_dataset, test_labels)
    encoded_trainloader = torch.utils.data.DataLoader(train_dataset_tensor, batch_size=16, shuffle=False)
    encoded_testloader = torch.utils.data.DataLoader(test_dataset_tensor, batch_size=16, shuffle=False)


    #umap_reducer = TSNE(n_components=2, random_state=SEED, metric='cosine')
    reduced_points = reducer.fit_transform(train_dataset)

    print('\nTraining classifier...')
    classifier = Net().to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.5)
    if os.path.exists(classifer_path):
        print('Pre-trained Classifier Exists!')
        classifier.load_state_dict(torch.load(classifer_path))
    else:
        train_loss = []
        test_loss = []
        for epoch in range(1, num_epochs + 1):
            train_loss.append(train_net(classifier, device, encoded_trainloader, optimizer, epoch))
            test_loss.append(test_net(classifier, device, encoded_testloader))
        torch.save(classifier.state_dict(), classifer_path)
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs[10:], train_loss[10:], 'r', epochs[10:], test_loss[10:], 'b')
        plt.title('Model Test/Train Loss')
        plt.show()
    classifier.eval()


    threshold = 0.2  # Threshold neuron activations should be above to be considered important
    cluster_points = pd.read_csv('../data/Neuron_Cluster_1_Input_Layer.csv').drop(
        ['ind', 'x', 'y', 'activation_0', 'activation_1', 'activation_2', 'activation_3', 'activation_4',
         'activation_5', 'activation_6', 'activation_7', 'activation_8', 'activation_9'], axis=1)
    important_neurons = cluster_points[cluster_points['average_activation'] > threshold].index
    important_neuron_vectors = pd.read_csv('../data/binary_vectors_cluster_1_input_layer_11-2-25.csv').iloc[
        important_neurons].to_numpy()

    classifier_changed = Net()
    classifier_changed.load_state_dict(classifier.state_dict())

    weights = classifier_changed.state_dict()
    weights_np = weights['fc_layers.0.weight'].numpy()

    increment = 0.5

    for i in range(0, len(important_neurons)):
        vector = important_neuron_vectors[i]
        for x in range(0, 30):
            if vector[x]:
                weights_np[i][x] += increment

    weights['fc_layers.0.weight'] = torch.Tensor(weights_np)
    classifier_changed.load_state_dict(weights)

    print('BEFORE ALTERATIONS:')
    test_net(classifier.to(device), device, encoded_testloader)

    print('AFTER ALTERATIONS:')
    test_net(classifier_changed.to(device), device, encoded_testloader)

    classifier.to('cpu')
    classifier_changed.to('cpu')

    # Create the axes intervals
    print('\nMaking Map...')

    x_intrvls_umap = np.linspace(0, 1, num=grid_size)
    y_intrvls_umap = np.linspace(0, 1, num=grid_size)
    pts_umap = cartesian((x_intrvls_umap, y_intrvls_umap))

    inverse_function = NNInv(train_dataset.numpy()).to(device)
    optimiser = optim.Adam(inverse_function.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    if os.path.exists(nninv_path):
        print('Pre-trained NNInv Exists!')
        inverse_function.load_state_dict(torch.load(nninv_path))
    else:
        train_model(inverse_function, train_dataset.numpy()[0:2000], reduced_points[0:2000], device, optimiser, epochs=300, loss_fn=loss_fn)
        torch.save(inverse_function.state_dict(), nninv_path)
    inverse_function.eval()
    inverse_function.to('cpu')

    classification_grid = np.zeros(((grid_size, grid_size)))
    confidence_grid = np.zeros(((grid_size, grid_size)))

    changed_classification_grid = np.zeros(((grid_size, grid_size)))
    changed_confidence_grid = np.zeros(((grid_size, grid_size)))

    row = 0
    column = 0
    for point in tqdm(pts_umap, desc="Processing Grid Points", unit="point"):
        point_inverse = inverse_function.inverse(X_2d=point)

        point_class = classifier.compute_class(point_inverse)
        point_conf = classifier.compute_confidence(point_inverse)
        classification_grid[row, column] = point_class
        confidence_grid[row, column] = point_conf

        c_point_class = classifier_changed.compute_class(point_inverse)
        c_point_conf = classifier_changed.compute_confidence(point_inverse)
        changed_classification_grid[row, column] = c_point_class
        changed_confidence_grid[row, column] = c_point_conf

        row += 1
        if row == grid_size:
            column += 1
            row = 0

    fig, axs = plt.subplots(2, 2)

    axs[0,0].imshow(classification_grid, cmap="tab10")
    axs[0,0].set_title("Default")
    axs[0,1].imshow(changed_classification_grid, cmap="tab10")
    axs[0,1].set_title("Changed")

    axs[1,0].imshow(confidence_grid, cmap="plasma")
    axs[1,0].set_title("Default")
    axs[1,1].imshow(changed_confidence_grid, cmap="plasma")
    axs[1,1].set_title("Changed")
    plt.show()

    print('Done!\n')


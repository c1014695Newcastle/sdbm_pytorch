import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

from sklearn.preprocessing import LabelBinarizer


class PtSSNP(nn.Module):

    def __init__(self, X, y):
        super().__init__()
        self.is_fitted = False
        self.label_bin = LabelBinarizer()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=X.shape[1], out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=X.shape[1])
        )
        n_classes = len(np.unique(y))
        if n_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(512, 1),  # Binary classification
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, n_classes),  # Multi-class classification
                nn.Softmax(dim=1)
            )
        self.is_fitted = True

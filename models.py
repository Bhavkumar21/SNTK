import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module):
    """Simple neural network model with one hidden layer"""
    def __init__(self, width=128, alpha=1.0):
        super(SimpleNN, self).__init__()
        self.width = width
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, 1)
        
        # Initialize with NTK parameterization
        nn.init.normal_(self.fc1.weight, std=1.0/np.sqrt(width))
        nn.init.normal_(self.fc2.weight, std=1.0/np.sqrt(width))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

        # Scale output layer more aggressively
        with torch.no_grad():
            self.fc2.weight *= 1.0/alpha
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

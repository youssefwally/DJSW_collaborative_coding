# -----------------------------------------------------
#Imports

import torch
import torch.nn as nn
# -----------------------------------------------------
#Class

class SMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron with 3 hidden layers (77 neurons each).
    
    Args:
        input_size (int): Dimension of input features (s)
        hidden_size (int): Number of neurons in each hidden layer (default: 77)
        output_size (int): Dimension of output (default: 6)
    """
    def __init__(self, input_size=784, hidden_size=77, output_size=6):
        super(SMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x
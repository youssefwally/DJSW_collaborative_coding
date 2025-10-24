# -----------------------------------------------------
#Imports

import torch
import torch.nn as nn
# -----------------------------------------------------
#Class

class WMLP(nn.Module):
    """
    Multi-Layer Perceptron with 2 hidden layers (100 neurons each).
    
    Args:
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output
        hidden_dim (int): Number of neurons in each hidden layer (default: 100)
        negative_slope (float): Controls the angle of the negative slope for LeakyReLU (default: 0.01)
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=100, negative_slope=0.01):
        super(WMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        return x
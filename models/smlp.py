# -----------------------------------------------------
#Imports

import torch
import torch.nn as nn
# -----------------------------------------------------
#Class

class SMLP(nn.Module):
    """Simple Multi-Layer Perceptron used in the project.

    The network has four linear layers. The first three are followed by ReLU
    activations and the final layer returns logits. The implementation expects
    flattened inputs of shape ``(batch_size, input_size)`` (for MNIST: input_size=784).

    Parameters
    ----------
    input_size : int
        Dimensionality of the flattened input (default: 784 for 28x28 images).
    hidden_size : int
        Number of neurons in the hidden layers (default: 77 in this project).
    output_size : int
        Number of output classes (default: 6 for digits 4..9 remapped to 0..5).
    """
    def __init__(self, input_size=784, hidden_size=77, output_size=6):
        super(SMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size). The function does
            not perform flattening; caller must supply a flattened tensor.

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch_size, output_size).
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x
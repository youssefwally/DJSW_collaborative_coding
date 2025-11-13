import torch
import torch.nn as nn

class DMLP(nn.Module):
    """
    MLP with 3 hidden layers and 300 neurons in each layer.
    LeakyReLU activation function.
    Args:
        input_dim: input dimension
        output_dim: output dimension
        hidden_dim: number of neurons in each hidden layer, default 300
        negative_slope: negative slope of LeakyReLU activation function, default 0.01
    """
    def __init__(self, input_dim, output_dim, hidden_dim=300, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

        # 3 hidden layers of 300 units each
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),

            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for LeakyReLU
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=self.negative_slope, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Compute forward pass.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns: Output tensor of shape (N, output_dim).
        """
        return self.net(x)

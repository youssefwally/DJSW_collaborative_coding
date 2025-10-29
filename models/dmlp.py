import torch
import torch.nn as nn

class DMLP(nn.Module):
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
        # He/Kaiming init for LeakyReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=self.negative_slope, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)  # logits

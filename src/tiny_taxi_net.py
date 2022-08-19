import torch
import torch.nn as nn

class TinyTaxiNet(nn.Module):
    def __init__(self):
        super(TinyTaxiNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            
            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 8),
            nn.ReLU(),
            
            nn.Linear(8, 2),
        )
    
    def forward(self, x):
        return self.main(x)
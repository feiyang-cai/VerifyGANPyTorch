from src.smaller_generator import SmallerGenerator
from src.tiny_taxi_net import TinyTaxiNet
import torch.nn as nn
import torch

class AllInOne(nn.Module):
    def __init__(self):
        super(AllInOne, self).__init__()

        self.smaller_generator = SmallerGenerator(2, 2)
        self.tiny_taxi_net = TinyTaxiNet()
        self.controller = nn.Linear(2, 1, bias=False)
    
    def forward(self, z, s):
        x = self.smaller_generator(z, s)
        x = (x + 1.0) / 2.0
        a = self.controller(self.tiny_taxi_net(x.view(-1, 128)))
        return a

if __name__ == "__main__":
    net = AllInOne()
    z = torch.randn(128, 2)
    s = torch.randn(128, 2)
    print(net(z, s).shape)


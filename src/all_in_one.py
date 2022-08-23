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
        self.denomalizer = nn.Linear(128, 128)
    
    def forward(self, x):
        x = self.smaller_generator.main(x)
        x = self.denomalizer(x)
        #x = self.tiny_taxi_net(x)
        a = self.controller(self.tiny_taxi_net(x))
        return a

if __name__ == "__main__":
    net = AllInOne()
    x = torch.randn(128, 4)
    print(net(x).shape)


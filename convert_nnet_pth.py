from src.nnet import NNet
from src.tiny_taxi_net import TinyTaxiNet
import torch

nnet = NNet("./models/nnet/TinyTaxiNet.nnet")
taxi_net = TinyTaxiNet()
for idx, (name, param) in enumerate(taxi_net.named_parameters()):
    if idx % 2 == 0:
        # weights
        nnet_param = nnet.weights[idx//2]
        param.data = torch.FloatTensor(nnet_param)
    else:
        # bias
        nnet_param = nnet.biases[idx//2]
        param.data = torch.FloatTensor(nnet_param)

taxi_net.eval()

torch.save(taxi_net.state_dict(), './models/nnet/TinyTaxiNet.pth')

taxi_net.load_state_dict(torch.load("./models/nnet/TinyTaxiNet.pth"))

import h5py
import numpy as np
import os

root_dir = "./data"

with h5py.File(os.path.join(root_dir, "SK_DownsampledGANFocusAreaData.h5"), 'r') as f:
    y = f.get('X_train')
    y = np.array(y, dtype=np.float32)
    images = f.get('y_train')
    images = np.array(images, dtype=np.float32)

x = images[1].flatten()
print(nnet.evaluate_network(x))
x = torch.tensor(x)
print(taxi_net(x.view(1, -1)))




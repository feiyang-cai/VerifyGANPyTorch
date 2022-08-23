from src.all_in_one import AllInOne
import torch

net = AllInOne()
net.smaller_generator.load_state_dict(torch.load("./smaller_outputs/smaller_generator.pth"))
net.tiny_taxi_net.load_state_dict(torch.load("./models/nnet/TinyTaxiNet.pth"))
net.controller.weight.data = torch.FloatTensor([[-0.74, -0.44]])
net.denomalizer.weight.data = torch.FloatTensor(torch.eye(128)*0.5)
net.denomalizer.bias.data = torch.FloatTensor(torch.ones_like(net.denomalizer.bias)*0.5)

torch.save(net.state_dict(), "./models/allinone/AllInOne.pth")

x = torch.randn(1, 4)

torch.onnx.export(net, 
                  x,
                  "./models/allinone/AllInOne.onnx",
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names






from src.all_in_one import AllInOne
import torch

net = AllInOne()
net.smaller_generator.load_state_dict(torch.load("./smaller_outputs/smaller_generator.pth"))
net.tiny_taxi_net.load_state_dict(torch.load("./models/nnet/TinyTaxiNet.pth"))
net.controller.weight.data = torch.FloatTensor([[-0.74, -0.44]])
torch.save(net.state_dict(), "./models/allinone/AllInOne.pth")

x = torch.randn(128, 2)
y = torch.randn(128, 2)

torch.onnx.export(net, 
                  (x,y),
                  "./models/allinone/AllInOne.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})






from src.all_in_one import AllInOne
from src.onnx_net import ONNXNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AllInOne().to(device)
net.load_state_dict(torch.load("./models/allinone/AllInOne.pth", map_location=device))
net.eval()

paper_net = ONNXNet().to(device)
paper_net.load_state_dict(torch.load("./models/papermodel/AllInOne.pth", map_location=device))
paper_net.eval()
print(paper_net)
#print(net)
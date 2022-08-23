from src.all_in_one import AllInOne
from src.dynamics import next_state
import torch
import numpy as np
import matplotlib.pyplot as plt

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AllInOne().to(device)
net.load_state_dict(torch.load("./models/allinone/AllInOne.pth", map_location=device))
net.eval()

for initial_crosstrack_error in range(-8, 10, 2):
    s = [initial_crosstrack_error, 0] # 3m, 10 degree
    ts = []
    crosstrack_errors = []
    for i in range(1000):
        ts.append(i*0.05)
        crosstrack_errors.append(s[0])
        s_np = np.array(s)/[6.36615, 17.247995]
        s_torch = torch.FloatTensor(s_np).to(device)
        s_torch = s_torch.view(-1, 2)
        z = torch.FloatTensor(np.random.uniform(-.8, .8, size=(1, 2))).to(device)
        x = torch.cat([z, s_torch], dim=1)
        with torch.no_grad():
            a = net(x)
        s = next_state(s, a)
    plt.plot(ts, crosstrack_errors)

plt.savefig("trajectory.png")
    








from src.verification_utils import Verification
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from src.all_in_one import AllInOne
from src.dynamics import next_state


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AllInOne().to(device)
net.load_state_dict(torch.load("./models/allinone/AllInOne.pth", map_location=device))
net.eval()


veri = Verification()
#veri.get_control_bound_graph()

fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

# plot grids
for p_lb in veri.p_lbs:
    X = [p_lb, p_lb]
    Y = [veri.theta_bins[0], veri.theta_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in veri.theta_lbs:
    Y = [theta_lb, theta_lb]
    X = [veri.p_bins[0], veri.p_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

p_lb = veri.p_lbs[94]
p_ub = veri.p_ubs[94]
theta_lb = veri.theta_lbs[112]
theta_ub = veri.theta_ubs[112]
print(p_lb, p_ub, theta_lb, theta_ub)
ps = np.random.uniform(p_lb, p_ub, [300,1])
thetas = np.random.uniform(theta_lb, theta_ub, [300,1])
states = np.concatenate([ps, thetas], axis=1)

over_range_index, over_range = veri.overapproaximate_dynamics(94, 112)
print(over_range, over_range_index)

states_ = []
for s in states:
    s_np = np.array(s)/[6.36615, 17.247995]
    s_torch = torch.FloatTensor(s_np).to(device)
    s_torch = s_torch.view(-1, 2)
    z = torch.FloatTensor(np.random.uniform(-.8, .8, size=(1, 2))).to(device)
    x = torch.cat([z, s_torch], dim=1)
    with torch.no_grad():
        a = net(x)
    s_ = next_state(s, a)
    states_.append(s_)
states_ = np.array(states_)

    

ax.scatter(states[:,0], states[:,1], color='gray', s=0.2)
ax.scatter(states_[:,0], states_[:,1], color='black', s=0.2)
width = veri.p_lbs[1]-veri.p_lbs[0]
height = veri.theta_lbs[1]-veri.theta_lbs[0]

# initial state cell
ax.add_patch(Rectangle((p_lb, theta_lb), width, height, edgecolor='gray', fill=False))

# overapproximated state
ax.add_patch(Rectangle((over_range[0][0], over_range[1][0]), over_range[0][1]-over_range[0][0], over_range[1][1]-over_range[1][0], alpha=0.1, color="gray"))

# next state cells
xs = np.arange(over_range_index[0][0], over_range_index[0][1]+1)
ys = np.arange(over_range_index[1][0], over_range_index[1][1]+1)
for x in xs:
    for y in ys:
        ax.add_patch(Rectangle((veri.p_lbs[x], veri.theta_lbs[y]), width, height, fill=False, edgecolor="teal"))


ax.set_xticks(veri.p_bins)
ax.set_yticks(veri.theta_bins)
ax.set_xticks([4.6, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0])
ax.set_yticks([12, 16, 20, 24])
ax.set_xlim([4.5, 7])
ax.set_ylim([8, 24])
ax.set_xlabel(r"$p$ (m)")
ax.set_ylabel(r"$\theta$ (degrees)")
plt.savefig("./strange_overapproximated_dynamics.png")

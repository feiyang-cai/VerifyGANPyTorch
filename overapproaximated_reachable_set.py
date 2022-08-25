from src.verification_utils import Verification
import matplotlib.pyplot as plt
import numpy as np

def plot_reachable_map(reachable_map, index):
    fig, ax = plt.subplots()
    ax.imshow(reachable_map.transpose(), cmap="gray")
    ax.set_xticks([0, 127/4, 127*2/4, 127*3/4, 127])
    ax.set_xticklabels([-10, -5, 0, 5, 10])
    ax.set_xlim([0, 127])

    ax.set_yticks([127/60*10, 127/60*30, 127/60*50])
    ax.set_yticklabels([-20, 0, 20])
    ax.set_ylim([0, 127])
    ax.set_xlabel(r"$p$ (m)")
    ax.set_ylabel(r"$\theta$ (degrees)")
    ax.set_title(r"$t = {}$ s".format(index))

    plt.savefig("fig8/{}s.png".format(index))
    plt.close()

veri = Verification()
veri.get_next_step_reachability()

steps = 20

p_start_index = 4
print(veri.p_lbs[p_start_index])
p_end_index = 123
print(veri.p_ubs[p_end_index])
theta_start_index = 42
theta_end_index = 85
reachable_map = np.ones([128, 128])
reachable_map[p_start_index:p_end_index+1, theta_start_index:theta_end_index+1] = 0
plot_reachable_map(reachable_map, 0)


for step in range(1, steps+1):
    reachable_map_ = np.ones([128, 128])
    for idx, x in np.ndenumerate(reachable_map):
        if x == 0:
            over_range_index = veri.one_step_reachability[idx][0]
            p_indexes = np.array(np.arange(over_range_index[0][0], over_range_index[0][1]+1),dtype=int)
            theta_indexes = np.array(np.arange(over_range_index[1][0], over_range_index[1][1]+1),dtype=int)
            reachable_map_[int(over_range_index[0][0]):int(over_range_index[0][1]+1), int(over_range_index[1][0]):int(over_range_index[1][1]+1)]=np.zeros([len(p_indexes), len(theta_indexes)])

            #print(reachable_map_[p_indexes][:,theta_indexes])

    reachable_map = reachable_map_
    plot_reachable_map(reachable_map, step)
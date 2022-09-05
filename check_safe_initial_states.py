import numpy as np
from src.verification_utils import Verification
import matplotlib.pyplot as plt

isSafe = np.ones([128,128])
veri = Verification()
veri.get_next_step_reachability()
veri.get_last_step_reachability()

helper = set() # store the unsafe state
new_unsafe_state = []

# initial check: set the state whose next state is not safe to 0
for i in range(128):
    for j in range(128):
        over_range_index = veri.one_step_reachability[(i, j)][0]
        if over_range_index[0][0] == -1:
            isSafe[(i, j)] = 0
            helper.add((i, j))
            new_unsafe_state.append((i, j))

while len(new_unsafe_state)>0:
    print(len(new_unsafe_state))
    temp = []
    for (i, j) in new_unsafe_state:
        for (_i, _j) in veri.last_step_reachability[(i, j)]:
            if (_i, _j) not in helper:
                temp.append((_i, _j))
                helper.add((_i, _j))
                isSafe[(_i, _j)] = 0
    new_unsafe_state = temp

#print(np.where(isSafe[64:96]==0))
#isSafe[77,112]=1
#isSafe[30+64,112]=1

fig, ax = plt.subplots()
graph = isSafe.transpose()
paddings = np.zeros([128, 10])
graph = np.concatenate([paddings, graph, paddings], axis=1)
print(graph.shape)
ax.imshow(graph, cmap="gray")
ax.set_xticks([0+10, 127/4+10, 127*2/4+10, 127*3/4+10, 127+10])
ax.set_xticklabels([-10, -5, 0, 5, 10])
ax.set_xlim([0, 147])

ax.set_yticks([127/60*10, 127/60*30, 127/60*50])
ax.set_yticklabels([-20, 0, 20])
ax.set_ylim([0, 127])
ax.set_xlabel(r"$p$ (m)")
ax.set_ylabel(r"$\theta$ (degrees)")

plt.savefig("fig7.png")
plt.close()
        

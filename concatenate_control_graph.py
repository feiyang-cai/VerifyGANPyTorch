import numpy as np
import os

path = "verification"
control_bound_graph = []
for file in sorted(os.listdir(path), key=lambda x:int(x.split("_")[-1][:-4])):
    data = np.load(os.path.join(path, file))
    control_bound_graph.append(data)

control_bound_graph = np.array(control_bound_graph)
np.save(os.path.join(path, file[:-8]+".npy"), control_bound_graph)


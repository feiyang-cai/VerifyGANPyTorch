import numpy as np
from src.verification_utils import Verification

veri = Verification()

print(veri.find_control_bound(77, 112))
#print(veri.find_control_bound(94, 112))
#print(veri.find_control_bound(30, 81))

#control_graph = np.load("./verification/control_bound_graph_-10.0_10.0_128_-30.0_30.0_128.npy")
#for row, x in enumerate(control_graph):
#    for col, y in enumerate(x):
#        print(row, col, y)
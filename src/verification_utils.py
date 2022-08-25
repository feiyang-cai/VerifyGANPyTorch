from array import array
import numpy as np
from nnenum import nnenum
from nnenum.specification import Specification
from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
import math
from tqdm import tqdm



class Verification():
    def __init__(self, onnx_filepath="./models/allinone/AllInOne.onnx", control_lb=-10.0, control_ub=+10.0, p_range=[-10, 10], p_num_bin=128, theta_range=[-30, 30], theta_num_bin=128, control_bound_precision=0.1) -> None:
        self.control_lb = control_lb
        self.control_ub = control_ub
        self.p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
        self.p_lbs = np.array(self.p_bins[:-1],dtype=np.float32)
        self.p_ubs = np.array(self.p_bins[1:], dtype=np.float32)

        self.theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
        self.theta_lbs = np.array(self.theta_bins[:-1],dtype=np.float32)
        self.theta_ubs = np.array(self.theta_bins[1:], dtype=np.float32)
        self.control_bound_precision = control_bound_precision
        self.network = nnenum.load_onnx_network_optimized(onnx_filepath)

    def check_property(self, p_index, theta_index, mid, sign):
        init_box = [[-0.8, 0.8], [-0.8, 0.8]]
        p_lb = self.p_lbs[p_index]/6.36615
        p_ub = self.p_ubs[p_index]/6.36615
        theta_lb = self.theta_lbs[theta_index]/17.247995
        theta_ub = self.theta_ubs[theta_index]/17.247995
        init_box.extend([[p_lb, p_ub], [theta_lb, theta_ub]])
        init_box = np.array(init_box, dtype=np.float32)
        Settings.PRINT_OUTPUT = False

        if sign == "<=":
            mat = np.array([[-1.]])
            rhs = np.array([mid*-1])
        elif sign == ">=":
            mat = np.array([[1.]])
            rhs = np.array([mid])
        spec = Specification(mat, rhs)

        nnenum.set_control_settings()
        res = enumerate_network(init_box, self.network, spec)
        result_str = res.result_str
        return True if result_str=="safe" else False

    def find_control_bound(self, p_index, theta_index):
        # binary search the control bound

        ## search the upper bound
        left = self.control_lb
        right = self.control_ub
        while right-left >= self.control_bound_precision:
            mid = (left+right)/2.0
            if self.check_property(p_index, theta_index, mid, "<="): # check if the outputs are guaranteed smaller or equal to "mid"
                right = mid
            else:
                left = mid
        ub = right

        ## search the lower bound (closed)
        left = self.control_lb
        right = ub
        while right-left >= self.control_bound_precision:
            mid = (left+right)/2.0
            if self.check_property(p_index, theta_index, mid, ">="): # check if the output are guaranteed greater or equal to "mid"
                left = mid
            else:
                right = mid
        lb = left

        return np.array([lb, ub], dtype=np.float32)

    def dynamics(self, control_bound, p_bound, theta_bound, steps=20):
        v, L, dt = 5, 5, 0.05
        (control_lb, control_ub) = control_bound
        (p_lb, p_ub) = p_bound
        (theta_lb, theta_ub) = theta_bound

        for step in range(steps): # 1s
            p_lb = p_lb + v*dt*math.sin(math.radians(theta_lb))
            p_ub = p_ub + v*dt*math.sin(math.radians(theta_ub))
            theta_lb = theta_lb + math.degrees(v/L*dt*math.tan(math.radians(control_lb)))
            theta_ub = theta_ub + math.degrees(v/L*dt*math.tan(math.radians(control_ub)))
        
        return (p_lb, p_ub), (theta_lb, theta_ub)
    
    
    def overapproaximate_dynamics(self, p_index, theta_index):
        p_lb = self.p_lbs[p_index]
        p_ub = self.p_ubs[p_index]
        theta_lb = self.theta_lbs[theta_index]
        theta_ub = self.theta_ubs[theta_index]
        if hasattr(self, "control_graph"):
            control_lb, control_ub = self.control_graph[p_index, theta_index]#self.find_control_bound(p_index, theta_index)
        else:
            control_lb, control_ub = self.find_control_bound(p_index, theta_index)

        (p_lb_, p_ub_), (theta_lb_, theta_ub_) = self.dynamics((control_lb, control_ub),
                                                               (p_lb, p_ub),
                                                               (theta_lb, theta_ub))

        overapproximated_range = np.array([[p_lb_, p_ub_], [theta_lb_, theta_ub_]], dtype=np.float32)  

        if p_lb_ < self.p_lbs[0]:
            p_index_lb = -1
        else:
            p_index_lb = np.searchsorted(self.p_lbs, p_lb_, "right")-1
        if p_ub_ > self.p_ubs[-1]:
            p_index_ub = len(self.p_ubs)
        else:
            p_index_ub = np.searchsorted(self.p_ubs, p_ub_)

        if theta_lb_ < self.theta_lbs[0]:
            theta_index_lb = -1
        else:
            theta_index_lb = np.searchsorted(self.theta_lbs, theta_lb_, "right")-1
        if theta_ub_ > self.theta_ubs[-1]:
            theta_index_ub = len(self.theta_ubs)
        else:
            theta_index_ub = np.searchsorted(self.theta_ubs, theta_ub_)

        overapproximated_range_index = np.array([[p_index_lb, p_index_ub], [theta_index_lb, theta_index_ub]])
        
        return overapproximated_range_index, overapproximated_range
    
    def get_control_bound_graph(self):
        control_graph_filepath="verification/control_bound_graph_{}_{}_{}_{}_{}_{}.npy".format(self.p_lbs[0], self.p_ubs[-1], len(self.p_lbs), self.theta_lbs[0], self.theta_ubs[-1], len(self.theta_lbs))
        self.graph = np.zeros([len(self.p_lbs), len(self.theta_lbs)])
        try:
            self.control_graph = np.load(control_graph_filepath)
            print("loaded the preconstructed control graph")
        except:
            print("constructing the control graph...")
            self.control_graph = []
            for p_idx in tqdm(range(len(self.p_lbs))):
                control_graph_line = []
                for theta_idx in tqdm(range(len(self.theta_lbs)), leave=False):
                    control_bound = self.find_control_bound(p_idx, theta_idx)
                    control_graph_line.append(control_bound)
                self.control_graph.append(control_graph_line)
                np.save(control_graph_filepath[:-4]+"_{}.npy".format(p_idx), np.array(control_graph_line))
            self.control_graph = np.array(self.control_graph)
            np.save(control_graph_filepath, self.control_graph)
    
    def get_next_step_reachability(self):
        one_step_reachability_filepath="verification/one_step_reachability_graph_{}_{}_{}_{}_{}_{}.npy".format(self.p_lbs[0], self.p_ubs[-1], len(self.p_lbs), self.theta_lbs[0], self.theta_ubs[-1], len(self.theta_lbs))
        try:
            self.one_step_reachability = np.load(one_step_reachability_filepath)
            print("loaded the preconstructed one step reachability graph.")
        except:
            print("constructing the one step reachability graph...")
            self.get_control_bound_graph()
            # loop through each cell
            self.one_step_reachability = []
            for p_idx in tqdm(range(len(self.p_lbs))):
                one_step_reachability_line = []
                for theta_idx in tqdm(range(len(self.theta_lbs)), leave=False):
                    over_range_index, over_range = self.overapproaximate_dynamics(p_idx, theta_idx)
                    # if reach out of the graph, over_range_index is set to [[-1, -1], [-1, -1]]
                    if over_range_index[0][0] < 0 or over_range_index[1][0] < 0 or over_range_index[0][1] >= len(self.p_lbs) or over_range_index[1][1] >= len(self.theta_lbs):
                        over_range_index = np.array([[-1, -1], [-1, -1]])
                    next_step_reachability = np.array([over_range_index, over_range])
                    one_step_reachability_line.append(next_step_reachability)
                self.one_step_reachability.append(one_step_reachability_line)
            self.one_step_reachability = np.array(self.one_step_reachability)
            print(self.one_step_reachability.shape)
            np.save(one_step_reachability_filepath, self.one_step_reachability)
    
    def get_last_step_reachability(self):
        last_step_reachability_filepath="verification/last_step_reachability_graph_{}_{}_{}_{}_{}_{}.npy".format(self.p_lbs[0], self.p_ubs[-1], len(self.p_lbs), self.theta_lbs[0], self.theta_ubs[-1], len(self.theta_lbs))
        try:
            self.last_step_reachability = np.load(last_step_reachability_filepath, allow_pickle=True)
            print("loaded the preconstructed one step reachability graph.")
        except:
            self.get_next_step_reachability()
            print("constructing the one step reachability graph...")
            self.last_step_reachability = [[ set() for _ in range(len(self.theta_lbs))] for _ in range(len(self.p_lbs))]
            for i in tqdm(range(len(self.p_lbs))):
                for j in tqdm(range(len(self.theta_lbs)), leave=False):
                    last_range_index = self.one_step_reachability[(i, j)][0]
                    if last_range_index[0][0] == -1:
                        continue
                    else:
                        for _p in np.array(np.arange(last_range_index[0][0], last_range_index[0][1]+1), dtype=int):
                            for _theta in np.array(np.arange(last_range_index[1][0], last_range_index[1][1]+1), dtype=int):
                                self.last_step_reachability[_p][_theta].add((i, j))
            np.save(last_step_reachability_filepath, self.last_step_reachability)
        
        
        
if __name__ == "__main__":
    veri = Verification()
    print(veri.p_lbs)
    print(veri.theta_lbs)

    #print(veri.check_property(27, 65, -10, ">="))
    #print(veri.find_control_bound(27, 65))
    #print(veri.overapproaximate_dynamics(27, 65))
    #veri.get_control_bound_graph()
    #veri.get_one_step_reachability()

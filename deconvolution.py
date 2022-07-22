from simulation_engine import SimulationEngine
from hits import Hits
import numpy as np
import matplotlib.pyplot as plt


class Deconvolution:
    def __init__(self, hits: Hits, simulation_engine: SimulationEngine):
        self.hits_dict = hits.hits_dict
        self.det_size_cm = simulation_engine.det_size_cm
        self.n_elements = simulation_engine.n_elements
        self.mura_elements = simulation_engine.mura_elements

    def shift_pos(self):
        # set the size to shift position
        pos_p = self.hits_dict["Position"]
        shift = self.det_size_cm / 2

        # update position to shifted so origin is lower left corner
        pos = [(pp[0] + shift, pp[1] + shift) for pp in pos_p]
        self.hits_dict["Position"] = pos

    def plot_heatmap(self, multiplier):
        # shift first
        self.shift_pos()
        xxes = [p[0] for p in self.hits_dict["Position"]]
        yxes = [p[1] for p in self.hits_dict["Position"]]

        heatmap, xedges, yedges = np.histogram2d(
            xxes, yxes, bins=multiplier * self.mura_elements
        )
        plt.imshow(heatmap.T, origin="lower", cmap="RdBu_r")
        plt.colorbar()
        plt.show()
        plt.close()

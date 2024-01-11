import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "../detector_analysis/")

from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *

results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/"

thick_raw_center = results_dir + "47-2-15/47-2.2-c-xy-0_1.00E+06_Mono_100_raw.txt"
thin_raw_center = results_dir + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_100_raw.txt"

thick_raw_edge = results_dir + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_100_raw.txt"
thin_raw_edge = results_dir + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_100_raw.txt"

thick_dc_center = results_dir + "47-2-15/47-2.2-c-xy-0_1.00E+06_Mono_100_dc.txt"
thin_dc_center = results_dir + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_100_dc.txt"

thick_dc_edge = results_dir + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_100_dc.txt"
thin_dc_edge = results_dir + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_100_dc.txt"

def plot_3d(ax, heatmap):
    x = np.arange(heatmap.shape[0])
    y = np.arange(heatmap.shape[1])
    X, Y = np.meshgrid(x, y)

    # Create the surface plot!
    surface = ax.plot_surface(X, Y, heatmap, cmap=cmap)

    # Add a color bar for reference
    #fig.colorbar(surface)

    # Show the plot
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.set_zlabel("signal")

data = np.random.rand(10, 10)

fig = plt.figure(figsize=(6,8))
grid = (4,3)

ax1 = plt.subplot2grid(grid, (0, 0))
ax1.imshow(np.loadtxt(thin_raw_center))
ax1.set_title("Raw \nImage")
ax1.axis('off')

ax2 = plt.subplot2grid(grid, (0, 1))
ax2.imshow(np.loadtxt(thin_dc_center))
ax2.set_title("Reconstructed \nImage")
ax2.axis('off')

ax3 = plt.subplot2grid(grid, (0, 2), projection="3d")
plot_3d(ax3, np.loadtxt(thin_dc_center)[60:75,65:80])
ax3.view_init(elev=22, azim=23)
ax3.set_title("Reconstructed \nSignal")
ax3.axis('off')

#ax4 = plt.subplot2grid(grid, (0, 3), colspan = 2, rowspan = 2)
#ax4.set_title("FWHM \n")
#fwhm = ax4.imshow(np.loadtxt(results_dir + "47-2-300/fwhm_interp_grid_instrument-only_edges-inc_1d.txt"))
#colorbar = fig.colorbar(fwhm, ax=ax4, pad=0.01,shrink=0.92)
#colorbar.set_label('# Pixels')
#ax4.axis('off')

ax5 = plt.subplot2grid(grid, (1, 0))
ax5.imshow(np.loadtxt(thin_raw_edge))
ax5.axis('off')

ax6 = plt.subplot2grid(grid, (1, 1))
ax6.imshow(np.loadtxt(thin_dc_edge))
ax6.axis('off')

ax7 = plt.subplot2grid(grid, (1, 2), projection="3d")
plot_3d(ax7, np.loadtxt(thin_dc_edge)[2:17,125:140])
ax7.view_init(elev=22, azim=23)
ax7.axis('off')

ax8 = plt.subplot2grid(grid, (2, 0))
ax8.imshow(np.loadtxt(thick_raw_center))
ax8.axis('off')

ax9 = plt.subplot2grid(grid, (2, 1))
ax9.imshow(np.loadtxt(thick_dc_center))
ax9.axis('off')

ax10 = plt.subplot2grid(grid, (2, 2), projection="3d")
plot_3d(ax10, np.loadtxt(thick_dc_center)[60:75,65:80])
ax10.view_init(elev=22, azim=23)
ax10.axis('off')

#ax11 = plt.subplot2grid(grid, (2, 3), colspan = 2, rowspan = 2)
#fwhm = ax11.imshow(np.loadtxt(results_dir + "47-2-15/fwhm_interp_grid_instrument-only_edges-removed_1d.txt"))
#colorbar = fig.colorbar(fwhm, ax=ax11, pad=0.01, shrink=0.92)
#colorbar.set_label('# Pixels')
#ax11.axis('off')

ax12 = plt.subplot2grid(grid, (3, 0))
ax12.imshow(np.loadtxt(thick_raw_edge))
ax12.axis('off')

ax13 = plt.subplot2grid(grid, (3, 1))
ax13.imshow(np.loadtxt(thick_dc_edge))
ax13.axis('off')

ax14 = plt.subplot2grid(grid, (3, 2), projection="3d")
plot_3d(ax14, np.loadtxt(thick_dc_edge)[2:17,125:140])
ax14.view_init(elev=22, azim=23)
ax14.axis('off')

fig.suptitle(r'1.5 mm Aperture        300 $\mu$m Aperture', fontsize=18, va='center', rotation=90, ha='left', x=0.05, y=0.5)

plt.savefig(results_dir+"fwhm.png", dpi=500)

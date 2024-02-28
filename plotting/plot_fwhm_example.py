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

results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/fwhm-figure/"

thin_raw_center = results_dir + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_600_raw.txt"

thick_raw_edge = results_dir + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_6000_raw.txt"
thin_raw_edge = results_dir + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_600_raw.txt"

thin_dc_center = results_dir + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_600_dc.txt"

thick_dc_edge = results_dir + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_6000_dc.txt"
thin_dc_edge = results_dir + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_600_dc.txt"

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


import cmocean

cmap = cmocean.cm.thermal
#lightcmap = cmocean.tools.crop_by_percent(cmap, 10, which='min', N=None)
#cmap =lightcmap

fig = plt.figure(figsize=(5.7,5.8))
grid = (3,3)

max_raw = np.max(np.loadtxt(thin_raw_center))
max_dc = np.max(np.loadtxt(thin_dc_center))


ax1 = plt.subplot2grid(grid, (0, 0))
ax1.imshow(np.loadtxt(thin_raw_center)/max_raw,cmap,vmin=0, vmax=1)
#ax1.set_title("Raw \nImage")
ax1.axis('off')

ax2 = plt.subplot2grid(grid, (0, 1))
ax2.imshow(np.loadtxt(thin_dc_center)/max_dc,cmap,vmin=0, vmax=1)
#ax2.set_title("Reconstructed \nImage")
ax2.axis('off')

ax3 = plt.subplot2grid(grid, (0, 2))
ax3.imshow(np.loadtxt(thin_dc_center)[59:79,62:82]/max_dc,cmap,vmin=0, vmax=1)
#plot_3d(ax3, np.loadtxt(thin_dc_center)[60:75,65:80])
#ax3.view_init(elev=22, azim=23)
#ax3.set_title("Reconstructed \nSignal")
ax3.axis('off')

#ax4 = plt.subplot2grid(grid, (0, 3), colspan = 2, rowspan = 2)
#ax4.set_title("FWHM \n")
#fwhm = ax4.imshow(np.loadtxt(results_dir + "47-2-300/fwhm_interp_grid_instrument-only_edges-inc_1d.txt"))
#colorbar = fig.colorbar(fwhm, ax=ax4, pad=0.01,shrink=0.92)
#colorbar.set_label('# Pixels')
#ax4.axis('off')

ax5 = plt.subplot2grid(grid, (1, 0))
ax5.imshow(np.loadtxt(thin_raw_edge)/max_raw,cmap,vmin=0, vmax=1)
ax5.axis('off')

ax6 = plt.subplot2grid(grid, (1, 1))
ax6.imshow(np.loadtxt(thin_dc_edge)/max_dc,cmap,vmin=0, vmax=1)
ax6.axis('off')

ax7 = plt.subplot2grid(grid, (1, 2))
ax7.imshow(np.loadtxt(thin_dc_edge)[0:19,122:142]/max_dc,cmap,vmin=0, vmax=1)
#ax7.view_init(elev=22, azim=23)
ax7.axis('off')

#ax8 = plt.subplot2grid(grid, (2, 0))
#ax8.imshow(np.loadtxt(thick_raw_center),cmap)
#ax8.axis('off')

#ax9 = plt.subplot2grid(grid, (2, 1))
#ax9.imshow(np.loadtxt(thick_dc_center),cmap)
#ax9.axis('off')

#ax10 = plt.subplot2grid(grid, (2, 2))
#ax10.imshow(np.loadtxt(thick_dc_center)[59:79,62:82],cmap)
#ax10.view_init(elev=22, azim=23)
#ax10.axis('off')

#ax11 = plt.subplot2grid(grid, (2, 3), colspan = 2, rowspan = 2)
#fwhm = ax11.imshow(np.loadtxt(results_dir + "47-2-15/fwhm_interp_grid_instrument-only_edges-removed_1d.txt"))
#colorbar = fig.colorbar(fwhm, ax=ax11, pad=0.01, shrink=0.92)
#colorbar.set_label('# Pixels')
#ax11.axis('off')

ax12 = plt.subplot2grid(grid, (2, 0))
fwhm=ax12.imshow(np.loadtxt(thick_raw_edge)/max_raw,cmap,vmin=0, vmax=1)
ax12.axis('off')

ax13 = plt.subplot2grid(grid, (2, 1))
ff1=ax13.imshow(np.loadtxt(thick_dc_edge)/max_dc,cmap,vmin=0, vmax=1)
ax13.axis('off')

ax14 = plt.subplot2grid(grid, (2, 2))
ff2=ax14.imshow(np.loadtxt(thick_dc_edge)[0:19,122:142]/max_dc,cmap,vmin=0, vmax=1)

#ax14.view_init(elev=22, azim=23)
ax14.axis('off')
ax12.text(-13,70,r'1.5 mm', fontsize=10, va='center', rotation=90, ha='left')
ax5.text(-13,70,r'300 $\mu$m', fontsize=10, va='center', rotation=90, ha='left')

rect = plt.Rectangle((62,59), 18, 18, fill=False, edgecolor='lightgrey', linewidth=1, linestyle='--')
ax2.add_patch(rect)
ax1.text(70,-7,'Raw', fontsize=10, va='center', ha='center')
ax2.text(70,-7,'Decoded', fontsize=10, va='center', ha='center')
ax3.text(9.75,-1.5,'Decoded (zoomed-in)', fontsize=10, va='center', ha='center')

rect = plt.Rectangle((122,0), 18,18, fill=False, edgecolor='lightgrey', linewidth=1, linestyle='--')
ax6.add_patch(rect)
rect = plt.Rectangle((122,0), 18,18, fill=False, edgecolor='lightgrey', linewidth=1, linestyle='--')
ax13.add_patch(rect)


# add the labels
t=ax1.text(15,15,'a)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax2.text(15,15,'b)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax3.text(1.5,1.5,'c)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))

t=ax5.text(15,15,'d)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax6.text(15,15,'e)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax7.text(1.5,1.5,'c)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))

t=ax12.text(15,15,'g)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax13.text(15,15,'h)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))
t=ax14.text(1.5,1.5,'i)', fontsize=10, va='center', ha='center')
t.set_bbox(dict(facecolor='white',edgecolor='white'))

ax12.annotate("", xy=(-1,144), xytext=(142,144),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1),
             annotation_clip=False)
ax13.annotate("", xy=(-1,144), xytext=(142,144),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1),
             annotation_clip=False)
ax14.annotate("", xy=(-.5,19), xytext=(18.5,19),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1),
             annotation_clip=False)
ax12.text(73,152,r"82$^\circ$", fontsize=10, va='center', ha='center')
ax13.text(73,152,r"82$^\circ$", fontsize=10, va='center', ha='center')
ax14.text(10,20.1,r"10$^\circ$", fontsize=10, va='center', ha='center')

#colorbar = fig.colorbar(ff1, ax=ax13, pad=0.1,orientation="horizontal")
#colorbar.set_label('Normalized signal')

fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.033, 0, 0.92, 0.015])
colorbar = fig.colorbar(ff1, cax=cbar_ax, pad=0.1,orientation="horizontal")
cbar_ax.xaxis.labelpad = 0.2
colorbar.set_label('Normalized signal')
cbar_ax.tick_params(axis='x',labelsize=10)

plt.tight_layout()
plt.savefig(results_dir+"4_fwhm.png", dpi=500,bbox_inches='tight',pad_inches=0.02)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
import numpy as np
from plot_settings import *

    """plot on-axis resolution and FOV across F-stop
    """

# loop through rank options (primes)
n_elements_original = [11, 31]  # n elements no mosaic
multipliers = [22, 8]
pixel = 0.055  # mm
det_size_cm = 1.408

element_size_mm_list = [
    pixel * multiplier for multiplier in multipliers
]  # element size in mm
n_elements_list = [
    (ne * 2) - 1 for ne in n_elements_original
]  # total number of elements

mask_size = [ne * ee for ne, ee in zip(n_elements_list, element_size_mm_list)]

# thickness of mask
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness
energies = [0.35, 0.65, 1.2, 3.1, 10]

# set up plot
fig, ax = plt.subplots(2, 1, sharex="col", sharey="row")

params = {'axes.labelsize': 16,
          'axes.titlesize': 16}
plt.rcParams.update(params)

thick_color = hex_colors[::int(np.ceil( len(hex_colors) / len(energies) ))]
linestyles = ['-','dotted']
for ti, thickness in enumerate(thicknesses):
    for ri, rank in enumerate(n_elements_original):

        data_path ="../simulation-results/prospectus/res/res_txt/%d_%d.txt" % (thickness, rank)
        data = np.loadtxt(data_path)
        distance = [d[0]+ ((150 + (thickness/2))*1e-4) for d in data]
        fstop = np.array(distance)/(2*det_size_cm)
        res = [d[1] for d in data]

        ax[0].plot(
            fstop, res, color=thick_color[ti],linestyle=linestyles[ri],linewidth=1.5)
        ax[0].set_yscale('log')

        data_path = "../simulation-results/prospectus/fov/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        res = [2*d[1] for d in data]

        ax[1].plot(
            fstop, res, color=thick_color[ti], linestyle=linestyles[ri],linewidth=1.5)
        #ax[1].set_yscale('log')

        #ax[1,ri].set_xlim([0,5.2])
        #ax[0,ri].set_xlim([0,5.2])
        

        #ax[0,ri].set_xticks([0,1,2,3,4])
        #ax[1,ri].set_xticks([0,1,2,3,4])
        #ax[0].set_yticks([0,25,50,75,100,125,150])
        #ax[1].set_yticks([0,25,50,75,100,125,150])

        #ax[1,ri].set_yscale('log')

#ax[0,0].set_ylim([0, 20])

legend_elements = []
for ti, tc in enumerate(thick_color):
    legend_elements.append(
        Line2D([0], [0], color=tc, lw=2, label="%.2f MeV" % energies[ti])
    )

fig.legend(handles=legend_elements, loc="upper center", ncol=5,prop={'size': 8.5}, bbox_to_anchor=(0.513, 0.95), columnspacing=0.8, fontsize=30)
# more legend elements

ax[0].set_ylabel("on-axis ang. res. [deg]")
ax[1].set_xlabel('F-number')

ax[0].set_xlim([-0.1,6])
ax[1].set_xlim([-0.1,6])

# grid
for tt in [0,1]:
    ax[tt].grid(which="major", color="#DDDDDD", linewidth=0.8)
    ax[tt].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    ax[tt].minorticks_on()

plt.subplots_adjust(hspace=0.075)
ax[1].set_ylabel('FCFOV')
plt.savefig(
    "../simulation-results/prospectus/res_fov.png",bbox_inches='tight',
    dpi=1000
)

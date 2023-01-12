import matplotlib.pyplot as plt
import numpy as np

# loop through rank options (primes)
n_elements_original = [11, 31]  # n elements no mosaic
multipliers = [22, 8]
pixel = 0.055  # mm

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
fig, ax = plt.subplots(2, 2, sharex="col", sharey="row")

params = {'axes.labelsize': 30,
          'axes.titlesize': 16}
plt.rcParams.update(params)

thick_color = ["#03045e","#0077b6","#00b4d8","#90e0ef","#caf0f8"]
thick_color.reverse()
for ti, thickness in enumerate(thicknesses):
    for ri, rank in enumerate(n_elements_original):

        data_path = "../results/parameter_sweeps/timepix_sim/res/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        distance = [d[0] for d in data]
        f = np.array(distance)/1.408
        res = [d[1] for d in data]
        res[0] = np.nan

        ax[0,ri].plot(
            f, res, color=thick_color[ti], marker="o")
        ax[0,ri].set_yscale('log')
        data_path = "../results/parameter_sweeps/timepix_sim/fov/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        distance = [d[0] for d in data]
        res = [2*d[1] for d in data]

        ax[1,ri].plot(
            f, res, color=thick_color[ti], marker="o")
        #ax[1,ri].set_xlim([0,5.2])
        #ax[0,ri].set_xlim([0,5.2])
        ax[0,ri].set_xticks([0,1,2,3,4])
        ax[1,ri].set_xticks([0,1,2,3,4])
        ax[ri,0].set_yticks([0,25,50,75,100,125,150])
        ax[ri,1].set_yticks([0,25,50,75,100,125,150])

        #ax[1,ri].set_yscale('log')



#ax[0,0].set_ylim([0, 20])

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
for ti, tc in enumerate(thick_color):
    legend_elements.append(
        Line2D([0], [0], color=tc, lw=2, label="%.2f MeV" % energies[ti])
    )

fig.legend(handles=legend_elements, loc="upper center", ncol=5)
# more legend elements

ax[0,0].set_ylabel("on-axis ang. res. [deg]")

# grid
for ti in [0,1]:
    for tt in [0,1]:
        ax[ti,tt].grid(which="major", color="#DDDDDD", linewidth=0.8)
        ax[ti,tt].grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
        ax[ti,tt].minorticks_on()

plt.savefig(
    "/home/rileyannereid/workspace/geant4/detector_analysis/plotting/poster_plot.png",
    dpi=1000
)

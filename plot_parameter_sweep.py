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

# set up plot
fig, ax = plt.subplots(3, 1, sharex="col", sharey="row")

colors = ["#9D4EDD", "#5A189A", "#240046"]

thick_color = ["#BCEDF6", "#6A8E7F", "#EAB464", "#A6808C", "#0C6291"]
linestyles = ["dashed", "dotted"]
for ti, thickness in enumerate(thicknesses):
    for ri, rank in enumerate(n_elements_original):

        data_path = "../results/parameter_sweeps/timepix_sim/res/%d_%d.txt" % (
            thickness,
            rank,
        )
        data = np.loadtxt(data_path)
        distance = [d[0] for d in data]
        res = [d[1] for d in data]

        ax[0].plot(
            distance, res, color=thick_color[ti], marker="o", linestyle=linestyles[ri]
        )

distance = np.array(distance)
th_fov1 = 2 * np.rad2deg(
    np.arctan(
        ((mask_size[0] - (pixel * n_elements_original[0] * multipliers[0])) / 2)
        / (10 * distance)
    )
)
th_fov2 = 2 * np.rad2deg(
    np.arctan(
        ((mask_size[1] - (pixel * n_elements_original[1] * multipliers[1])) / 2)
        / (10 * distance)
    )
)


ax[1].plot(distance, th_fov1, color="black", marker="o", linestyle="dashed")
ax[1].plot(distance, th_fov2, color="black", marker="o", linestyle="dotted")

ax[2].set_xlabel("distance between mask and detector [cm]")
ax[0].set_ylim([0, 20])

ax[1].set_ylabel("FCFOV [deg]")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = []
for ti, tc in enumerate(thick_color):
    legend_elements.append(
        Line2D([0], [0], color=tc, lw=2, label="%d um" % thicknesses[ti])
    )


fig.legend(handles=legend_elements, loc="upper center", ncol=5)
# more legend elements
legend_elements = []
legend_elements.append(
    Line2D([0], [0], color="black", lw=2, label="rank 11", linestyle="dashed")
)
legend_elements.append(
    Line2D([0], [0], color="black", lw=2, label="rank 31", linestyle="dotted")
)


ax[0].legend(handles=legend_elements, loc="upper right", ncol=2)

ax[0].set_ylabel("angular resolution")

# grid
for a in ax:
    a.grid(which="major", color="#DDDDDD", linewidth=0.8)
    a.grid(which="minor", color="#EEEEEE", linestyle=":", linewidth=0.5)
    a.minorticks_on()

plt.savefig(
    "/home/rileyannereid/workspace/geant4/results/parameter_sweeps/timepix_sim/param_sweep_tpx_e.png",
    dpi=400,
)

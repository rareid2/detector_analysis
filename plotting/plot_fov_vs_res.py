import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

detector_size_mm = 50
ranks = [
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
]  # 103, 107, 109]
colors = ["#3C2541", "#603B68", "#83528E", "#A271AD", "#BB97C3", "#D5BEDA"]
# colors.reverse()
cmap = LinearSegmentedColormap.from_list("custom", colors, N=int(len(ranks)))
import cmocean

cmap = cmocean.cm.thermal
orange_color = "#F99940"
plum = "#83528E"

fig, ax1 = plt.subplots(figsize=(5.7, 3))
ax2 = ax1.twinx()

for ri, rank in enumerate(ranks):

    element_size_mm = detector_size_mm / rank
    # mask size
    mask_size = element_size_mm * (rank * 2 - 1)

    fovs = []
    ress = []

    dd = np.linspace(0.6, 10)
    for d in dd:
        f = d * 10
        fov = np.arctan(
            ((mask_size * np.sqrt(2) / 2) - (detector_size_mm * np.sqrt(2) / 2)) / f
        )
        res = 2 * np.arctan(element_size_mm / f)
        fovs.append(np.rad2deg(fov))
        ress.append(np.rad2deg(res))

    det_f = dd * 10 / detector_size_mm

    if rank != 26:
        ax1.plot(det_f, ress, color=cmap(rank / 101), linewidth=2)
    else:
        ax1.plot(det_f, ress, color="#56A9F7", linestyle="dashed")


ax2.plot(det_f, fovs, color="black", linewidth=2, linestyle="--")

# formatting
ax2.tick_params(axis="y", labelcolor="black", labelsize=10)
ax2.grid(False)  # ,color=orange_color,linestyle='--', linewidth=0.5)
ax2.set_xlabel("f-number")
ax2.set_ylabel(r"-- Half-Angle FCFOV [$^\circ$]", fontsize=10, color="black")

ax1.set_ylabel(r"Angular Resolution [$^\circ$]", fontsize=10, color="black")
ax1.set_xlabel("f-number", fontsize=10, color="black")
ax1.tick_params(axis="y", labelcolor="black", labelsize=10)
ax1.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
ax2.set_ylim([18, 81])
ax2.set_yticks([round(i) for i in np.linspace(27.5, 77.75, 5)])
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))


ax1.set_ylim([0.5, 10.5])
ax1.set_xlim([0.1, 1.8])
cbar_ax = fig.add_axes([0.8, 0.647, 0.02, 0.2])  # [left, bottom, width, height]
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
cbar.set_ticks([0, 0.33, 0.666, 1])  # Set the positions of the ticks
# [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]

ax1.text(1.55, 8.78, "Rank", fontsize=10, rotation=90, ha="center", va="center")
# ax1.text(0.2, 0.6, 'Region I: Imaging eV to \nseveral hundred keV e-', fontsize=12, ha='left', va='center',color='#56A9F7')
# ax1.text(1.1, 9.35, 'Region II: Imaging to \n     few MeV e-', fontsize=12, ha='left', va='center',color='#56A9F7')

cbar.set_ticklabels([11, 37, 67, 101])  # Set the labels for the ticks
cbar.ax.tick_params(labelsize=10)
cbar_ax.yaxis.labelpad = 0.2

cbar_ax.yaxis.labelpad = 0.2
ax1.yaxis.labelpad = 0.2
ax1.xaxis.labelpad = 0.2

plt.savefig(
    "../simulation-results/final-images/3_fov_res.png",
    dpi=500,
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.02,
)

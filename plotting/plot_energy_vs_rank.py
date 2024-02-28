import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# script to plot electron energy vs mask thickness for 1/e stopping power of electrons
# data from ESTAR

# density
W_rho = 19.3  # g/cm^3

# find data file
fname = "stoppingpower_W.txt"
file1 = open(fname, "r")
Lines = file1.readlines()

count = 0
energies = []
ranges = []

# extract data from file
for line in Lines:
    count += 1
    # ignore header
    if count > 8:
        line_split = line.split(" ")
        # save energy and range
        energies.append(float(line_split[0]))
        # convert range from g/cm^2 to cm by diving by rho
        ranges.append(float(line_split[1]) / W_rho)

# convert range and energy
ranges = np.array(ranges) * 10  # convert to mm
energies = np.array(energies) * 1000  # convert to keV

from scipy.interpolate import interp1d

linear_interp = interp1d(energies, ranges, kind="linear", fill_value="extrapolate")

energies_plot = np.logspace(1, 4, 40)  # keV

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

color = "#39329E"
fig, ax1 = plt.subplots(figsize=(5.7, 2))

# for each energy, plot the highest rank we could use assuming max size

max_ranks = []
for ei, energy in enumerate(energies_plot):
    check = 0
    for ri, rank in enumerate(ranks):
        element_size_mm = detector_size_mm / rank
        # mask size
        mask_size = element_size_mm * (rank * 2 - 1)

        # get range
        thickness = linear_interp(energy)

        if element_size_mm < thickness:
            max_ranks.append(rank)
            check = 1
            break
        else:
            pass

        if ri == len(ranks) - 1 and check == 0:
            max_ranks.append(rank)
ax1.set_axisbelow(True)
ax1.grid(True, linestyle="--", color="lightgrey", linewidth=0.5)
ax1.plot(energies_plot, max_ranks, color=color, linewidth=2)
ax1.set_xscale("log")
ax1.set_ylabel(r"Rank", fontsize=8)
ax1.set_xlabel("Electron Energy [keV]", fontsize=8, color="black")
ax1.tick_params(axis="both", labelsize=8)
ax1.xaxis.labelpad = 0.2
ax1.yaxis.labelpad = 0.2
plt.savefig(
    "../simulation-results/final-images/3p1_energy_rank.png",
    dpi=500,
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.02,
)

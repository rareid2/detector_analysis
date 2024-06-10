import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm

nt = 59**2
rho = 0.5
psi = 2 / nt

import cmocean

cmap = cmocean.cm.thermal

results_dir = "../simulation-results/brehm/"

# get arrays
thicknesses = np.linspace(100, 4000, 25)
energies = np.linspace(10**2.4, 10**4.05, 25)  # keV

# Create a grid from x and y values
X, Y = np.meshgrid(energies, thicknesses)


# Create a function that determines the color based on x and y values
def color_function(energy, thickness):
    results_txt = results_dir + f"{int(energy)}_sec_gamma.txt"
    brehm = np.loadtxt(results_txt)
    results_txt = results_dir + f"{int(energy)}_sec_e.txt"
    sec_e = np.loadtxt(results_txt)

    results_txt = results_dir + f"{int(energy)}_primaries.txt"
    primary = np.loadtxt(results_txt)

    thick_index = np.where(thicknesses == thickness)

    total_secondary_prod = brehm[thick_index] + sec_e[thick_index]
    primary_only = primary[thick_index] - total_secondary_prod

    I = primary_only
    ksi = (total_secondary_prod / I) / nt

    snr = (
        np.sqrt(nt * I)
        * np.sqrt(rho * (1 - rho))
        * psi
        / ((rho + (1 - 2 * rho) * psi + ksi))
    )
    snr_no = (
        np.sqrt(nt * I)
        * np.sqrt(rho * (1 - rho))
        * psi
        / ((rho + (1 - 2 * rho) * psi + 0))
    )

    # print(snr / snr_no)
    snr_reduction = 1 - (snr / snr_no)

    return brehm[thick_index], sec_e[thick_index], primary_only, snr_reduction


es_array = []
snr_array = []
for energy in energies:
    snrs = []
    es = []
    for thickness in thicknesses:
        brehm, sec_e, primary_only, snr_reduction = color_function(energy, thickness)
        snrs.append(1000 * brehm / primary_only)
        es.append(1000 * sec_e / primary_only)

        # snrs.append(snr_reduction)
    snr_array.append(np.array(snrs))
    es_array.append(np.array(es))

fig = plt.figure(figsize=(5.7, 2.5))

sec_brehm = []
sec_e = []

# Large subplot spanning 2 rows and 2 columns
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

Z = np.hstack(snr_array)
Ze = np.hstack(es_array)
# Create a contour plot
contour = ax1.contourf(X / 1000, Y / 1000, Z, cmap=cmap)
cbar = plt.colorbar(contour, pad=0.02)
cbar.set_label("Brem. per 1000 e-", fontsize=8)

cbar.ax.yaxis.labelpad = 1.2
cbar.ax.tick_params(axis="y", labelsize=8)

ax1.set_xlabel("Electron Energy [MeV]", fontsize=8)
ax1.set_ylabel(r"Tungsten Thickness [mm]", fontsize=8)
rect = plt.Rectangle(
    (energies[18] / 1000, 0.12),
    0.2,
    3.85,
    fill=False,
    edgecolor="lightgrey",
    linestyle="--",
    linewidth=0.5,
)
ax1.axvline(
    energies[18] / 1000, 0, 3.87, color="lightgrey", linestyle="--", linewidth=0.5
)
ax1.tick_params(axis="y", which="both", labelsize=8)
ax1.tick_params(axis="x", which="both", labelsize=8)
ax1.yaxis.labelpad = 0.2
ax1.xaxis.labelpad = 0.2
# plt.xscale('log')

energy = energies[18]
print(energy)
snrs = []
electrons = []
for thickness in thicknesses:
    brehm, sec_e, primary_only, _ = color_function(energy, thickness)
    snrs.append(1000 * brehm / primary_only)
    electrons.append(1000 * sec_e / primary_only)

line_color = "#39329E"

# Bottom-left subplot
"""
ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
ax3.plot(thicknesses / 1000, electrons, color=line_color, linewidth=2)
ax3.yaxis.tick_right()

ax3.set_ylabel("Seconday e- \n per 1000 e-", fontsize=8)
ax3.yaxis.set_label_position("right")
ax3.tick_params(axis="x", which="both", labelsize=8)
ax3.tick_params(axis="y", which="both", labelsize=8)
ax3.yaxis.labelpad = 11
ax3.xaxis.labelpad = 0.2
"""
# Top-right subplot
ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
ax2.plot(thicknesses / 1000, snrs, color=line_color, linewidth=2)
ax2.yaxis.tick_right()

ax2.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)

ax2.set_ylabel("Brem. \n per 1000 e-", fontsize=8)
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis="x", which="both", labelsize=8)
ax2.tick_params(axis="y", which="both", labelsize=8)
ax2.yaxis.labelpad = 0.2
ax2.xaxis.labelpad = 0.2
# ax3.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)

ax1.text(0.5, 3.9, "a)", fontsize=8, ha="left", va="top", color="white")
ax2.text(0.03, 425, "b)", fontsize=8, ha="left", va="top", color="black")
# ax3.text(0.03, 6.7, "c)", fontsize=8, ha="left", va="top", color="black")
ax2.set_xlabel("Tungsten Thickness [mm]", fontsize=8)
# plt.setp(ax2.get_xticklabels(), visible=False)
plt.tight_layout()

plt.savefig(
    "../simulation-results/final-images/11_brem.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.clf()


# ------------------------------ AGAIN-----------------------------

# Large subplot spanning 2 rows and 2 columns
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan=2, colspan=2)

# Create a contour plot
contour = ax1.contourf(X / 1000, Y / 1000, Ze, cmap=cmap)
cbar = plt.colorbar(contour, pad=0.02)
cbar.set_label("Secondary e- per 1000 e-", fontsize=8)

cbar.ax.yaxis.labelpad = 1.2
cbar.ax.tick_params(axis="y", labelsize=8)

ax1.set_xlabel("Electron Energy [MeV]", fontsize=8)
ax1.set_ylabel(r"Tungsten Thickness [mm]", fontsize=8)
rect = plt.Rectangle(
    (energies[18] / 1000, 0.12),
    0.2,
    3.85,
    fill=False,
    edgecolor="lightgrey",
    linestyle="--",
    linewidth=0.5,
)
ax1.axvline(
    energies[18] / 1000, 0, 3.87, color="lightgrey", linestyle="--", linewidth=0.5
)
ax1.tick_params(axis="y", which="both", labelsize=8)
ax1.tick_params(axis="x", which="both", labelsize=8)
ax1.yaxis.labelpad = 0.2
ax1.xaxis.labelpad = 0.2
# plt.xscale('log')

energy = energies[18]
print(energy)
snrs = []
electrons = []
for thickness in thicknesses:
    brehm, sec_e, primary_only, _ = color_function(energy, thickness)
    snrs.append(1000 * brehm / primary_only)
    electrons.append(1000 * sec_e / primary_only)

line_color = "#39329E"

# Top-right subplot
ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
ax2.plot(thicknesses / 1000, electrons, color=line_color, linewidth=2)
ax2.yaxis.tick_right()

ax2.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)

ax2.set_ylabel("Secondary e- \n per 1000 e-", fontsize=8)
ax2.yaxis.set_label_position("right")
ax2.tick_params(axis="x", which="both", labelsize=8)
ax2.tick_params(axis="y", which="both", labelsize=8)
ax2.yaxis.labelpad = 0.2
ax2.xaxis.labelpad = 0.2
# ax3.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)

ax1.text(0.5, 3.9, "a)", fontsize=8, ha="left", va="top", color="white")
ax2.text(0.03, 6.8, "b)", fontsize=8, ha="left", va="top", color="black")
# ax3.text(0.03, 6.7, "c)", fontsize=8, ha="left", va="top", color="black")

# plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_xlabel("Tungsten Thickness [mm]", fontsize=8)
plt.tight_layout()

plt.savefig(
    "../simulation-results/final-images/11_sec_e.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.clf()

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.ticker
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


#     "#DA6821",
hex_list = ["#3C2541", "#603B68", "#83528E", "#A271AD", "#BB97C3", "#D5BEDA"]
cmap = get_continuous_cmap(hex_list)

# snr as a funciton of transparency (not background) --- time exposure
nt = 47**2  # can vary
I = 1e7  # can vary
rho = 0.5
psi = 1 / nt
ksis = [
    0.000458,
    0.0006820000000004045,
    0.000448000000000226,
    0.0005339999999999234,
    0.00031000000000003247,
    0.0018679999999999808,
    0.0004579999999998474,
    0.00323799999999963,
    0.0028760000000003227,
    0.0005980000000000985,
    0.0035479999999998846,
    0.0041439999999999255,
    0.007066000000000239,
    0.011113999999999846,
    0.024151999999999507,
    0.08024600000000048,
    0.176396,
    0.27405800000000013,
    0.38719999999999954,
    0.52553,
]
colors = [
    cmap(i) for i in np.linspace(0, 1, len(ksis))
]  # Adjust the number of lines accordingly


snr0 = (
    (1 - 0)
    * np.sqrt(nt * I)
    * np.sqrt(rho * (1 - rho))
    * psi
    / (np.sqrt((1 - 0) * (rho + (1 - 2 * rho) * psi) + 0))
)
signal = []
ntht0 = (1) / np.sqrt((1) * rho + 1)

for i, ksi in enumerate(ksis):
    snr = (
        (1 - ksi)
        * np.sqrt(nt * I)
        * np.sqrt(rho * (1 - rho))
        * psi
        / (np.sqrt((1 - ksi) * (rho + (1 - 2 * rho) * psi + ksi)))
    )
    ntht = (1 - ksi) / np.sqrt((1 - ksi) * rho + ksi)
    signal.append(100 * (snr0 / snr) ** 2 - 100)

from scipy.interpolate import interp1d

ksise = np.logspace(1, 4, 20)
# Create a linear interpolation function
linear_interp = interp1d(ksise, signal, kind="linear", fill_value="extrapolate")
linear_interp2 = interp1d(ksise, ksis, kind="linear", fill_value="extrapolate")
ksise = np.logspace(1, 4, 500)
signal = linear_interp(ksise)
signal2 = linear_interp2(ksise)

fig, ax1 = plt.subplots(figsize=(5.7, 2))
color = "#39329E"
ax1.semilogx(ksise, signal, color=color, linewidth=2)
ax2 = ax1.twinx()
ax2.semilogx(ksise, 100 * np.abs(signal2), color="#D57965", linestyle="--", linewidth=2)
ax2.set_ylabel(r"% Transparency (300$\mu$m W)", fontsize=8, color="#D57965")
ax2.tick_params(axis="y", labelcolor="#D57965", labelsize=8)
# ax2.grid(True,color="#DC6922",linestyle='--', linewidth=0.5)

ax1.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
ax2.set_ylim([-5 / 7, 55])
ax1.set_ylim([-5, 345])
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.set_yticks([round(i) for i in np.linspace(0, 51, 5)])
# ax2.yaxis.set_major_locator(plt.MaxNLocator(5))


# ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
ax1.tick_params(axis="y", labelcolor=color, labelsize=8)
ax1.tick_params(axis="x", labelsize=8)

ax1.set_xlabel("Electron Energy [keV]", fontsize=8)
ax1.set_ylabel("% Increase in Exposure", fontsize=8, color=color)
ax1.xaxis.labelpad = 0.2
ax1.yaxis.labelpad = 0.2
ax2.yaxis.labelpad = 0.2
plt.savefig(
    "10_snr_energy_transparency.png", dpi=500, bbox_inches="tight", pad_inches=0.02
)

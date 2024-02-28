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
hex_list = ["#3C214C", "#B26EDF", "#56A9F7", "#D66853", "#D7C9AA"]
hex_list = [
    "#03071e",
    "#370617",
    "#6a040f",
    "#9d0208",
    "#d00000",
    "#dc2f02",
    "#e85d04",
    "#f48c06",
    "#faa307",
    "#ffba08",
]
hex_list.reverse()
hex_list = ["#3C2541", "#603B68", "#83528E", "#A271AD", "#BB97C3", "#D5BEDA"]
hex_list = ["#E87659", "#F89041", "#FCAF3C", "#F9C641", "#F7D245", "#EBEF55"]

cmap = get_continuous_cmap(hex_list)
import cmocean

cmap = cmocean.cm.thermal
nt = 59**2  # can vary
I = 16873112.601  # can vary

# x, y = np.meshgrid(psi,ksi)
rho = 0.5
psi = np.linspace(1 / nt, 1, 1000)
ksis = [(((0.007 * (1.007 * I)) + 100) / I) / nt]
ksis = [0]
# colors = [cmap(i) for i in np.linspace(0, 1, len(ksis))]  # Adjust the number of lines accordingly
colors = ["#39329E"]

fig, ax = plt.subplots(figsize=(5.7, 2))
snr_max = np.sqrt(nt * I) * np.sqrt(rho * (1 - rho)) / (np.sqrt(rho + (1 - 2 * rho)))
for i, ksi in enumerate(ksis):
    # print(ksi)
    snr = (
        np.sqrt(nt * I)
        * np.sqrt(rho * (1 - rho))
        * psi
        / (np.sqrt(rho + (1 - 2 * rho) * psi + ksi))
    )
    # pinhole = np.sqrt(I) * psi / np.sqrt(psi + ksi)
    # ntht = np.sqrt(4*nt*I) * np.sqrt(0.125 / 2) * psi / np.sqrt(0.125 + ksi)
    advatnage = snr
    plt.plot(psi, advatnage, color=colors[i])


def custom_formatter(value, tick_number):
    return rf"{round(value,2)}"


def custom_formatter_val(value, tick_number):
    return rf"{round(value*nt)}/$N_T$"


from matplotlib.colorbar import ColorbarBase

# cbar = ColorbarBase(ax.inset_axes([1.01, 0, 0.03, 1]), cmap=cmap)
# cbar.set_label(r'Background Level (per pixel) $\xi$',fontsize=10)
# cbar.ax.yaxis.labelpad = 0.4
ax.yaxis.labelpad = 0.2
ax.xaxis.labelpad = 0.2

# cbar_ticks = [0, 0.25, 0.5, 0.75, 1, 1.25]
cbar_tick_labels = []

for i in range(4):
    large_number = (i) * 50000
    if i == 5:
        scientific_notation = format(large_number, ".2e")
    else:
        scientific_notation = format(large_number, ".1e")
    # Extract the components of scientific notation
    coefficient, exponent = scientific_notation.split("e")
    # Format the result with the desired format
    if int(exponent) < 0:
        formatted_result = rf"{coefficient}x$10^{{{(int(exponent))}}}$"
    else:
        formatted_result = rf"{coefficient}x$10^{int(exponent)}$"
    cbar_tick_labels.append(formatted_result)
ksls = []
ksis_labels = np.linspace(0, max(ksis), 6)
for ksl in ksis_labels:
    formatted_result = round(ksl, 2)
    ksls.append(formatted_result)
ax.set_yticklabels(cbar_tick_labels)
# cbar.ax.tick_params(labelsize=10)

plt.grid()
plt.xlabel(r"Source Exent $\psi_{i,j}$", fontsize=8)
plt.ylabel("SNR", fontsize=8)
plt.xlim([0.001, 1])
plt.ylim([1, snr_max + 10])
ax.tick_params(labelsize=8)
# plt.xlim([0,5/nt])

# Set the zoom-in region
x_start, x_end = 1 / nt, 5 / nt
y_start, y_end = 0, 250
bbox_to_anchor = (0.083, -0.245, 1, 1.25)

axins = inset_axes(
    ax,
    width="30%",
    height="30%",
    loc="upper left",
    bbox_to_anchor=bbox_to_anchor,
    bbox_transform=ax.transAxes,
)

# Create the inset plot
for i, ksi in enumerate(ksis):
    # print(ksi)
    snr = (
        np.sqrt(nt * I)
        * np.sqrt(rho * (1 - rho))
        * psi
        / (np.sqrt(rho + (1 - 2 * rho) * psi + ksi))
    )
    pinhole = np.sqrt(I) * psi / np.sqrt(psi + ksi)
    advatnage = snr

    axins.plot(psi, advatnage, color=colors[i])

axins.set_xlim(x_start, x_end)
axins.set_ylim(y_start, y_end)
axins.tick_params(labelsize=10)
ticks = np.arange(x_start, x_end + 1 / nt, 1 / nt)
axins.set_xticks(ticks)


def custom_format_func(value, pos):
    if value == 0:
        return "0"
    else:
        scientific_notation = format(value, ".1e")
        # Extract the components of scientific notation
        coefficient, exponent = scientific_notation.split("e")
        formatted_result = rf"{coefficient}x$10^{{{int(exponent)}}}$"
        return formatted_result


# Apply the custom formatter to the x-axis
# axins.yaxis.set_major_formatter(FuncFormatter(custom_format_func))
axins.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axins.xaxis.set_major_formatter(FuncFormatter(custom_formatter_val))
plt.subplots_adjust(right=0.9)
ax.text(0.95, -0.15 * max(snr), "Point \n source", fontsize=6, ha="center", va="center")
ax.text(
    0,
    -0.15 * max(snr),
    "Uniformly bright \n source",
    fontsize=6,
    ha="center",
    va="center",
)
ax.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
axins.tick_params(labelsize=8)
# plt.text(1,-0.1,"Point source", fontsize=10)

from matplotlib.patches import Rectangle

rectangle = Rectangle(
    (0, 0), 50 * 5 / nt, 50 * 250, facecolor="white", edgecolor="black", linewidth=1
)
ax.add_patch(rectangle)

plt.savefig("7_snr.png", bbox_inches="tight", pad_inches=0.02, dpi=500)

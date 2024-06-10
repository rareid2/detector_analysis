import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plot_settings import *
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import FWHM
params = [4.04753095e-04, 2.89058707e00]

data = np.loadtxt(
    "../simulation-results/rings/89-4.49-ring-pt_1.00E+08_Mono_100_dc.txt"
)
det_size_cm = 4.984  # cm
distance = 4.49  # cm
pixel_count = int(89 * 3)
center_pixel = int(89 * 3 / 2)
geometric_factor = 15
pixel_size = 0.18666667 * 0.1  # cm

# need to create bins based on FWHM
# start with first radial distance (0)
max_rad_dist = np.sqrt(2) * det_size_cm / 2
fwhm_step = 0
bins = []
radial_distances = []
bin_sizes = []
while pixel_size * fwhm_step < max_rad_dist:
    fwhm_z = polynomial_function(fwhm_step, *params)
    radial_distance_1 = fwhm_step * pixel_size
    angle1 = np.rad2deg(np.arctan(radial_distance_1 / distance))
    fwhm_step += fwhm_z

    radial_distance_2 = fwhm_step * pixel_size
    angle2 = np.rad2deg(np.arctan(radial_distance_2 / distance))
    # define bin edges using the step
    bin_edges = (angle1, angle2)
    bin_size = angle2 - angle1
    bins.append(angle2)
    bin_sizes.append(bin_size)
    radial_distances.append(
        radial_distance_1 + (radial_distance_2 - radial_distance_1) / 2
    )
bins.insert(0, 0)
# now we have bins, loop through each pixel and save pixel ID into bins
angles = np.zeros((89 * 3, 89 * 3))
bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}

for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        radial_distance = np.sqrt(relative_x**2 + relative_y**2)  # in cm

        # find the geometrical theta angle of the pixel
        angle = np.rad2deg(np.arctan(radial_distance / distance))
        angles[y, x] = angle

        # find which bin its in
        for ii, bn in enumerate(bins[:-1]):
            if angle >= bn and angle < bins[ii + 1]:
                bins_ids[f"{ii}"].append((y, x))
            if ii == len(bins[:-1]):
                if angle >= bn and angle <= bins[ii + 1]:
                    bins_ids[f"{ii}"].append((y, x))

# for each radial slice, we need to calaculate the azimuth angle
gryophase_bins = {}
factors = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    15,
    18,
    20,
    24,
    30,
    36,
    40,
    45,
    72,
    90,
    120,
    180,
    360,
]
for bi, radial_distance in enumerate(radial_distances):
    gryophase_bins[f"{bi}"] = []

    fwhm_az = polynomial_function(radial_distance, *params)
    # get curcumference
    n_pix = len(bins_ids[f"{bi}"])
    if n_pix > 0:
        circum_cm = 2 * np.pi * radial_distance
        gryo_bin_width = 360 / (circum_cm / (fwhm_az * pixel_size))
        # now we have the bin width, use it to define the bins i think -btu

        larger_elements = [element for element in factors if element > gryo_bin_width]
        closest_factor = min(larger_elements, key=lambda x: x - gryo_bin_width)

        gyro_bins = np.arange(0, 360 + closest_factor, closest_factor)

        gryophase_bins[f"{bi}-bins"] = gyro_bins
        gryophase_bins[f"{bi}-indices"] = {
            f"{key}": [] for key in range(len(gyro_bins) - 1)
        }

        # now we need to find which bins each radial slice falls into
        for y, x in bins_ids[f"{bi}"]:
            # calculate azimuth angle
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            gryo_angle = np.rad2deg(np.arctan2(relative_y, relative_x))
            gryo_angle = (gryo_angle + 360) % 360
            for gi, gb in enumerate(gyro_bins[:-1]):
                if gryo_angle >= gb and gryo_angle < gyro_bins[gi + 1]:
                    # inside the bin, save index
                    gryophase_bins[f"{bi}-indices"][f"{gi}"].append((y, x))


# okay I now have radial bins, the indices that belong to each,
# and for each radial bin i have gryo bins, and the indices that belong to each


fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.set_aspect("equal", adjustable="box")


def custom_formatter(value, _):
    # normalize
    rr = value * np.sqrt(2) * 4.984 / 2
    # Use your formula to convert the tick value to the desired label
    new_label = round(
        np.rad2deg(np.arctan(rr / distance)), 1
    )  # Replace this with your formula
    return rf"{new_label}$^\circ$"  # Format the label as needed


# Set the custom formatter for the radial ticks
ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax.tick_params(axis="y", colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.xaxis.grid(color="white")
ax.yaxis.grid(color="white")

# Add labels for the radius and y-axis
# ax.set_ylabel(r'$\theta$', color='white')  # Label for the radius axis
# ax.set_xlabel(r'$\phi$', color='white')    # Label for the y-axis

plt.savefig("polar_grid.png", bbox_inches="tight", pad_inches=0, transparent=True)
plt.close()
plt.clf()
# so i need to try and plot one of them
pitch_angle_bins = bins

fig, ax = plt.subplots(
    1,
    1,
    figsize=(3, 3),
)
polar_grid_image = plt.imread("polar_grid.png")

import cmocean

cmap = cmocean.cm.thermal

im = ax.imshow(data, cmap, origin="lower")

cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.01, shrink=0.8)
ax.axis("off")


cbar.set_label(r"$\times 10^3 cm^{-2}sr^{-1}$", fontsize=8)
cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
cbar.ax.tick_params(axis="x", labelsize=8)
# ax[0].text(1.15,-0.08,rf'$\times 10^{int(3)}$', ha='right', va='top', transform=ax[0].transAxes, fontsize=8)
cbar.ax.xaxis.labelpad = 0.2

new_position = [-0.09, 0.11, 0.77, 0.77]  # [left, bottom, width, height]
ax.set_position(new_position)

new_position = [-0.09, 0.053, 0.77, 0.04]  # [left, bottom, width, height]
cbar.ax.set_position(new_position)

overlay_position = (0.5, 0.5)  # Example: (0.5, 0.5) is the center

circle = plt.Circle((133, 133), 6, fill=None, edgecolor="white", linewidth=0.3)
ax.add_patch(circle)
circle = plt.Circle(
    (133, 133), 1, fill=True, edgecolor="white", linewidth=0.3, facecolor="white"
)
ax.add_patch(circle)

ax.text(117, 145, r"$\vec{B}$", fontsize=6, ha="left", va="top", color="white")

plt.savefig(
    "../simulation-results/final-images/rings-sim.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig, ax = plt.subplots(
    1,
    1,
    figsize=(7, 2),
)
# ax[0].imshow(polar_grid_image, extent=(overlay_position[0], overlay_position[0]+0.2, overlay_position[1], overlay_position[1]+0.2),alpha=0.2)

norm = plt.Normalize(0, 41399)
total_signal = 0
all_counts = 0
values = []

for pn in range(len(pitch_angle_bins) - 1):
    for gi, gb_bin in enumerate(gryophase_bins[f"{pn}-bins"][:-1]):
        x_loc = gb_bin

        # get value
        integrated_val = 0
        for index in gryophase_bins[f"{pn}-indices"][f"{gi}"]:
            yin, xin = index
            integrated_val += data[yin, xin]
            values.append(integrated_val)

        if pn == 3:
            total_signal += integrated_val

        all_counts += integrated_val
        color = cmap(norm(integrated_val))
        x_width = gryophase_bins[f"{pn}-bins"][gi + 1] - gb_bin
        y_loc = pitch_angle_bins[pn]
        y_height = pitch_angle_bins[pn + 1] - y_loc

        if integrated_val == 0:
            rectangle = plt.Rectangle(
                (x_loc, y_loc), x_width, y_height, facecolor="white", edgecolor="white"
            )
        else:
            rectangle = plt.Rectangle(
                (x_loc, y_loc),
                x_width,
                y_height,
                facecolor=color,
                edgecolor="black",
                lw=0.2,
            )
        ax.add_patch(rectangle)


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


print(max(values))
print(total_signal)
print(all_counts)
plt.xlim([0, 360])
plt.ylim([0, 38.36])
plt.xlabel(r"Gyrophase $\phi^\circ$", fontsize=8)
plt.ylabel(r"Pitch Angle $\theta^\circ$", fontsize=8)

ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)


cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
cbar.ax.tick_params(axis="y", labelsize=8)
# ax[1].text(1.2, -0.05, rf'$\times 10^{int(4)}$', ha='right', va='top', transform=ax[1].transAxes, fontsize=8)
cbar.set_label(r"$\times 10^4cm^{-2}sr^{-1}$", fontsize=8)


cbar.ax.yaxis.labelpad = 1.2
cbar.ax.tick_params(axis="y", labelsize=8)
ax.xaxis.labelpad = 0.2
ax.yaxis.labelpad = 0.2


new_position = [0.49, 0.065, 0.37, 0.81]  # [left, bottom, width, height]
ax.set_position(new_position)

original_position = cbar.ax.get_position()
print(original_position)

new_position = [0.865, 0.065, 0.92, 0.81]  # [left, bottom, width, height]
cbar.ax.set_position(new_position)


plt.savefig(
    "../simulation-results/final-images/rings-projected.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plot_settings import *
from matplotlib import cm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter


def find_edge_pixels(group_pixels, pixel_size, grid_size):
    edge_pixels = []
    for y, x in group_pixels:
        pixel_edges = []  # Store the edges for the current pixel
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue  # Skip the current pixel
                neighbor_y = y + dy * pixel_size
                neighbor_x = x + dx * pixel_size
                if (
                    neighbor_y < 0
                    or neighbor_x < 0
                    or neighbor_y >= grid_size[0]
                    or neighbor_x >= grid_size[1]
                    or (neighbor_y, neighbor_x) not in group_pixels
                ):
                    # Determine the side of the edge
                    edge_side = ""
                    if dy < 0 and dx == 0:
                        edge_side = "left"
                    elif dy > 0 and dx == 0:
                        edge_side = "right"
                    elif dx < 0 and dy == 0:
                        edge_side = "bottom"
                    elif dx > 0 and dy == 0:
                        edge_side = "top"
                    pixel_edges.append(edge_side)
        # Append the pixel coordinates and its edge sides to edge_pixels
        if pixel_edges:
            edge_pixels.append(((y, x), pixel_edges))
    return edge_pixels


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
print(bins[:10])
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
def custom_formatter(value, _):
    # normalize
    rr = value * np.sqrt(2) * 4.984 / 2
    # Use your formula to convert the tick value to the desired label
    new_label = round(
        np.rad2deg(np.arctan(rr / distance)), 1
    )  # Replace this with your formula
    return rf"{new_label}$^\circ$"  # Format the label as needed


pitch_angle_bins = bins
fig, ax = plt.subplots(
    1,
    2,
    figsize=(5.7, 3),
)
import cmocean

cmap = cmocean.cm.thermal

im = ax[0].imshow(np.zeros_like(data), cmap="Greys", origin="lower", vmin=0, vmax=1)
im = ax[1].imshow(np.zeros_like(data), cmap="Greys", origin="lower", vmin=0, vmax=1)

nbins = 8
norm = plt.Normalize(0, nbins)
total_signal = 0
all_counts = 0
values = []

for pn in range(nbins):  # first 10 bins only
    for gi, gb_bin in enumerate(gryophase_bins[f"{pn}-bins"][:-1]):
        for index in gryophase_bins[f"{pn}-indices"][f"{gi}"]:
            yin, xin = index
            color = cmap(norm(pn))
            rectangle = plt.Rectangle(
                (yin, xin),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                lw=0.2,
            )
            ax[0].add_patch(rectangle)

kwargs = {}
center_pixel = 267 // 2
all_gyro_bins = []
for pn in range(nbins):  # first 10 bins only
    norm = plt.Normalize(0, len(gryophase_bins[f"{pn}-bins"][:-1]) - 1)
    flat_bins = []
    for gi, gb_bin in enumerate(gryophase_bins[f"{pn}-bins"][:-1]):
        for ei, index in enumerate(gryophase_bins[f"{pn}-indices"][f"{gi}"]):
            yin, xin = index
            flat_bins.append((yin, xin))
            color = cmap(norm(gi))
            rectangle = plt.Rectangle(
                (yin, xin),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                lw=0.2,
            )
            ax[1].add_patch(rectangle)
            """
            if pn < 1:
                bin_limit = pn * 3 + 1
                if (
                    np.abs(center_pixel - xin) > bin_limit
                    or np.abs(center_pixel - yin) > bin_limit
                ):
                    kwargs["linewidth"] = 1
                    kwargs["color"] = "black"
                    width = 1
                    height = 1
                    if center_pixel - yin < -1 * bin_limit:
                        ax[1].plot(
                            [xin, xin + width], [yin + height, yin + height], **kwargs
                        )  # Top border
                    if center_pixel - yin > bin_limit:
                        ax[1].plot(
                            [xin, xin + width], [yin, yin], **kwargs
                        )  # Bottom border
                    if center_pixel - xin > bin_limit:
                        ax[1].plot(
                            [xin, xin], [yin, yin + height], **kwargs
                        )  # Left border
                    if center_pixel - xin < -1 * bin_limit:
                        ax[1].plot(
                            [xin + width, xin + width], [yin, yin + height], **kwargs
                        )  # Right border
                    """
    all_gyro_bins.append(flat_bins)

pixel_size = 1
for fb in all_gyro_bins[:nbins]:
    edge_px = find_edge_pixels(fb, 1, (267, 267))
    for (y, x), sides in edge_px:
        for side in sides:
            if side == "top":
                ax[1].plot([y, y + 1], [x + 1, x + 1], "black")
                ax[0].plot([y, y + 1], [x + 1, x + 1], "black")
            if side == "bottom":
                ax[1].plot([y, y + 1], [x, x], "black")
                ax[0].plot([y, y + 1], [x, x], "black")
            if side == "left":
                ax[1].plot([y, y], [x, x + 1], "black")
                ax[0].plot([y, y], [x, x + 1], "black")
            if side == "right":
                ax[1].plot([y + 1, y + 1], [x, x + 1], "black")
                ax[0].plot([y + 1, y + 1], [x, x + 1], "black")

limits = 16
for axes in ax:
    axes.axis("off")
    axes.set_xlim([133 - limits, 133 + limits + 1])
    axes.set_ylim([133 - limits, 133 + limits + 1])

# ax[0].text(133 - limits + 2, 133 + limits - 2, r"$\theta$", fontsize=12)
# t = ax[1].text(
#    133 - limits + 2, 133 + limits - 2, r"$\phi$", fontsize=12, color="black"
# )
# t.set_bbox(dict(facecolor="white", edgecolor="white"))

plt.savefig(
    "../simulation-results/final-images/bins.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)

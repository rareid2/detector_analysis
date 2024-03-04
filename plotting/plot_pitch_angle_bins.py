import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import numpy as np


# first define function that describes the map
def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# geom factor
geom_factor = 21.4
distance = 1.9764  # cm
# import the grid
params = [1.39403798e-04, 1.35313095e-03, 9.05102084e-01]
grid_size = 61
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)
fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
gf_grid = geom_factor * fwhm_grid / np.sum(fwhm_grid)
# gf_grid = np.ones((61, 61)) * (geom_factor / 61**2)
print(np.sum(gf_grid))
output_name = "../simulation-results/strahl/variation_in_gf"
plt.imshow(gf_grid)
plt.colorbar(label="gf at each pixel [cm^2 sr]")
plt.savefig(f"{output_name}_2D.png", dpi=300)

fwhm_step = 0
det_size_cm = 4.941  # cm
pixel = 0.81  # mm
pixel_size = pixel * 0.1
max_rad_dist = np.sqrt(2) * det_size_cm / 2
bin_sizes = []
bins = []
while pixel_size * fwhm_step < max_rad_dist:
    # fwhm_z = polynomial_function(fwhm_step, *params)
    fwhm_z = 1
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
bins.insert(0, 0)
bins = bins[:-1]  # remove the last bin
pixel_count = 61
center_pixel = int(pixel_count // 2)
bins_ids = {f"{key}": [] for key in range(len(bins) - 1)}
gf_ids = {f"{key}": [] for key in range(len(bins) - 1)}

signal = np.random.rand(61, 61)
signal = np.ones((61, 61))

plt.clf()
bn = 7
bn_count = 0
for x in range(pixel_count):
    for y in range(pixel_count):
        relative_x = (x - center_pixel) * pixel_size
        relative_y = (y - center_pixel) * pixel_size

        aa = np.sqrt(relative_x**2 + relative_y**2)

        # find the geometrical theta angle of the pixel
        angle = np.rad2deg(np.arctan(aa / distance))

        # find the correct bin
        for ii, bn in enumerate(bins[:-1]):
            if angle >= bn and angle < bins[ii + 1]:
                bins_ids[f"{ii}"].append(signal[y, x])
                gf_ids[f"{ii}"].append(gf_grid[y, x])
                """
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)
                bn_count += 1
                """

# now sum it
print("total image signal", np.sum(signal))
gf_sum = []
total_sum = 0
px_sum = 0
gf_sum_2 = 0
for ii, bn in enumerate(bins[:-1]):
    px_sum += np.sum(np.array(bins_ids[f"{ii}"]))
    gf_sum_2 += np.sum(np.array(gf_ids[f"{ii}"]))

    bin_val = np.sum(np.array(bins_ids[f"{ii}"])) / np.sum(np.array(gf_ids[f"{ii}"]))
    gf_sum.append(bin_val / 43)
    total_sum += bin_val / 43

print(px_sum, gf_sum_2, total_sum, total_sum / (61**2 / 21.4))
plt.scatter(bins[:-1], np.array(gf_sum))
plt.xlabel("pitch angle")
plt.ylabel("flux [/cm^2 sr]")

output_name = f"../simulation-results/strahl/signal"
plt.savefig(f"{output_name}.png", dpi=300)

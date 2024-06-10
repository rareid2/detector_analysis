import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cmocean

cmap = cmocean.cm.thermal
colors = ["#39329E", "#88518D", "#D76C6B", "#FCA63B", "#E9F758"]
colors = ["#39329E", "#39329E", "#39329E", "#39329E", "#39329E"]

colors.reverse()


# Function to load 2D array from a txt file
def load_data(file_path):
    return np.loadtxt(file_path)


def fmt(x, pos):
    if x == 0:
        return "0"
    else:
        a, b = "{:.1e}".format(x).split("e")
        b = int(b)
        # print(b)
        a = float(a)
        if a % 1 > 0.1:
            pass
        else:
            a = int(a)
        return f"{a}"


# Create a 3x5 grid of subplots
fig, axs = plt.subplots(
    5,
    3,
    figsize=(5.7, 8),
    gridspec_kw={"hspace": 0, "wspace": 0},
    sharex=True,
    sharey=True,
)

thetas = [2, 13, 24, 35, 46]
n_p = [5e6, 5e7, 5e8]
fcfov = 44.79
distance = 3.47


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


# import FWHM
params = [2.52336124e-04, -2.83882554e-03, 8.86278977e-01]
grid_size = 59
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid

# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))

# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)

# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)

# need to make it % deviation
fwhm_grid_dev = (np.min(fwhm_grid) - fwhm_grid) / np.min(fwhm_grid)

# make it positive
fwhm_grid_positive_dev = 1 + fwhm_grid_dev

# make it sum to 18
gf_grid = 18 * fwhm_grid_positive_dev / np.sum(fwhm_grid_positive_dev)

fig, axs = plt.subplots(1, 3, figsize=(8, 3))

im1 = axs[0].imshow(fwhm_grid, cmap=cmap)
im2 = axs[1].imshow(fwhm_grid_dev, cmap=cmap)
im3 = axs[2].imshow(gf_grid, cmap=cmap)
cbar1 = fig.colorbar(im1, ax=axs[0], orientation="horizontal", pad=0.01, shrink=0.84)
cbar2 = fig.colorbar(im2, ax=axs[1], orientation="horizontal", pad=0.01, shrink=0.84)
cbar3 = fig.colorbar(im3, ax=axs[2], orientation="horizontal", pad=0.01, shrink=0.84)

cbar1.set_label("FWHM [pixels]", fontsize=10)
cbar2.set_label(r"% Deviation in FWHM", fontsize=10)
cbar3.set_label(r"Geometric Factor [cm$^2$ sr]", fontsize=10)

for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.savefig(
    "../simulation-results/final-images/gf-grid.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)

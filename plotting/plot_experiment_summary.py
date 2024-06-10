import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mtick
from numpy.typing import NDArray
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as opt

results_folder = "/home/rileyannereid/workspace/geant4/experiment_results/"
import cmocean

cmap = cmocean.cm.thermal


def plot_3D_signal(heatmap):
    x = np.arange(heatmap.shape[0])
    y = np.arange(heatmap.shape[1])
    X, Y = np.meshgrid(x, y)

    # Create a 3D figure
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create the surface plot!
    surface = ax.plot_surface(X, Y, heatmap, cmap=cmap)

    # Add a color bar for reference
    fig.colorbar(surface)

    # Show the plot
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.set_zlabel("signal")

    plt.savefig("3d.png", dpi=300)


def calculate_fwhm(data):
    def gaussian2d(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
        xo = float(xo)
        yo = float(yo)
        x, y = xy
        gg = offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x**2) + ((y - yo) ** 2) / (2 * sigma_y**2))
        )
        return gg.ravel()

    def getFWHM_GaussianFitScaledAmp(img):
        """Get FWHM(x,y) of a blob by 2D gaussian fitting
        Parameter:
            img - image as numpy array
        Returns:
            FWHMs in pixels, along x and y axes.
        """
        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        x, y = np.meshgrid(x, y)

        initial_guess = (img.shape[1] / 2, img.shape[0] / 2, 3, 3, 1, 0)
        params, _ = opt.curve_fit(
            gaussian2d, (x, y), img.ravel(), p0=initial_guess, maxfev=5000
        )
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = params

        FWHM_x = 2 * sigmaX * np.sqrt(2 * np.log(2))
        FWHM_y = 2 * sigmaY * np.sqrt(2 * np.log(2))
        fwhm = FWHM_x
        print(offset)

        return fwhm, params

    # get the shifted image and scale
    max_index_flat = np.argmax(data)
    max_index = np.unravel_index(max_index_flat, data.shape)
    img = data / np.amax(data)
    sect = 30

    # get the section where the signal is
    img_bound = img.shape[0] - 1
    if max_index[0] > img_bound - sect and max_index[1] < img_bound - sect:
        img_clipped = img[
            max_index[0] - sect :, max_index[1] - sect : max_index[1] + sect + 1
        ]

        missing_rows = img_bound - max_index[0]
        missing_signal = img[
            max_index[0] - sect : max_index[0] - missing_rows,
            max_index[1] - sect : max_index[1] + sect + 1,
        ]
        flipped_pre_signal = missing_signal[::-1]
        img_clipped = np.vstack((img_clipped, flipped_pre_signal))

    elif max_index[1] > img_bound - sect and max_index[0] < img_bound - sect:
        img_clipped = img[
            max_index[0] - sect : max_index[0] + sect + 1, max_index[1] - sect :
        ]
        missing_cols = img_bound - max_index[1]
        missing_signal = img[
            max_index[0] - sect : max_index[0] + sect + 1,
            max_index[1] - sect : max_index[1] - missing_cols,
        ]
        flipped_pre_signal = missing_signal[:, ::-1]
        img_clipped = np.hstack((img_clipped, flipped_pre_signal))

    elif max_index[0] > img_bound - sect and max_index[1] > img_bound - sect:
        img_clipped = img[max_index[0] - sect :, max_index[1] - sect :]

        missing_rows = img_bound - max_index[0]
        missing_signal = img[
            max_index[0] - sect : max_index[0] - missing_rows, max_index[1] - sect :
        ]
        flipped_pre_signal = missing_signal[::-1]
        img_clipped = np.vstack((img_clipped, flipped_pre_signal))

        missing_cols = img_bound - max_index[1]
        missing_signal = img_clipped[:, : (sect - missing_cols)]
        flipped_pre_signal = missing_signal[:, ::-1]
        img_clipped = np.hstack((img_clipped, flipped_pre_signal))

    else:
        img_clipped = img[
            max_index[0] - sect : max_index[0] + sect + 1,
            max_index[1] - sect : max_index[1] + sect + 1,
        ]

    FWHM, params = getFWHM_GaussianFitScaledAmp(img_clipped)
    return FWHM


# try it out
data = np.loadtxt(f"{results_folder}Cd109-0-tf_dc.txt")
plot_3D_signal(data)
fwhm = calculate_fwhm(data)
print(fwhm)

"""plot data collected in experiment with minipix edu
"""


inds = ["0", "20"]
col_names = ["raw", "cleaned", "dc"]

# Create subplots with shared x-axis and y-axis
fig, axs = plt.subplots(2, 3, figsize=(8, 4), sharex="col")

panels = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]

# Plot data and colorbars
for i in range(3):
    for j in range(2):
        fname = f"{results_folder}Cd109-{inds[j]}_{col_names[i]}.txt"
        if i == 2:
            data = np.loadtxt(fname) / 11**2
        else:
            data = np.loadtxt(fname)
        ax = axs[j, i]
        ax.set_box_aspect(1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if i == 0:
            vmax = 100
            label = "keV/pixel"
            extend = "max"
        elif i == 1:
            vmax = 3
            label = "Counts"
            extend = "max"
        else:
            vmax = None
            label = "Counts"
            extend = "neither"

        if i < 2:  # First two columns
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmax=vmax)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                pad=0.01,
                label=label,
                extend=extend,
            )
            cbar.ax.yaxis.labelpad = 1.2

        else:  # Last column
            im = ax.imshow(data, aspect="auto", cmap=cmap)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                pad=0.01,
                label="Reconstructed Counts",
            )
            cbar.ax.yaxis.labelpad = 1.2
        if i < 1:
            ax.text(6, 20, panels[j][i], c="white")
        else:
            ax.text(3, 10, panels[j][i], c="white")
        if j == 0:
            if i == 0:
                ax.set_title("Raw Data")
            elif i == 1:
                ax.set_title("Filtered Data")
            else:
                ax.set_title("Reconstructed Signal")
# Adjust layout
fig.tight_layout()
plt.savefig(
    f"{results_folder}summary_1.png",
    dpi=500,
    bbox_inches="tight",
)

plt.clf()
inds = ["0-2", "0-5", "0-7"]
col_names = ["raw", "cleaned", "dc"]

panels = [["a)", "b)", "c)"], ["d)", "e)", "f)"], ["g)", "h)", "i)"]]

# Create subplots with shared x-axis and y-axis
fig, axs = plt.subplots(3, 3, figsize=(8, 6), sharex="col")

# Plot data and colorbars
for i in range(3):
    for j in range(3):
        fname = f"{results_folder}Cd109-{inds[j]}_{col_names[i]}.txt"
        if i == 2:
            data = np.loadtxt(fname) / 11**2
        else:
            data = np.loadtxt(fname)
        ax = axs[j, i]
        ax.set_box_aspect(1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if i == 0:
            vmax = 100
            label = "keV/pixel"
            extend = "max"
        elif i == 1:
            vmax = 3
            label = "Counts"
            extend = "max"
        else:
            vmax = None
            label = "Counts"
            extend = "neither"

        if i < 2:  # First two columns
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmax=vmax)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                pad=0.01,
                label=label,
                extend=extend,
            )
            cbar.ax.yaxis.labelpad = 1.2

        else:  # Last column
            im = ax.imshow(data, aspect="auto", cmap=cmap)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.04,
                pad=0.01,
                label="Reconstructed Counts",
            )
            cbar.ax.yaxis.labelpad = 1.2
        if i < 1:
            ax.text(6, 20, panels[j][i], c="white")
        else:
            ax.text(3, 10, panels[j][i], c="white")

        if j == 0:
            if i == 0:
                ax.set_title("Raw Data")
            elif i == 1:
                ax.set_title("Filtered Data")
            else:
                ax.set_title("Reconstructed Signal")

# Adjust layout
plt.tight_layout()
plt.savefig(
    f"{results_folder}summary_2.png",
    dpi=500,
    bbox_inches="tight",
)


plt.clf()


inds = ["0-2-close", "0-2-far"]
col_names = ["raw", "cleaned", "dc"]

# Create subplots with shared x-axis and y-axis
fig, axs = plt.subplots(2, 3, figsize=(8, 4), sharex="col")

panels = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]

# Plot data and colorbars
for i in range(3):
    for j in range(2):
        fname = f"{results_folder}Cd109-{inds[j]}_{col_names[i]}.txt"
        if i == 2:
            data = np.loadtxt(fname) / 11**2
        else:
            data = np.loadtxt(fname)
        ax = axs[j, i]
        ax.set_box_aspect(1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if i == 0:
            vmax = 100
            label = "keV/pixel"
            extend = "max"
        elif i == 1:
            vmax = 3
            label = "Counts"
            extend = "max"
        else:
            vmax = None
            label = "Counts"
            extend = "neither"
        if i < 2:  # First two columns
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmax=vmax)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                pad=0.01,
                label=label,
                extend=extend,
            )
            cbar.ax.yaxis.labelpad = 1.2

        else:  # Last column
            im = ax.imshow(data, aspect="auto", cmap=cmap)
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.05,
                pad=0.01,
                label="Reconstructed Counts",
            )
            cbar.ax.yaxis.labelpad = 1.2
        if i < 1:
            ax.text(6, 20, panels[j][i], c="white")
        else:
            ax.text(3, 10, panels[j][i], c="white")
        if j == 0:
            if i == 0:
                ax.set_title("Raw Data")
            elif i == 1:
                ax.set_title("Filtered Data")
            else:
                ax.set_title("Reconstructed Signal")
# Adjust layout
fig.tight_layout()
plt.savefig(
    f"{results_folder}summary_3.png",
    dpi=500,
    bbox_inches="tight",
)

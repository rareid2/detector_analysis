import numpy as np
import matplotlib.pyplot as plt

import sys
from matplotlib.patches import Rectangle

sys.path.insert(1, "../detector_analysis/")
import scipy.optimize as opt
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/"

thin_raw_center = results_dir + "47-2-300/47-2.2-0-xy-0_1.00E+06_Mono_600_raw.txt"
thin_dc_center = results_dir + "47-2-300/47-2.2-0-xy-0_1.00E+06_Mono_600_dc.txt"


def calculate_fwhm(image, direction, max_index, scale):
    def gaussian2d(xy, xo, yo, sigma_x, sigma_y, amplitude, offset):
        xo = float(xo)
        yo = float(yo)
        x, y = xy
        gg = offset + amplitude * np.exp(
            -(((x - xo) ** 2) / (2 * sigma_x**2) + ((y - yo) ** 2) / (2 * sigma_y**2))
        )
        return gg.ravel()

    def getFWHM_GaussianFitScaledAmp(img, direction):
        """Get FWHM(x,y) of a blob by 2D gaussian fitting
        Parameter:
            img - image as numpy array
        Returns:
            FWHMs in pixels, along x and y axes.
        """
        x = np.linspace(0, img.shape[1], img.shape[1])
        y = np.linspace(0, img.shape[0], img.shape[0])
        x, y = np.meshgrid(x, y)

        initial_guess = (img.shape[1] / 2, img.shape[0] / 2, 2, 2, 1, 0)
        params, _ = opt.curve_fit(gaussian2d, (x, y), img.ravel(), p0=initial_guess)
        xcenter, ycenter, sigmaX, sigmaY, amp, offset = params

        FWHM_x = 2 * sigmaX * np.sqrt(2 * np.log(2))
        FWHM_y = 2 * sigmaY * np.sqrt(2 * np.log(2))

        # return based on direction
        if direction == "x":
            fwhm = FWHM_y
        elif direction == "y":
            fwhm = FWHM_x
        else:
            if FWHM_x > FWHM_y:
                fwhm = FWHM_x
            else:
                fwhm = FWHM_y

        return fwhm, params

    # get the shifted image and scale
    img = (
        image / scale
    )  # this is the average signal over 0 after shifting for pt source centered
    import matplotlib.pyplot as plt

    sect = 6

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

    # self.plot_3D_signal(img_clipped,save_name='3d.png')
    FWHM, params = getFWHM_GaussianFitScaledAmp(img_clipped, direction)

    # for plotting if interested
    xcenter, ycenter, sigmaX, sigmaY, amp, offset = params
    x = np.linspace(0, img_clipped.shape[1], 100)
    y = np.linspace(0, img_clipped.shape[0], 100)
    x, y = np.meshgrid(x, y)
    g2d = offset + amp * np.exp(
        -(
            ((x - xcenter) ** 2) / (2 * sigmaX**2)
            + ((y - ycenter) ** 2) / (2 * sigmaY**2)
        )
    )

    return FWHM, g2d, x, y


def plot_3d(ax, heatmap, g=False):
    x = np.linspace(-10, 10, heatmap.shape[0])
    y = np.linspace(-10, 10, heatmap.shape[1])
    X, Y = np.meshgrid(x, y)
    print(np.amax(heatmap))
    # Create the surface plot!
    if g:
        # surface = ax.plot_surface(X, Y, heatmap / 1.4038, color="grey", alpha=0)
        ax.plot_surface(
            X,
            Y,
            heatmap / 1.4,
            cmap="Blues",
            linewidth=0.5,
            alpha=1,
        )

    else:
        surface = ax.plot_surface(X, Y, heatmap, cmap=cmap, zorder=1)

    # Add a color bar for reference
    # fig.colorbar(surface)

    # Show the plot
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.set_zlabel("signal")


import cmocean

cmap = cmocean.cm.thermal
# lightcmap = cmocean.tools.crop_by_percent(cmap, 10, which='min', N=None)
# cmap =lightcmap

fig = plt.figure(figsize=(5.7, 3))
grid = (1, 3)

max_raw = np.max(np.loadtxt(thin_raw_center))
max_dc = np.max(np.loadtxt(thin_dc_center))

ax1 = plt.subplot2grid(grid, (0, 0))
ff1 = ax1.imshow(np.loadtxt(thin_raw_center) / max_raw, cmap, vmin=0, vmax=1)
# ax1.set_title("Raw \nImage")
ax1.axis("off")

ax2 = plt.subplot2grid(grid, (0, 1))
ax2.imshow(np.loadtxt(thin_dc_center) / max_dc, cmap, vmin=0, vmax=1)
# ax2.set_title("Reconstructed \nImage")
ax2.axis("off")

inset_ax = ax2.inset_axes([0.7, 0.5, 0.4, 0.4])
inset_ax.imshow(np.loadtxt(thin_dc_center)[66:75, 66:75] / max_dc, cmap, vmin=0, vmax=1)
inset_ax.axis("off")
rect1 = Rectangle(
    (0.45, 0.45),
    0.1,
    0.1,
    edgecolor="white",
    facecolor="none",
    linewidth=1,
    transform=ax2.transAxes,
)
rect2 = Rectangle(
    (0.7, 0.5),
    0.4,
    0.4,
    edgecolor="white",
    facecolor="none",
    linewidth=2,
    transform=ax2.transAxes,
)
ax2.add_patch(rect1)
ax2.add_patch(rect2)

ax3 = plt.subplot2grid(grid, (0, 2), projection="3d")

plot_3d(ax3, np.loadtxt(thin_dc_center)[63:78, 63:78] / max_dc)
max_index = (70, 70)
FWHM, gg, _, _ = calculate_fwhm(np.loadtxt(thin_dc_center) / max_dc, "xy", max_index, 1)
# plot_3d(ax3, gg, g=True)

ax3.view_init(elev=22, azim=23)

ax1.text(73, 154, r"82$^\circ$", fontsize=10, va="center", ha="center")
ax2.text(73, 154, r"82$^\circ$", fontsize=10, va="center", ha="center")
ax3.text(1, -0.5, -0.98, r"$\sim$8$^\circ$", fontsize=10, va="center", ha="center")
ax3.axis("off")

t = ax1.text(15, 15, "a)", fontsize=10, va="center", ha="center")
t.set_bbox(dict(facecolor="white", edgecolor="white"))
t = ax2.text(15, 15, "b)", fontsize=10, va="center", ha="center")
t.set_bbox(dict(facecolor="white", edgecolor="white"))
t = ax3.text(10, -10, 1.45, "c)", fontsize=10, va="center", ha="center")
t.set_bbox(dict(facecolor="white", edgecolor="white"))


ax1.annotate(
    "",
    xy=(-1, 144),
    xytext=(142, 144),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    annotation_clip=False,
)
ax2.annotate(
    "",
    xy=(-1, 144),
    xytext=(142, 144),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    annotation_clip=False,
)
plt.annotate(
    "",
    xy=(-0.07, -0.1),
    xytext=(0.06, -0.1),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    annotation_clip=False,
)

plt.annotate(
    "",
    xy=(-0.07, -0.1),
    xytext=(0.06, -0.1),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1),
    annotation_clip=False,
)

ax1.text(70, -7, "Raw", fontsize=10, va="center", ha="center")
ax2.text(70, -7, "Decoded", fontsize=10, va="center", ha="center")
ax3.text(0, 0, 1.68, "Decoded (zoom, 3D)", fontsize=10, va="center", ha="center")


fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.033, 0.11, 0.92, 0.015])
colorbar = fig.colorbar(ff1, cax=cbar_ax, pad=0.1, orientation="horizontal")
cbar_ax.xaxis.labelpad = 0.2
colorbar.set_label("Normalized signal")
cbar_ax.tick_params(axis="x", labelsize=10)


plt.tight_layout()
plt.savefig(
    "../simulation-results/final-images/4_fwhm_process.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.02,
)

import numpy as np
import matplotlib.pyplot as plt
from plot_settings import *
import os
from polar_mapping import *

dist = "cos"
results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/rings/"
my_image = np.ones((67, 67))


image_data = np.zeros([324, 324]).astype("int")
for i in range(10):
    radii = np.random.rand(10000) * 100
    theta = np.random.rand(10000) * 2 * np.pi
    np.add.at(
        image_data,
        (
            np.rint(161.5 + (radii * np.cos(theta))).astype("int"),
            np.rint(161.5 + (radii * np.sin(theta))).astype("int"),
        ),
        1,
    )
plt.imshow(image_data)
plt.savefig(f"{results_dir}test.png")
plt.clf()

PolarImage, r_grid, phi_grid = reproject_image_into_polar(
    my_image,
    Jacobian=True,
)

pixel_size = 0.02  # cm

fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
im = axs.imshow(
    PolarImage,
    aspect="auto",
    origin="lower",
    extent=(np.min(phi_grid), np.max(phi_grid), np.min(r_grid), np.max(r_grid)),
    cmap=cmap,
)

axs.set_title("Polar")
axs.set_xlabel("Phi")
axs.set_ylabel("r")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

fig.colorbar(im, ax=axs, cax=cbar_ax)

# plt.tight_layout()
plt.savefig(f"{results_dir}polar_mapping-{dist}.png")

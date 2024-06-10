import numpy as np
import matplotlib.pyplot as plt
import cmocean

cmap = cmocean.cm.thermal

results_folder = "/home/rileyannereid/workspace/geant4/simulation-results/mask/"
i = 0
pt_1 = np.loadtxt(f"{results_folder}11-2_1.00E+06_Mono_20_ideal-{i}_dc.txt")
pt_1_max = np.amax(pt_1)

pt_2 = np.loadtxt(f"{results_folder}11-2_1.00E+06_Mono_20_CAD-{i}_dc.txt")
cir_1 = np.loadtxt(f"{results_folder}11-2_1.00E+07_Mono_20_ideal-extend-{i}_dc.txt")
cir_2 = np.loadtxt(f"{results_folder}11-2_1.00E+07_Mono_20_CAD-extend-{i}_dc.txt")
cir_1_max = np.amax(cir_1)
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Plot data and color bars
shrink = 0.8

im = axs[0, 0].imshow(pt_1 / pt_1_max, cmap=cmap)
cbar = fig.colorbar(
    im,
    ax=axs[0, 0],
    orientation="horizontal",
    shrink=shrink,
    pad=0.01,
)
axs[0, 0].axis("off")

im = axs[0, 1].imshow(pt_2 / pt_1_max, cmap=cmap, vmax=np.amax(pt_2) / pt_1_max)
cbar = fig.colorbar(
    im,
    ax=axs[0, 1],
    orientation="horizontal",
    shrink=shrink,
    pad=0.01,
)
axs[0, 1].axis("off")


im = axs[1, 0].imshow(cir_1 / cir_1_max, cmap=cmap)
cbar = fig.colorbar(im, ax=axs[1, 0], orientation="horizontal", shrink=shrink, pad=0.01)
axs[1, 0].axis("off")


im = axs[1, 1].imshow(cir_2 / cir_1_max, cmap=cmap, vmax=np.amax(cir_2) / cir_1_max)
cbar = fig.colorbar(im, ax=axs[1, 1], orientation="horizontal", shrink=shrink, pad=0.01)
axs[1, 1].axis("off")


# Adjust layout
plt.tight_layout()

# Show plot
plt.savefig(f"{results_folder}mask_imperfections.png", dpi=500, bbox_inches="tight")

import sys

sys.path.insert(1, "../detector_analysis")
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *
from matplotlib import ticker
# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = True
simulate = False

trim = None
mosaic = True
multiplier = 3

# designs to evaluate
det_size_cms = [4.956]  # cm
pixels = [0.28]  # mm
n_elements_originals = [59]
thicknesses = [300]
distances = [7.434]
radii = [8.16]
i = 0

energies = np.logspace(2, 6.7, 15) / 1000  # keV
energy = 100
# ------------------- simulation parameters ------------------
#for energy in energies:
det_size_cm = det_size_cms[i]
pixel = pixels[i]
n_elements_original = n_elements_originals[i]
thickness = thicknesses[i]
distance = distances[i]
radius = radii[i]

n_particles = 1e8

pixel_size = pixel * 0.1
element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1
mask_size = element_size * n_elements

# --------------set up simulation---------------
simulation_engine.set_config(
    det1_thickness_um=300,
    det_gap_mm=30,  # gap between first and second (unused detector)
    win_thickness_um=100,  # window is not actually in there
    det_size_cm=det_size_cm,
    n_elements=n_elements,
    mask_thickness_um=thickness,
    mask_gap_cm=distance,
    element_size_mm=element_size,
    mosaic=mosaic,
    mask_size=mask_size,
    radius_cm=radius,
)

# --------------set up source---------------
energy_type = "Mono"
energy_level = energy  # keV

# --------------set up data naming---------------
fname_tag = f"{n_elements_original}-{distance}"

fname = f"../simulation-data/geom-factor/{fname_tag}_{n_particles:.2E}_{energy_type}_{round(energy_level,3)}.csv"

if txt:
    fname = f"../simulation-results/geom-factor/{fname_tag}_{n_particles:.2E}_{energy_type}_{round(energy_level,3)}_raw.txt"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_level, None],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=None,
    ring=False,
    radius_cm=radius,
)

# --------------RUN---------------
if simulate:
    simulation_engine.run_simulation(fname, build=False, rename=True)

# ---------- process results -----------
myhits = Hits(fname=fname, experiment=False, txt_file=txt)
if not txt:
    myhits.get_det_hits(
        remove_secondaries=True, second_axis="y", energy_level=energy_level
    )

# directory to save results in
results_dir = "../simulation-results/geom-factor/"
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{round(energy_level,3)}"
results_save = results_dir + results_tag

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

deconvolver.deconvolve(
    downsample=int(n_elements_original),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=False,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=None,
    hits_txt=txt,
    rotate=False,
    delta_decoding=False,
    apply_noise=False,
    resample_array=True
)
deconvolver2 = Deconvolution(myhits, simulation_engine)

deconvolver2.deconvolve(
    downsample=int(n_elements_original*multiplier),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=False,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=None,
    hits_txt=txt,
    rotate=False,
    delta_decoding=True,
    apply_noise=False,
)
print(np.sum(deconvolver.raw_heatmap))
print(np.sum(np.loadtxt(results_save + "_raw.txt")))
print(
    4
    * np.pi**2
    * radius**2
    * np.sum(np.loadtxt(results_save + "_raw.txt"))
    / n_particles
)

fig, axs = plt.subplots(1, 3, figsize=(5.7, 3))

copper = mpl.colormaps['Greys_r'].resampled(3)
hex_list = ["#000000","#808080","#ffffff",]
copper = get_continuous_cmap(hex_list)
copper = copper.resampled(3)

im1 = axs[0].imshow(deconvolver.mask, cmap=copper,vmin=-1)
axs[0].set_title('Moasic Aperture\n59-Rank',fontsize=10,pad=0.05)
axs[0].axis('off')
rect = patches.Rectangle((28,28), 59, 59, linewidth=2, edgecolor='#87518D', facecolor='none')

# Add the rectangle to the plot
axs[0].add_patch(rect)

im2 = axs[1].imshow(deconvolver.decoder, cmap=copper)
axs[1].set_title('Balanced\nDecoding',fontsize=10,pad=0.05)
axs[1].axis('off')

im3 = axs[2].imshow(deconvolver2.decoder, cmap=copper)
axs[2].set_xlim([0,59*3])
axs[2].set_title('Delta\nDecoding',fontsize=10, pad=0.05)
axs[2].axis('off')

plt.tight_layout()

# Add a colorbar below the three panels
cbar = fig.colorbar(im3, ax=axs, orientation='horizontal', pad=0.01, aspect=80)
cbar.set_ticks([-1,0,1])
cbar.ax.tick_params(labelsize=10)
# Display the plot

plt.savefig('4p1_decoders.png',dpi=500,pad_inches=0.02,bbox_inches='tight')
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from plotting.plot_settings import *

from macros import find_disp_pos
import numpy as np
import os

import matplotlib.pyplot as plt

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# flat field array
# txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400/"
# flat_field = np.loadtxt(f"{txt_folder}interp_grid.txt")
flat_field = None

# general detector design
det_size_cm = 2.68  # cm
pixel = 0.2  # mm

# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 67
multiplier = int(0.4 / pixel)

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 400  # um

# focal length
distance = 2  # cm

dist = "iso"

n_particles = 1e9

raw_hits = np.zeros((122, 122))

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
    radius_cm=1,
)

# --------------set up source---------------
energy_type = "Mono"
energy_level = 100  # keV
i = 0
fname_tag = f"{n_elements_original}-{distance}-iso-{i}-normal"
fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_level, None],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=None,
)

# --------------RUN---------------
simulation_engine.run_simulation(fname, build=False, rename=True)

# ---------- process results -----------
results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/rings/"

# load them for processing
myhits = Hits(fname=fname, experiment=False, txt_file=True)
# myhits.get_det_hits(remove_secondaries=True, second_axis="y", energy_level=energy_level)
deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
results_save = results_dir + results_tag

deconvolver.deconvolve(
    downsample=int(multiplier * n_elements_original),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=flat_field,
    hits_txt=True,
    special=False,
    correct_collimation=True,
)
"""

signal_normal = deconvolver.deconvolved_image

fname_tag = f"{n_elements_original}-{distance}-cos-{i}-rotate"
fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

# load them for processing
myhits = Hits(fname=fname, experiment=False, txt_file=True)
deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
results_save = results_dir + results_tag

deconvolver.deconvolve(
    downsample=int(multiplier * n_elements_original),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=flat_field,
    hits_txt=False,
    special=True,
    correct_collimation=False,
)

signal_rotate = deconvolver.deconvolved_image
flattened_data = signal_normal.flatten()
plt.clf()
plt.hist(flattened_data, bins=50)  # Adjust the number of bins as needed
plt.savefig(f"{results_dir}hist-invert.png")
plt.clf()

# summed image
summed_image = signal_rotate + signal_normal
plt.imshow(summed_image, cmap=cmap)
cbar = plt.colorbar()
plt.savefig(results_dir + "67_summed_image.png")

input_flux = 1e7 / ((3.72066 * 2) ** 2 * 2 * np.pi)
geom_factor = summed_image / input_flux

plt.imshow(geom_factor, cmap=cmap)
cbar = plt.colorbar()
plt.savefig(f"{results_dir}geometric-factor.png")

# histogram
flattened_data = summed_image.flatten()

# Create a histogram of the flattened data
plt.clf()
plt.hist(flattened_data, bins=50)  # Adjust the number of bins as needed
plt.savefig(f"{results_dir}hist.png")
plt.clf()

# now we need to loop through the pixel
# maybe re sample radially first ? convert to polar space

import abel

PolarImage, r_grid, phi_grid = abel.tools.polar.reproject_image_into_polar(
    summed_image, Jacobian=False
)

pixel_size = 0.02  # cm
theta_grid = np.rad2deg(np.arctan(r_grid[:, 0] * pixel_size / 2))

fig, axs = plt.subplots(1, 1, figsize=(7, 3.5))
im = axs.imshow(
    PolarImage,
    aspect="auto",
    origin="lower",
    extent=(np.min(phi_grid), np.max(phi_grid), np.min(theta_grid), np.max(theta_grid)),
    cmap=cmap,
)

axs.set_title("Polar")
axs.set_xlabel("Phi")
axs.set_ylabel("r")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

fig.colorbar(im, ax=axs, cax=cbar_ax)

# plt.tight_layout()
plt.savefig(f"{results_dir}polar_mapping-{dist}-rotate.png")

plt.clf()
x_axis = np.cos(np.deg2rad(theta_grid))
plt.plot(theta_grid, np.sum(PolarImage, 1))  # Adjust the number of bins as needed
# plt.ylim([0, 50000])
plt.ylabel("geometric factor (cm^2 sr)")
plt.savefig(f"{results_dir}PAD-Jacobian.png")
plt.clf()
"""

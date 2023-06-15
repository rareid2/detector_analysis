from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np
import os

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# timepix design
det_size_cm = 1.408
pixel = 0.055  # mm

# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 11
multiplier = 22

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# set edge trim - can't use all pixels to downsample to integer amount
trim = 7
mosaic = True

# -------------------------------------

# -------- pinhole set up -------------
"""
rank = 1
element_size = 1.76/4 # mm
n_elements = rank
mask_size = det_size_cm * 10 # convert to mm
trim = None
mosaic = False
"""
# -------------------------------------

# thickness of mask
thickness = 500  # um

# focal length
distance = 1  # cm

n_particles = 1e9

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
)

# --------------set up source---------------
energy_type = "Mono"
energy_level = 500  # keV

# --------------set up data naming---------------
fname_tag = f"sphere-ca-rotate"
fname = (
    f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"
)

simulation_engine.set_macro(
    n_particles=n_particles,
    energy_keV=[energy_type, energy_level, None],
    sphere=True,
    radius_cm=3.25,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
)

# --------------RUN---------------
simulation_engine.run_simulation(fname, build=True)

# ---------- process results -----------
myhits = Hits(fname=fname, experiment=False)
myhits.get_det_hits()

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_dir = "../simulation-results/validating-iso/"
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
results_save = results_dir + results_tag

deconvolver.deconvolve(
    downsample=2,
    rtim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=True,
    plot_conditions=False,
    save_peak=results_save + "peak.png",
)

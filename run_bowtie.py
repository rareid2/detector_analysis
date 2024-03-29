from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess

from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = False

trim = None
mosaic = True
multiplier = 1

# designs to evaluate
det_size_cms = [4.941]  # cm
pixels = [0.81]  # mm
n_elements_originals = [61]
thicknesses = [200]
distances = [1.9764]
radii = [9.64]
i = 0

bin_edges = np.logspace(2, 3, 24)
bin_centers = []
for bi, be in enumerate(bin_edges[:-1]):
    bc = (bin_edges[bi + 1] - be) / 2 + be
    bin_centers.append(bc)
energies = np.array(bin_centers) / 1000
gfs = []
# convert to keV
# ------------------- simulation parameters ------------------
for ei, energy in enumerate(energies):
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
            remove_secondaries=True,
            second_axis="y",
            energy_level=energy_level,
            energy_bin=[bin_edges[ei] / 1000, bin_edges[ei + 1] / 1000],
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
    )
    print(np.sum(deconvolver.raw_heatmap))
    print(np.sum(np.loadtxt(results_save + "_raw.txt")))
    gf = (
        4
        * np.pi**2
        * radius**2
        * np.sum(np.loadtxt(results_save + "_raw.txt"))
        / n_particles
    )
    print(gf)

    # save GF
    gfs.append(gf)

plt.plot(energies, gfs)

color = "#39329E"
fig, ax1 = plt.subplots(figsize=(5.7, 2))
ax1.set_axisbelow(True)
ax1.grid(True, linestyle="--", color="lightgrey", linewidth=0.5)
ax1.plot(energies * 1000, gfs, color=color, linewidth=2)
ax1.set_xscale("log")
ax1.set_xlabel("Electron Energy [eV]", fontsize=8, color="black")
ax1.tick_params(axis="both", labelsize=8)
ax1.xaxis.labelpad = 0.2
ax1.yaxis.labelpad = 0.2
plt.savefig(
    "../simulation-results/final-images/geometric_factors_strahl_no_secondary.png",
    dpi=500,
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.02,
)

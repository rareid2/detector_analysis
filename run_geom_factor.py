from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
import subprocess
import copy

from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = True
nthreads = 14

trim = None
mosaic = True
multiplier = 1

# designs to evaluate
det_size_cms = [4.94]  # cm
pixels = [3.8]  # mm
n_elements_originals = [13]
thicknesses = [100]
distances = [4.3]
radii = [7.3]
i = 0

energies = [0.235]
# ------------------- simulation parameters ------------------
for energy in energies:
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
    if not txt:
        for hi in range(nthreads):
            print(hi)
            fname_hits = fname[:-4] + "-{}.csv".format(hi)
            myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
            hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
                remove_secondaries=True, second_axis="y", det_thick_cm=0.03
            )
            if hi != 0:
                # update fields in hits dict
                myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
                myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])

                hits_copy = copy.copy(myhits)
            else:
                hits_copy = copy.copy(myhits)
    else:
        myhits = Hits(fname=fname, experiment=False, txt_file=txt)

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
    print(
        4
        * np.pi**2
        * radius**2
        * np.sum(np.loadtxt(results_save + "_raw.txt"))
        / n_particles
    )

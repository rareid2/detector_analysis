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
from plotting.plot_settings import *
import copy

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = False

# general detector design
det_size_cm = 2.19  # cm
pixel = 0.1  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 73
multiplier = 3

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

thetas = [
    1.145,
    2.865,
    4.575,
    6.275,
    7.965,
    9.645,
    11.305,
    12.955,
    14.575,
    16.175,
    17.745,
    19.295,
    20.805,
    22.295,
    23.745,
    25.175,
    26.565,
    27.925,
    29.245,
    30.545,
    31.795,
    33.025,
]

current_hits = np.zeros((201, 201))
raw_hits = []
thetas = [2.865, 11.305, 19.295, 26.565, 33.025]

# for pinhole
# n_elements_original = 1
# n_elements = 1
# mosaic = False
# mask_size = 21.9

thetas = [thetas[-1]]

# ------------------- simulation parameters ------------------
for ii, theta in enumerate(thetas):
    print(theta)

    # simulate 3e8 per cm^2 per sr per s
    n_particles = int(3e8 * (1 - np.cos(np.deg2rad(theta))) * (2.445 * 2) ** 2)

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

    # --------------set up data naming---------------
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle"

    fname = f"../simulation-data/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

    if txt:
        fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
        ring=True,
    )

    # --------------RUN---------------
    if simulate:
        simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    raw_txt = np.zeros((219, 219))
    if not txt:
        ii = 1
        nlines = 10000000
        total_lines = 112318271
        for nstart in range(0, total_lines, nlines):
            # directory to save results in
            results_dir = "../simulation-results/rings/"
            results_tag = (
                f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{ii}"
            )
            results_save = results_dir + results_tag
            raw_txt += np.loadtxt(results_save + "_raw.txt")

            """
            myhits = Hits(
                fname=fname,
                experiment=False,
                txt_file=txt,
                nlines=nlines,
                nstart=nstart,
            )
            myhits.get_det_hits(
                remove_secondaries=True,
                second_axis="y",
                energy_level=energy_level,
            )
            """
            ii += 1

        # now save it
        np.savetxt(fname, raw_txt)

        txt = True

        results_dir = "../simulation-results/rings/"
        results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
        results_save = results_dir + results_tag

        myhits = Hits(
            fname=fname,
            experiment=False,
            txt_file=txt,
        )

        # deconvolution steps
        deconvolver = Deconvolution(myhits, simulation_engine)

        deconvolver.deconvolve(
            downsample=int(3 * n_elements_original),
            trim=trim,
            vmax=None,
            plot_deconvolved_heatmap=True,
            plot_raw_heatmap=True,
            save_raw_heatmap=results_save + "_raw.png",
            save_deconvolve_heatmap=results_save + "_dc.png",
            plot_signal_peak=False,
            plot_conditions=False,
            flat_field_array=None,
            hits_txt=txt,
            rotate=True,
            delta_decoding=True,
        )

        raw_hits.append(np.sum(deconvolver.rawIm))
        print(np.sum(deconvolver.raw_heatmap))
        print(np.sum(deconvolver.deconvolved_image))

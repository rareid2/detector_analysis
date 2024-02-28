from scipy import interpolate

import numpy as np
import os
import subprocess

from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plot_settings import *

import sys

sys.path.insert(1, "../detector_analysis")
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from run_part_csv import read_csv_in_parts

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

txt = True

# general detector design
det_size_cm = 4.956  # cm
pixel = 0.28  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 59
multiplier = 3

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 300  # um

# focal length
distance = 3.47  # cm

theta = 45
n_particles_frac = np.logspace(0, -2, 8)
n_particles_frac = n_particles_frac[1:]

# first, combine raw txt files with non-rotated ones

fname_fcfov = "../simulation-results/rings/59-3.47-22p00-deg_5.90E+08_Mono_100_raw.txt"
fcfov = np.loadtxt(fname_fcfov)

# combine it with the pcfov
# ------------------- simulation parameters ------------------
for ii, npf in enumerate(n_particles_frac[:1]):
    n_particles = (
        int(
            (8e7 * (5.030 * 2) ** 2)
            * (np.cos(np.deg2rad(theta)) - np.cos(np.deg2rad(70)))
        )
        * npf
    )
    
    print(f"{n_particles:.2E}")
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
    fname_tag_pcfov = f"{n_elements_original}-{distance}-{formatted_theta}-deg"
    fname_pcfov = f"../simulation-results/rings/{fname_tag_pcfov}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
    pcfov = np.loadtxt(fname_pcfov)

    fname_tag = f"{n_elements_original}-{distance}-22-45-deg-fov"
    fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

    # save the fov together
    # downsample pcfov temporarily
    """
    original_size = len(pcfov)

    new_array = np.zeros((len(pcfov) // 3, len(pcfov) // 3))

    for i in range(0, original_size, 3):
        k = i // 3
        for j in range(0, original_size, 3):
            n = j // 3
            new_array[k, n] = np.sum(fcfov[i : i + 3, j : j + 3])
    fcfov = new_array
    """
    np.savetxt(fname, fcfov + pcfov)

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
        # radius_cm=3,
    )

    # ---------- process results -----------
    # directory to save results in
    results_dir = "../simulation-results/rings/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

    myhits = Hits(fname=fname, experiment=False, txt_file=txt)

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    deconvolver.deconvolve(
        downsample=int(n_elements_original),
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
        delta_decoding=False,
        apply_noise=False,
        resample_array=True,
    )
  
    # combine with the pcfov rotated
    fcfov_rotated = np.loadtxt(
        f"../simulation-results/rings/59-3.47-46p00-deg-rot_1.69E+09_Mono_100_dc.txt"
    )
    pcfov_rotated = np.loadtxt(
        f"../simulation-results/rings/59-3.47-45p00-deg-rotate_{n_particles:.2E}_Mono_100_dc.txt"
    )
    pcfov = np.loadtxt(
        f"../simulation-results/rings/59-3.47-45p00-deg_{n_particles:.2E}_Mono_100_dc.txt"
    )

    pcfov_combined = pcfov+pcfov_rotated
    plt.imshow(pcfov_combined, cmap=cmap)
    plt.colorbar()
    plt.savefig(results_save + "_combined_pcfov.png", dpi=300)
    plt.clf()
    

    # combine them
    signal = np.loadtxt("../simulation-results/rings/59-3.47-22p00-deg_5.90E+08_Mono_100_dc.txt")

    combined_final = deconvolver.deconvolved_image + pcfov_rotated
    #combined_final = signal + pcfov_combined #+ fcfov_rotated
    
    plt.imshow(combined_final, cmap=cmap)
    plt.colorbar()
    plt.savefig(results_save + "_combined.png", dpi=300)

    np.savetxt(results_save+"_combined.txt", combined_final)

    plt.clf()
    # now find the percent bad

    #signal = fcfov_rotated + signal
    plt.imshow(signal, cmap=cmap)
    plt.colorbar()
    plt.savefig(results_save + "_signal.png", dpi=300)
    plt.clf()
    pixel_count = int(n_elements_original)
    max_value = np.max(signal)
    signal_count = 0
    total_count = 0
    center_pixel = int(59 / 2)
    geometric_factor = 18
    pcfov_signal = 0
    pixel_size = 0.28 * multiplier * 0.1

    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            if np.rad2deg(angle) < (22 + 0.5):
                signal_count += 1
                total_count += signal[y, x]
                pcfov_signal += combined_final[y, x]

                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )
                #plt.gca().add_patch(rect)

    #plt.imshow(signal, cmap=cmap)
    #plt.colorbar()
    #plt.show()

    px_factor = signal_count / (pixel_count**2)
    print("recorded flux", total_count / (geometric_factor * px_factor))
    print("new flux", pcfov_signal / (geometric_factor * px_factor))
    print(
        "error", 100
        * (
            (pcfov_signal / (geometric_factor * px_factor))
            / (total_count / (geometric_factor * px_factor))
        ),
    )

# the for sure right way to do this would be to combine the rotated and not rotated, deconvolve both, then add together

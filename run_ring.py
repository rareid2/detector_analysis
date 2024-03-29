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

raw_5 = np.loadtxt("../simulation-results/rings/89-4.49-5p12_1.00E+08_Mono_100_raw.txt")
raw_25 = np.loadtxt("../simulation-results/rings/89-4.49-24p92_1.00E+08_Mono_100_raw.txt")
pts = np.loadtxt("../simulation-results/89-4-300/89-4.49-82-xy-0_1.00E+06_Mono_100_raw.txt")
pts2 = np.loadtxt("../simulation-results/89-4-300/89-4.49-14.65-xy-0_1.00E+06_Mono_100_raw.txt") # keep 15.6 -0.2
pts3 = np.loadtxt("../simulation-results/89-4-300/89-4.49-15.6-xy-0_1.00E+06_Mono_100_raw.txt") # keep 15.6 -0.2
pts4 = np.loadtxt("../simulation-results/89-4-300/89-4.49-77-xy-0_1.00E+06_Mono_100_raw.txt")
combined = (raw_5+raw_25+pts+pts2+pts4+pts3) / 15
np.savetxt("../simulation-results/rings/89-4.49-ring-pt_1.00E+08_Mono_100_raw.txt", combined)

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = True
simulate = False

# general detector design
det_size_cm = 4.984  # cm
pixel = 0.1866666666667  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 89
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
distance = 4.49  # cm

thetas = [5.125,24.93]

# for pinhole
# n_elements_original = 1
# n_elements = 1
# mosaic = False
# mask_size = 21.9

# ------------------- simulation parameters ------------------
for ii, theta in enumerate(thetas[1:]):
    print(theta)
    # simulate 3e8 per cm^2 per sr per s
    # 3e8 for full circle * sr * cm
    # 1e7 for ring * cm
    # 5e8 for sphere
    n_particles = 1e8

    #if theta > 30:
    #    large_file = True
    #    txt = False

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
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}"
    fname_tag = f"{n_elements_original}-{distance}-ring-pt"

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
        # radius_cm=3,
    )

    # --------------RUN---------------
    if simulate:
        simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    # directory to save results in
    results_dir = "../simulation-results/rings/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

    myhits = Hits(fname=fname, experiment=False, txt_file=txt)
    if not txt:
        myhits.get_det_hits(
            remove_secondaries=True, second_axis="y", energy_level=energy_level
        )

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    deconvolver.deconvolve(
        downsample=int(n_elements_original*multiplier),
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
        apply_noise=False,
    )
    np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)
    print(results_save + "_dc.txt")
    print(np.sum(deconvolver.raw_heatmap))
    print(np.sum(deconvolver.deconvolved_image))
    """
    signal = deconvolver.deconvolved_image
    pixel_count = int(n_elements_original*multiplier)
    max_value = np.max(signal)
    signal_count = 0
    total_count = 0
    center_pixel = int(int(n_elements_original*multiplier) / 2)
    geometric_factor = 1

    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            if np.rad2deg(angle) < (theta + 0.4) and np.rad2deg(angle) > (theta - 0.4):
                signal_count += 1
                total_count += signal[y, x]

                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

    plt.imshow(signal, cmap=cmap)
    plt.colorbar()
    plt.show()

    px_factor = signal_count / (pixel_count**2)
    print("recorded flux", total_count )#/ (geometric_factor * px_factor))
    """


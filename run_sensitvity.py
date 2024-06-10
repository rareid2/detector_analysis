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
from run_part_csv import read_csv_in_parts
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


geom_factor = 26  # cm^2 sr
# construct = CA and TD
# source = DS and PS
nthreads = 7
simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = False

# general detector design
det_size_cm = 4.956  # cm
pixel = 0.84  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 59
multiplier = 1

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 100  # um

# focal length
distance = 0.923  # cm

# run each central theta
pixel_size = multiplier * pixel_size

thetas = np.linspace(0.01, 65, 14)

params = [1.48515239e-04, 6.90018731e-03, 8.08592346e-01]
grid_size = 59 * 3
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt(((x - center) / 3) ** 2 + ((y - center) / 3) ** 2)
# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)
fwhm_grid_save = fwhm_grid

fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
gf_grid = geom_factor * fwhm_grid / np.sum(fwhm_grid)

theta_lower = 0

# for processing
pixel_count = int(n_elements_original * 3)
center_pixel = int(pixel_count // 2)
pixel_size = 0.028
sensitivity = []
# ------------------- simulation parameters ------------------
for nn, theta in enumerate(thetas):

    # get the FWHM here
    # get radial distance from center
    radial_d_theta = np.tan(np.deg2rad(theta)) * distance / pixel_size  # pixels
    fwhm_theta = polynomial_function(
        radial_d_theta, *params
    )  # get current FWHM at edge

    fwhm_theta_deg = np.rad2deg(np.arctan(fwhm_theta * 3 * pixel_size / distance))
    theta_bound = theta + np.ceil(fwhm_theta_deg)

    signal_count = 0
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            # add a half bin size
            if theta_lower <= np.rad2deg(angle) <= theta_bound:
                signal_count += gf_grid[y, x]

    n_particles = int(1e6 * (5.265 * 2) ** 2)
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
    energy_level = 0.235  # keV

    # --------------set up data naming---------------
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg"

    fname = f"../simulation-data/strahl/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

    if txt:
        fname = f"../simulation-results/strahl/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
        theta_lower=0,
        ring=True,
        # radius_cm=3,
    )

    # --------------RUN---------------
    if simulate:
        simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    # directory to save results in
    results_dir = "../simulation-results/strahl/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_inter"
    results_save = results_dir + results_tag

    if not txt:
        for hi in range(nthreads):
            print(hi)
            fname_hits = fname[:-4] + "-{}.csv".format(hi + 7)
            myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
            hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
                remove_secondaries=False, second_axis="y"
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

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    deconvolver.deconvolve(
        downsample=int(n_elements_original * 3),
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
    np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)
    print("raw counts", np.sum(deconvolver.raw_heatmap))

    signal_sum = 0
    gf_factor = 0

    signal = deconvolver.deconvolved_image / 9  # for unpsample
    plt.imshow(signal)
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)
            angle = np.rad2deg(angle)

            if theta_lower <= angle <= theta_bound:
                signal_sum += signal[y, x]
                gf_factor += gf_grid[y, x]
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor="none",
                )

                plt.gca().add_patch(rect)
    plt.xlim([75, 100])
    plt.ylim([75, 100])
    plt.savefig(f"{results_save}_test.png")

    print("AH YES the signal", signal_sum)
    print("reconstructed flux is ", signal_sum / signal_count)
    sensitivity.append(signal_sum / signal_count)

np.savetxt(f"{results_dir}/sensitivity.txt", np.array(sensitivity))

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

# construct = CA and TD
# source = DS and PS


def fcfov_plane(
    detector_size_mm: float,
    mask_size_mm: float,
    mask_detector_distance_mm: float,
    mask_plane_distance_mm: float,
    theta: float,
):
    detector_diagonal_mm = detector_size_mm * np.sqrt(2)
    mask_diagonal_mm = mask_size_mm * np.sqrt(2)

    detector_half_diagonal_mm = detector_diagonal_mm / 2
    mask_half_diagonal_mm = mask_diagonal_mm / 2

    # FCFOV half angle
    theta_fcfov_deg = np.rad2deg(
        np.arctan(
            (mask_half_diagonal_mm - detector_half_diagonal_mm)
            / mask_detector_distance_mm
        )
    )
    # print(theta_fcfov_deg, "half angle")

    # pcfov
    fov = np.rad2deg(
        np.arctan(
            (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
            / mask_detector_distance_mm
        )
    )
    # print("PCFOV", fov - theta_fcfov_deg, "Half angle pcfov")
    # project this to a distance
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    additional_diagonal_mm = np.tan(np.deg2rad(theta)) * plane_distance_to_detector_mm

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2

    return plane_half_side_length_mm


def polynomial_function(x, *coefficients):
    return np.polyval(coefficients, x)


def resample(array):
    original_size = len(array)

    new_array = np.zeros((len(array) // 3, len(array) // 3))

    for i in range(0, original_size, 3):
        k = i // 3
        for j in range(0, original_size, 3):
            n = j // 3
            new_array[k, n] = np.sum(array[i : i + 3, j : j + 3])
    array = new_array
    return array


simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
nthreads = 14
txt = False
simulate = True
large_file = False

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
thickness = 300  # um

# focal length
distance = 3.47  # cm

thetas = [2, 13, 24]
geometric_factor = 18
# import FWHM
params = [4.72336124e-04, -2.83882554e-03, 8.86278977e-01]
grid_size = 59
center = (grid_size - 1) / 2  # Center pixel in a 0-indexed grid
# Create a meshgrid representing the X and Y coordinates of each pixel
x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
# Calculate the radial distance from the center for each pixel
radial_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
# now i have radiatl distance, use the FWHM thing
fwhm_grid = polynomial_function(radial_distance, *params)
# need to normalize to 1
fwhm_grid = 2 - (fwhm_grid / np.min(fwhm_grid))
# make it sum to 18
gf_grid = 18 * fwhm_grid / np.sum(fwhm_grid)

mask_size_mm = 98.28
mask_plane_distance_mm = 5  # half mask

# ------------------- simulation parameters ------------------
theta_lower = 0
for ii, theta in enumerate(thetas[:1]):
    theta = 0.1
    pixel_count = 59
    center_pixel = 59 // 2
    signal_count = 0
    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)
            angle = np.arctan(aa / distance)

            if theta_lower <= np.rad2deg(angle) < theta + 0.5:
                # signal_count += gf_grid[y, x]
                signal_count += 1

    og_gf = signal_count
    px_factor = geometric_factor * signal_count / (pixel_count**2)
    n_particles = int(
        5e6 * (5.265 * 2) ** 2 * 2 * np.pi * (1 - np.cos(np.deg2rad(theta)))
    )

    plane_size_mm = fcfov_plane(
        det_size_cm * 10, mask_size_mm, distance * 10, mask_plane_distance_mm, theta
    )
    # run the flux desired (1e6) * size of plane
    plane_size_mm = 52.65
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
    energy_level = 500  # keV

    # --------------set up data naming---------------
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = "{}-{}-{}-deg-test".format(
        n_elements_original, distance, formatted_theta
    )

    fname = "../simulation-data/circle-test/{}_{:.2E}_{}_{}_{}.csv".format(
        fname_tag, n_particles, energy_type, energy_level, formatted_theta
    )

    if txt:
        fname = "../simulation-results/circle-test/{}_{:.2E}_{}_{}_raw.txt".format(
            fname_tag, n_particles, energy_type, energy_level
        )

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
        theta_lower=theta_lower,
        ring=True,
        plane_size_cm=round(plane_size_mm / 10, 3),
        # radius_cm=3,
    )

    # --------------RUN---------------
    if simulate:
        simulation_engine.run_simulation(fname, build=True, rename=True)

    # ---------- process results -----------
    # directory to save results in
    results_dir = "../simulation-results/circle-test/"
    results_tag = "{}_{:.2E}_{}_{}".format(
        fname_tag, n_particles, energy_type, energy_level
    )
    results_save = results_dir + results_tag

    if not txt:
        for hi in range(nthreads):
            fname_hits = fname[:-4] + "-{}.csv".format(hi)
            myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
            hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
                remove_secondaries=False,
                second_axis="y",
            )
            print(sec_brehm, sec_e, len(myhits.hits_dict["Position"]))
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
        resample_array=False,
    )
    np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)

    print(np.sum(deconvolver.raw_heatmap))
    print(np.sum(deconvolver.deconvolved_image))

    signal = deconvolver.deconvolved_image
    max_value = np.max(signal)
    signal_count = 0
    total_count = 0
    geometric_factor = 18

    for x in range(pixel_count):
        for y in range(pixel_count):
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            # find the geometrical theta angle of the pixel
            angle = np.arctan(aa / distance)

            if np.rad2deg(angle) < (theta + 0.5):
                signal_count += gf_grid[y, x]
                # signal_count += 1

                total_count += signal[y, x]

                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor="none",
                )

                # plt.gca().add_patch(rect)

    # plt.imshow(signal, cmap=cmap)
    # plt.colorbar()
    # plt.show()

    # px_factor = geometric_factor * signal_count / (pixel_count**2)
    print("OG GF FAC", px_factor)
    print("GF FAC", signal_count)
    print("recorded flux", total_count / px_factor)

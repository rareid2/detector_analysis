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

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)


# FWHM array
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/67-2-fwhm/"
fwhm = np.loadtxt(f"{txt_folder}fwhm_interp_grid_instrument-only_edges-removed_1d.txt")

radial_1D = np.linspace(0, 2 * 45.254833995939045, 100)
# f = interpolate.interp1d(radial_1D, fwhm)

txt = True
run_rotate = False
simulate = False
combine = False

# general detector design
det_size_cm = 2.68  # cm
pixel = 0.2  # mm
pixel_size = pixel * 0.1

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
"""
inds = []
# first, which pixel do we want to hit
thetas = []
first_theta_ind = 2
# amount of spreading
first_theta_fwhm = f(first_theta_ind)
first_theta = np.rad2deg(np.arctan(first_theta_ind * pixel_size / distance))
next_theta_ind = (
    np.ceil((first_theta_fwhm / 2) + first_theta_ind) + 1
)  # expected spreading

thetas.append(first_theta)
las_theta = 0
inds.append(first_theta_ind)
inds.append(next_theta_ind)

rects = []
uncertainties = []

while next_theta_ind < 67:
    # now get next theta using nextx fwhm

    next_theta_fwhm = f(next_theta_ind)

    next_theta = np.rad2deg(np.arctan(next_theta_ind * pixel_size / distance))
    thetas.append(next_theta)

    # then compute the next theta
    next_theta_ind = np.ceil((next_theta_fwhm / 2) + next_theta_ind) + 1

    # print(next_theta-las_theta)
    las_theta = next_theta
    inds.append(next_theta_ind)
"""
n_particles = 1e8
nhits = []
# color_ring = ["#FF006E", "#FB5607", "#FFBE0B", "#3A86FF", "#8338EC"]
# color_ring = list(next(cycle([color_ring])) for _ in range(len(thetas)))
# color_ring = [item for sublist in color_ring for item in sublist]

# inds = inds[10:11]
thetas = [17.74]

# ------------------- simulation parameters ------------------
for ii, theta in enumerate(thetas):
    print(theta)
    angles = []
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
    # formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    formatted_theta = "17p74"
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-d2-3p98"

    # only is simulating rotated
    if run_rotate:
        fname_tag = (
            f"{n_elements_original}-{distance}-{formatted_theta}-deg-d2-3p98-rotate"
        )

    fname = f"../simulation-data/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

    # if processing combined
    fname_tag_r = (
        f"{n_elements_original}-{distance}-{formatted_theta}-deg-d2-3p98-rotate"
    )
    fname_r = f"../simulation-data/rings/{fname_tag_r}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

    if txt:
        fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
        # only used if doing combined
        fname_r = f"../simulation-results/rings/{fname_tag_r}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

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
    myhits = Hits(fname=fname, experiment=False, txt_file=txt)
    if not txt:
        myhits.get_det_hits(
            remove_secondaries=True, second_axis="y", energy_level=energy_level
        )

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    # directory to save results in
    results_dir = "../simulation-results/rings/"
    results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

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
        rotate=False,
        delta_decoding=True,
    )
    signal = deconvolver.deconvolved_image
    deconvolver.lucy_richardson(results_save + "LR.png")
    data = {"signal": signal}

    # Specify the file path where you want to save the .mat file
    file_path = "signal.mat"

    # Save the grid data as a .mat file
    savemat(file_path, data)

    deconvolver.export_to_matlab()

    print(np.sum(signal))
    print(np.sum(deconvolver.raw_heatmap))

    if combine:
        myhits = Hits(fname=fname_r, experiment=False, txt_file=txt)
        if not txt:
            myhits.get_det_hits(
                remove_secondaries=True, second_axis="y", energy_level=energy_level
            )

        # deconvolution steps
        deconvolver = Deconvolution(myhits, simulation_engine)

        # directory to save results in
        results_dir = "../simulation-results/rings/"
        results_tag = f"{fname_tag_r}_{n_particles:.2E}_{energy_type}_{energy_level}"
        results_save = results_dir + results_tag

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

        signal_r = deconvolver.deconvolved_image

        fname_tag_c = (
            f"{n_elements_original}-{distance}-{formatted_theta}-deg-d2-3p98-combine"
        )
        results_tag_c = f"{fname_tag_c}_{n_particles:.2E}_{energy_type}_{energy_level}"
        results_save_c = results_dir + results_tag_c

        signal = signal_r + signal
        print(np.sum(signal))

        # save it as well
        plt.clf()
        plt.imshow(signal, cmap=cmap)
        plt.colorbar()
        plt.savefig(results_save_c + ".png")
        plt.clf()
    """
    # for now, take a small upper section without banding
    # subtract RMS!
    noise_floor = np.mean(signal[:20, :20])

    # subtract the noise floor
    signal = signal - noise_floor

    max_value = np.max(signal)

    pixel_count = 134
    center_pixel = 67
    rx_rect = []
    for x in range(pixel_count):
        for y in range(pixel_count):
            # find pixels with signal over threshold
            if signal[y, x] > max_value / 4:
                relative_x = (x - center_pixel) * pixel_size
                relative_y = (y - center_pixel) * pixel_size

                aa = np.sqrt(relative_x**2 + relative_y**2)

                # find the geometrical theta angle of the pixel
                angle = np.arctan(aa / distance)

                # largest expected distance is 3 pixels  -- check that we are within that
                largest_expected_px_distance = np.rad2deg(
                    np.arctan(2.7 * pixel_size / distance)
                )
                if np.abs(np.rad2deg(angle) - (theta)) < largest_expected_px_distance:
                    # for plotting the pixels that are identified with signal
                    rx_rect.append((x, y))
                    angles.append(angle)
                else:
                    # print("got some noise")
                    pass
    rects.append(rx_rect)

    plt.clf()
    for ri, rr in enumerate(rects):
        for rx in rr:
            rx = patches.Rectangle(
                (rx[0] - 0.5, rx[1] - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor=color_ring[ri],
                facecolor=color_ring[ri],
                alpha=0.5,
            )
            plt.gca().add_patch(rx)

    plt.imshow(np.zeros_like(signal), cmap="gray_r")
    plt.xlim([67 - (ind + 4), 67 + (ind + 4)])
    plt.ylim([67 - (ind + 4), 67 + (ind + 4)])

    plt.savefig(
        f"/home/rileyannereid/workspace/geant4/simulation-results/rings/indices{ii}.png",
        dpi=800,
    )

    uncertainties.append(np.abs((min(angles) - max(angles)) / 2))

np.savetxt(f"{results_dir}uncertainties.txt", np.array(uncertainties))
np.savetxt(f"{results_dir}thetas.txt", np.array(thetas))

with open(f"{results_dir}inds.txt", "w") as file:
    for sublist in rects:
        for x, y in sublist:
            # Convert the tuple to a formatted string (e.g., "1,2")
            line = f"{x},{y}\n"
            file.write(line)
        # Add a newline character after each sublist
        file.write("\n")
"""

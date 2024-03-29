import sys

sys.path.insert(1, "../detector_analysis")
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from plotting.calculate_solid_angle import get_sr, get_total_sr

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import os
import copy

# construct = CA and TD
# source = DS and PS

plot = True
from plotting.plot_settings import *

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

# general detector design
det_size_cm = 2.68  # cm
pixel = 0.2  # mm

start = 0
end = 43
step = 1.1459
thetas = [start + i * step for i in range(int((end - start) / step) + 1)]

imaginary_plane_k = 2.0303

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

fake_radius = 1

pixel_size = pixel * 0.1  # cm
pixel_count = n_elements_original * multiplier

# ---------- flat field array loading -------
# Load x and y values from separate text files
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/67-2-fwhm/"

hits = True
remove_edges = True

if remove_edges:
    edges_str = "edges-removed"
else:
    edges_str = "edges-inc"
if hits:
    hits_str = "instrument-only"
else:
    hits_str = "shielding-inc"

data_product = "fwhm"
output_name = f"{txt_folder}{data_product}_interp_grid_{hits_str}_{edges_str}"
fwhm_map = np.loadtxt(f"{output_name}.txt")

data_product = "signal"
output_name = f"{txt_folder}{data_product}_interp_grid_{hits_str}_{edges_str}"
flatfield_map = np.loadtxt(f"{output_name}.txt")

#data_product = "hits-ratio"
#output_name = f"{txt_folder}{data_product}_interp_grid_{hits_str}_{edges_str}"
#hitsratio_map = np.loadtxt(f"{output_name}.txt")

# geometric pixel counting method
signals_geo = []
uncertanties_geo = []
central_angles_geo = []
all_indices_geo = []

# simulation method (using source)
uncertanties_sim = []

# check raw hits
raw_hits_len = []
px_counts = []
# ------------------- simulation parameters ------------------
n_particles = 1e7
ii = 0
theta_start = 1
theta_end = 2
thetas_run = thetas[theta_start:theta_end]
for theta in thetas_run:
    print("running ", theta)
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
        radius_cm=fake_radius,
    )

    # --------------set up source---------------
    energy_type = "Mono"
    energy_level = 100  # keV

    # --------------set up data naming---------------
    formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-d2-3p98"
    fname = f"/home/rileyannereid/workspace/geant4/simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
    raw_hits = np.sum(np.loadtxt(fname))

    raw_hits_len.append(np.sum(np.loadtxt(fname)))

    simulation_engine.set_macro(
        n_particles=int(n_particles),
        energy_keV=[energy_type, energy_level, None],
        surface=True,
        progress_mod=int(n_particles / 10),  # set with 10 steps
        fname_tag=fname_tag,
        confine=False,
        detector_dim=det_size_cm,
        theta=theta,
    )

    # ---------- process results -----------

    myhits = Hits(fname=fname, experiment=False, txt_file=True)
    """
    # for combined
    # check if not first iteration
    if ii != 0:
        myhits.txt_hits += hits_copy.txt_hits
        hits_copy = copy.copy(myhits)
    else:
        hits_copy = copy.copy(myhits)

    print(fname_tag)

    ii += 1
    """

    # deconvolution steps
    deconvolver = Deconvolution(myhits, simulation_engine)

    # directory to save results in
    results_dir = "../simulation-results/rings/"
    results_tag = f"{fname_tag}-{n_particles:.2E}_{energy_type}_{energy_level}"
    results_save = results_dir + results_tag

    deconvolver.deconvolve(
        downsample=int(multiplier * n_elements_original),
        trim=trim,
        vmax=None,
        plot_deconvolved_heatmap=False,
        plot_raw_heatmap=False,
        flat_field_array=None,
        hits_txt=True,
    )

    # mx, my = deconvolver.reverse_raytrace()
    # deconvolver.export_to_matlab(np.column_stack((mx, my)))

    signal = deconvolver.deconvolved_image

    # for now, take a small upper section without banding
    # subtract RMS!
    noise_floor = np.mean(signal[:20, :20])

    # subtract the noise floor
    signal = signal - noise_floor

    # if plot:
    #    plt.imshow(signal,cmap=cmap)
    #    plt.colorbar()

    # save the geometrical angles that correspond to each pixel with signal
    angles_geo = []
    px_coverage_geo = []
    signal_sum = 0

    center_pixel = 67  # 122 pixels, center is between index 60 and 61 ????
    indices = []

    # find max signal and relative location
    max_value = np.max(signal)
    max_indices = np.argwhere(signal == max_value)
    max_indices = max_indices[0]

    relative_xmax = (max_indices[1] - center_pixel) * pixel_size
    relative_ymax = (max_indices[0] - center_pixel) * pixel_size

    # calculate central angle based on pixel w max signal
    aamax = np.sqrt(relative_xmax**2 + relative_ymax**2)
    central_angle = np.arctan(aamax / distance)
    central_angles_geo.append(np.rad2deg(central_angle))
    phi = 0

    plot_map = np.zeros_like(signal)
    px_count = 0
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
                if (
                    np.abs(np.rad2deg(angle) - np.rad2deg(central_angle))
                    < largest_expected_px_distance
                ):
                    px_count += 1
                    # save the geometrical angle and the signal of the pixel
                    angles_geo.append(np.rad2deg(angle))

                    # save the indices of the pixel for later
                    indices.append((y, x))

                    # and get the solid angle coverage of the pixel
                    fwhm = fwhm_map[y, x] * pixel_size

                    sphere_radius = 8
                    #sr = get_sr(
                    #    imaginary_plane_k,
                    #    pixel,
                    #    pixel,
                    #    (relative_x, relative_y),
                    #    sphere_radius,
                    #    deconvolver.mask,
                    #    calc_pinhole=False,
                    #    girard=False,
                    #)

                    # units are now counts per solid angle per cm^2
                    # get the calibration to counts
                    #hits_ratio = hitsratio_map[y, x]
                    signal_sum += pixel_size**2
                    #plot_map[y, x] = sr

                    if plot:
                        #for plotting the pixels that are identified with signal
                        rect = patches.Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            linewidth=2,
                            edgecolor="black",
                            facecolor="none",
                        )
                        plt.gca().add_patch(rect)
    if plot:
        plt.imshow(signal, cmap=cmap)
        plt.colorbar()
        plt.show()
"""
    # find the uncertainty in the angle with the geometrical method as the angular spread
    uncertanties_geo.append(np.abs((min(angles_geo) - max(angles_geo)) / 2))

    # now find the corresponding FWHM based on the pixel with max signal
    uncertanties_sim.append(
        np.rad2deg(
            np.arctan(pixel_size * fwhm_map[max_indices[0], max_indices[1]] / distance)
        )
    )

    # save the total signal strength
    signals_geo.append(signal_sum)

    # save the indices of the pixels with signal for later
    all_indices_geo.append(indices)

    ii += 1
    px_counts.append(px_count)

central_angles_sim = thetas_run

# create array
signals_geo = np.array(signals_geo)  # / np.array(px_counts)

# normalize by total possible hits
possible_hits = np.loadtxt(f"{results_dir}hits.txt")
possible_hits = possible_hits / possible_hits[0]
possible_hits = possible_hits[theta_start:theta_end]

# signals_geo = signals_geo / possible_hits
raw_hits_len = 2 * np.array(raw_hits_len) / np.array(possible_hits)

plt.scatter(central_angles_geo, raw_hits_len)
plt.ylabel("pre-deconvolution total hits")
plt.xlabel("pitch angle")

plt.savefig(f"{results_dir}raw_counts.png")

source_sr = np.loadtxt(f"{results_dir}source-sr.txt")
source_area = np.loadtxt(f"{results_dir}source-area.txt")

source_sr = source_sr[theta_start:theta_end]
source_area = source_area[theta_start:theta_end]

# input flux in # particles / (sr * cm^2)
# should be on a unit sphere so sr ^2
# input_flux = possible_hits * n_particles / (source_sr**2)
fov_sr = get_total_sr()
input_flux = np.ones(len(possible_hits)) * n_particles / fov_sr**2

# huh
# signals_geo = signals_geo * 4300
# remove
# signals_geo = signals_geo / input_flux

# this seemed like it might almost work!
signals_geo = raw_hits_len / signals_geo

# bump it for aperture collimation

# save results
np.savetxt(f"{results_dir}uncertainties-geo.txt", np.array(uncertanties_geo))
np.savetxt(f"{results_dir}uncertainties-sim.txt", np.array(uncertanties_sim))
np.savetxt(f"{results_dir}central-angles-geo.txt", np.array(central_angles_geo))
np.savetxt(f"{results_dir}central-angles-sim.txt", np.array(central_angles_sim))

# and save all the indices nicely if we need them
with open(f"{results_dir}inds.txt", "w") as file:
    for sublist in all_indices_geo:
        for x, y in sublist:
            # Convert the tuple to a formatted string (e.g., "1,2")
            line = f"{x},{y}\n"
            file.write(line)
        # Add a newline character after each sublist
        file.write("\n")

# -----------plotting-------------
y_label = "counts / (sr cm^2)"
# Create the scatter plot
plt.clf()
plt.scatter(central_angles_geo, signals_geo, label="Data Points", marker="o", s=8)

# Plot x-axis uncertainty bars
for x, x_err, y in zip(central_angles_geo, uncertanties_geo, signals_geo):
    plt.plot([x - x_err, x + x_err], [y, y], color="blue")
plt.xlabel("pitch angle")
plt.ylabel(y_label)

plt.savefig(f"{results_dir}pitch-angle-distribution-geo.png")
plt.clf()

# Create the scatter plot
plt.scatter(central_angles_sim, signals_geo, label="Data Points", marker="o", s=8)

# Plot x-axis uncertainty bars
for x, x_err, y in zip(central_angles_sim, uncertanties_geo, signals_geo):
    plt.plot([x - x_err, x + x_err], [y, y], color="blue")
plt.xlabel("pitch angle")
plt.ylabel(y_label)

plt.savefig(f"{results_dir}pitch-angle-distribution-sim.png")
plt.clf()

# Create the scatter plot
plt.scatter(
    central_angles_sim,
    input_flux,
    label="Data Points",
    marker="o",
    s=8,
)
plt.xlabel("Pitch Angle")
plt.ylabel("Input Flux counts / [sr cm^2]")
plt.savefig(f"{results_dir}input-flux.png")
plt.clf()



source_angle = np.loadtxt(f"{results_dir}source_angle.txt")
source_spread = np.loadtxt(f"{results_dir}source_spread.txt")
source_radius = 6.0333333  # cm

source_sr = []
source_area = []
for mean_angle, angle_std in zip(source_angle, source_spread):
    # calculate solid angle of a ring viewed from center
    sr_source = (
        2
        * np.pi
        * (
            np.cos(np.deg2rad(mean_angle) - (np.deg2rad(angle_std) / 2))
            - np.cos(np.deg2rad(mean_angle) + (np.deg2rad(angle_std) / 2))
        )
    )
    area_source = (
        2
        * np.pi
        * source_radius**2
        * (
            np.cos(np.deg2rad(mean_angle) - (np.deg2rad(angle_std) / 2))
            - np.cos(np.deg2rad(mean_angle) + (np.deg2rad(angle_std) / 2))
        )
    )

    source_sr.append(sr_source)
    source_area.append(area_source)
"""

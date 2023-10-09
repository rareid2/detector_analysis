# go through angular space and see which pixel is lit up and how many surrpiunding (based on some threshold)
# calculate the spread based on the pixel spread

# trace each surrounding pixel to a pinhole to find angular spread of it
# that IS the FWHM
# find the pixel centers for each og the angles run
# then

# okay first load each image
# trace each pixel that is over a threshold
# what should threshold be? 4x of the noise floor?

import sys

sys.path.insert(1, "../detector_analysis")
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

import matplotlib.pyplot as plt

import numpy as np
import os
import copy

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

# general detector design
det_size_cm = 3.05  # cm
pixel = 0.25  # mm

start = 0
end = 47
step = 1.43 / 2

# Create the list using a list comprehension
thetas = [start + i * step for i in range(int((end - start) / step) + 1)]


# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 61
multiplier = 2

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

# flat field array
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400-d3-2p25/"
flat_field = np.loadtxt(f"{txt_folder}interp_grid.txt")

# before i run this, get point source strength
# re run the plotting script

signals = []
uncertanties = []
central_angles = []
all_indices = []
# ------------------- simulation parameters ------------------
n_particles = 1e8
ii = 0

for theta in thetas[:29]:
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
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-d3-2p25"
    # fname = f"../simulation-data/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"
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
    )

    # ---------- process results -----------

    myhits = Hits(fname=fname, experiment=False, txt_file=True)
    """
    # myhits.get_det_hits(
    #    remove_secondaries=True, second_axis="y", energy_level=energy_level
    # )
    # myhits.exclude_pcfov(det_size_cm, mask_size / 10, distance, 3, "y")

    # check if not first iteration
    if ii != 0:
        # update fields in hits dict
        # myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
        # myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
        # myhits.hits_dict["Vertices"].extend(hits_copy.hits_dict["Vertices"])

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

    signal = deconvolver.deconvolved_image

    noise_floor = np.average(deconvolver.deconvolved_image[:40, :40])

    # calculate angle of each pixel lol
    pixel_size = pixel * 0.1  # cm
    pixel_count = n_elements_original * multiplier

    # Initialize an empty list to store the coordinates
    angles = []
    center_pixel = 61

    signal_sum = 0

    # max signal
    max_value = np.max(signal)
    max_indices = np.argwhere(signal == max_value)
    max_indices = max_indices[0]

    indices = []

    for x in range(pixel_count):
        for y in range(pixel_count):
            # Calculate the relative position from the center
            relative_x = (x - center_pixel) * pixel_size
            relative_y = (y - center_pixel) * pixel_size

            aa = np.sqrt(relative_x**2 + relative_y**2)

            angle = np.arctan(aa / distance)

            if signal[y, x] > noise_floor * 2:
                angles.append(np.rad2deg(angle))
                signal_sum += signal[y, x]
                indices.append((y, x))
                if y == max_indices[0] and x == max_indices[1]:
                    central_angles.append(np.rad2deg(angle))
                    print(np.rad2deg(angle))
                    print(max_indices)

    uncertanties.append((min(angles) - max(angles)) / 2)
    signals.append(signal_sum / 89357784.00000003)
    all_indices.append(indices)

    ii += 1

# Create the scatter plot
plt.scatter(central_angles, signals, label="Data Points", marker="o", s=8)

# Plot x-axis uncertainty bars
for x, x_err, y in zip(central_angles, uncertanties, signals):
    plt.plot([x - x_err, x + x_err], [y, y], color="blue")

plt.savefig("../simulation-results/rings/pitch-angle-distribution.png")

np.savetxt("../simulation-results/rings/uncertainties.txt", np.array(uncertanties))
np.savetxt("../simulation-results/rings/central-angles.txt", np.array(central_angles))

with open("../simulation-results/rings/inds.txt", "w") as file:
    for sublist in all_indices:
        for x, y in sublist:
            # Convert the tuple to a formatted string (e.g., "1,2")
            line = f"{x},{y}\n"
            file.write(line)
        # Add a newline character after each sublist
        file.write("\n")


# TODO: should I use the central angle or the simulation angle
# for now, use central angle

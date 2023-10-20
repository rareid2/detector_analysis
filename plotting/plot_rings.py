import sys

sys.path.insert(1, "../detector_analysis")
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

import numpy as np
import os
import copy
import matplotlib.pyplot as plt

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=False)

# general detector design
det_size_cm = 3.05  # cm
pixel = 0.5  # mm

start = 0
end = 47
step = 1.43 / 2

# Create the list using a list comprehension
thetas = [start + i * step for i in range(int((end - start) / step) + 1)]


# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 61
multiplier = 1

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
txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400/"
flat_field = np.loadtxt(f"{txt_folder}interp_grid.txt")

# ------------------- simulation parameters ------------------
n_particles = 1e8
ii = 0
for theta in thetas:
    print(theta)
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
    fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-d3-5p3"
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

    # weight by cosine theta
    # myhits.txt_hits = myhits.txt_hits * np.cos(np.deg2rad(theta))

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

# deconvolution steps
deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_dir = "../simulation-results/rings/"
results_tag = f"combined-{n_particles:.2E}_{energy_type}_{energy_level}"
results_save = results_dir + results_tag

deconvolver.deconvolve(
    downsample=int(multiplier * n_elements_original),
    trim=trim,
    vmax=None,
    plot_deconvolved_heatmap=True,
    plot_raw_heatmap=True,
    save_raw_heatmap=results_save + "_raw.png",
    save_deconvolve_heatmap=results_save + "_dc.png",
    plot_signal_peak=False,
    plot_conditions=False,
    flat_field_array=None,
    hits_txt=True,
)

"""
mx, my = deconvolver.reverse_raytrace()
deconvolver.export_to_matlab(np.column_stack((mx, my)))
signal = deconvolver.deconvolved_image

# read in the correct locations of everything
indices = []
with open("../simulation-results/rings/inds.txt", "r") as file:
    sublist = []  # Initialize an empty sublist
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:  # Check if the line is not empty
            x, y = map(int, line.split(","))  # Split the line and convert to integers
            sublist.append((x, y))  # Add the tuple to the current sublist
        else:
            if sublist:
                indices.append(sublist)  # Add the sublist to the main list if not empty
            sublist = []  # Reset the sublist

# Add the last sublist if it exists
if sublist:
    indices.append(sublist)

# now, we go through each indices in the deconvolved image

# read in the angles as well
angles = np.loadtxt("../simulation-results/rings/central-angles.txt")
uncertanties = np.loadtxt("../simulation-results/rings/uncertainties.txt")

signals = []
for px_inds in indices:
    signal_sum = 0
    for x, y in px_inds:
        signal_sum += signal[x, y]
    signals.append(signal_sum / len(px_inds))


# Create the scatter plot
# plt.scatter(angles, signals, label="Data Points", marker="o", s=8)

# Plot x-axis uncertainty bars
# for x, x_err, y in zip(angles, uncertanties, signals):
#    plt.plot([x - x_err, x + x_err], [y, y], color="blue")
# plt.ylim([0, 3e6])
# plt.savefig("../simulation-results/rings/pitch-angle-distribution-combined-section.png")
"""

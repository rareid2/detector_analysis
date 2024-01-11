from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from plotting.plot_settings import *

from macros import find_disp_pos
import numpy as np
import os, copy
import matplotlib.pyplot as plt

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# flat field array
# txt_folder = "/home/rileyannereid/workspace/geant4/simulation-results/aperture-collimation/61-2-400/"
# flat_field = np.loadtxt(f"{txt_folder}interp_grid.txt")
flat_field = None

# general detector design
det_size_cm = 2.19  # cm
pixel = 0.1  # mm

# ---------- coded aperture set up ---------

# set number of elements
n_elements_original = 73
multiplier = 3

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = round(element_size * n_elements, 2)
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask
thickness = 400  # um

# focal length
distance = 1  # cm

n_particles = 1e7

radius_cm = 7

txt = True

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
    radius_cm=radius_cm,
)

# --------------set up source---------------
energy_type = "Mono"
energy_level = 100  # keV

total_raw_hits = 0

fname_tag = f"{n_elements_original}-{distance}-cos-plane-rotate-0"
if txt:
    fname = f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
else:
    fname = f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_level, None],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=None,
    radius_cm=radius_cm,
    ring=True,
)

# --------------RUN---------------
# simulation_engine.run_simulation(fname, build=False, rename=True)

# ---------- process results -----------
results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/"

# load them for processing
myhits = Hits(fname=fname, experiment=False, txt_file=txt)

if not txt:
    myhits.get_det_hits(
        remove_secondaries=True, second_axis="y", energy_level=energy_level
    )
    print(len(myhits.hits_dict["Position"]))
    # myhits.exclude_pcfov(det_size_cm, mask_size * 0.1, distance, 2.0303, "y",radius_cm)
    # print(len(myhits.hits_dict["Position"]))

if txt:
    total_raw_hits += np.sum(np.loadtxt(fname))
    print(np.sum(np.loadtxt(fname)))

deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
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
    flat_field_array=flat_field,
    hits_txt=txt,
    rotate=False,
    delta_decoding=True,
)

signal = deconvolver.deconvolved_image
print(np.sum(signal))


fname_tag = f"{n_elements_original}-{distance}-cos-plane-0"
if txt:
    fname = f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
else:
    fname = f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

myhits = Hits(fname=fname, experiment=False, txt_file=txt)


deconvolver = Deconvolution(myhits, simulation_engine)

# directory to save results in
results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
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
    flat_field_array=flat_field,
    hits_txt=txt,
    rotate=True,
    delta_decoding=True,
)

if txt:
    total_raw_hits += np.sum(np.loadtxt(fname))
    print(np.sum(np.loadtxt(fname)))

signal_r = deconvolver.deconvolved_image
print(np.sum(signal_r))

signal_combined = signal_r + signal
plt.imshow(signal_combined, cmap=cmap)
plt.colorbar()
plt.savefig(f"{results_dir}signal_combined.png")

"""
for i in range(0):
fname_tag = f"{n_elements_original}-{distance}-cos-sphere-fovlimited-rotate-{i}"
if txt:
    fname = f"../simulation-results/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
else:
    fname = f"../simulation-data/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

simulation_engine.set_macro(
    n_particles=int(n_particles),
    energy_keV=[energy_type, energy_level, None],
    surface=True,
    progress_mod=int(n_particles / 10),  # set with 10 steps
    fname_tag=fname_tag,
    confine=False,
    detector_dim=det_size_cm,
    theta=None,
    radius_cm=radius_cm,
    ring=True,
)

# --------------RUN---------------
# simulation_engine.run_simulation(fname, build=False, rename=True)

    # ---------- process results -----------
    results_dir = "/home/rileyannereid/workspace/geant4/simulation-results/"

    # load them for processing
    myhits = Hits(fname=fname, experiment=False, txt_file=txt)

    if not txt:
        myhits.get_det_hits(
            remove_secondaries=True, second_axis="y", energy_level=energy_level
        )
        # print(len(myhits.hits_dict["Position"]))
        # myhits.exclude_pcfov(
        #    det_size_cm, mask_size * 0.1, distance, 2.0303, "y", radius_cm
        # )

    if txt:
        total_raw_hits += np.sum(np.loadtxt(fname))
        print(np.sum(np.loadtxt(fname)))
        # check if not first iteration
        if i != 0:
            myhits.txt_hits += hits_copy.txt_hits
            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)
    # print(np.sum(np.loadtxt(fname)))
    print(fname_tag)

    # myhits.exclude_pcfov(det_size_cm, mask_size * 0.1, distance, 2.0303, "y",radius_cm)

    deconvolver = Deconvolution(myhits, simulation_engine)

    # directory to save results in
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
        flat_field_array=flat_field,
        hits_txt=txt,
        rotate=True,
        delta_decoding=False,
    )

    signal_rotate = deconvolver.deconvolved_image
    print(np.sum(signal_rotate))

# now combined them
signal_combined = signal_rotate + signal
plt.imshow(signal_combined, cmap=cmap)
plt.colorbar()
plt.savefig(f"{results_dir}combined_flatfield.png", dpi=500, transparent=True)
plt.clf()

# shit up
# signal_combined += np.abs(np.min(signal_combined))

# now get weighted intensities
# signal_combined_norm = total_raw_hits * signal_combined / np.sum(signal_combined)

# signal_combined_norm = signal_combined

flux = 4e9 / (4 * np.pi**2 * radius_cm**2)
geom_factor = signal_combined / flux
plt.clf()
plt.imshow(geom_factor, cmap=cmap)
plt.colorbar(label="cm^2 * sr")
plt.savefig(f"{results_dir}geom_factor.png", dpi=500, transparent=True)

# now we have geometric factor
# NOW we add it all up and multiply by geometric factor

result = []

with open(f"{results_dir}rings/inds.txt", "r") as file:
    subarray = []
    for line in file:
        line = line.strip()
        if line:
            # Non-empty line, split and convert to integers
            pair = [int(x) for x in line.split(",")]
            subarray.append(pair)
        else:
            # Empty line, start a new subarray
            if subarray:
                result.append(subarray)
                subarray = []

# Add the last subarray if the file doesn't end with an empty line
if subarray:
    result.append(subarray)

srss = []
for sb in result:
    srs = 0
    count = 0
    for ss in sb:
        count += 1
        srs += signal_combined[ss[0], ss[1]] / geom_factor[ss[0], ss[1]]
    # print(count, srs)
    srss.append(srs)
plt.clf()

# plot a sine theta on top

thetas = np.loadtxt(f"{results_dir}rings/thetas.txt")
unc = np.rad2deg(np.loadtxt(f"{results_dir}rings/uncertainties.txt"))
fig, ax1 = plt.subplots()
plt.errorbar(thetas, np.array(srss), yerr=None, xerr=unc, color="pink", fmt="o")
plt.plot(thetas, 1.9e9 * np.sin(np.deg2rad(np.array(thetas))))
ax1.grid(True, linestyle="--", linewidth=0.5)
plt.savefig(f"{results_dir}final.png", dpi=500, transparent=True)
# okay now loop through

# theory - we have an issue that the larger rings are aliasing back in and potentiatlly causing
# problems....
# can i remove them? does that make sense?
# i think there is a lot of them and I need to remove them in post..... :(
"""

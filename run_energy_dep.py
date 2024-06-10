from simulation_engine import SimulationEngine
from mlem import fft_conv
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
import sys
from run_flatfield import run_geom_corr
import scipy
import cmocean
from matplotlib.colors import LogNorm


def read_float_lists_from_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            float_strings = line.strip().replace(",", "").split()
            float_list = [float(x) for x in float_strings]
            data.append(float_list)
    return data


def get_plane_size(distance, det_size_cm, mask_plane_distance_mm, theta_fcfov_deg):
    mask_detector_distance_mm = distance * 10
    plane_distance_to_detector_mm = mask_detector_distance_mm + mask_plane_distance_mm

    detector_diagonal_mm = det_size_cm * 10 * np.sqrt(2)
    detector_half_diagonal_mm = detector_diagonal_mm / 2

    additional_diagonal_mm = (
        np.tan(np.deg2rad(theta_fcfov_deg)) * plane_distance_to_detector_mm
    )

    plane_diagonal_mm = (additional_diagonal_mm + detector_half_diagonal_mm) * 2

    plane_side_length_mm = plane_diagonal_mm / np.sqrt(2)

    # geant asks for half side length

    plane_half_side_length_mm = plane_side_length_mm / 2
    return plane_half_side_length_mm


def get_hits(fname, dthick, energy_level):
    txt = False
    nthreads = 14
    for hi in range(nthreads):
        print(hi)
        fname_hits = fname[:-4] + "-{}.csv".format(hi)
        myhits = Hits(fname=fname_hits, experiment=False, txt_file=txt)
        hits_dict, sec_brehm, sec_e = myhits.get_det_hits(
            remove_secondaries=True,
            det_thick_cm=dthick,
        )
        if hi != 0:
            # update fields in hits dict
            myhits.hits_dict["Position"].extend(hits_copy.hits_dict["Position"])
            myhits.hits_dict["Energy"].extend(hits_copy.hits_dict["Energy"])
            myhits.hits_dict["Edep"].extend(hits_copy.hits_dict["Edep"])

            hits_copy = copy.copy(myhits)
        else:
            hits_copy = copy.copy(myhits)
    return myhits


simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)
det_size_cm = 5
thickness_um = 100  # um

energy_type = "Mono"

rank = 23
distance = 3.9

n_particles = 1e6

mosaic = True

detector_diagonal_mm = det_size_cm * 10 * np.sqrt(2)
n_elements = (2 * rank) - 1
pixel_size_mm = round(10 * det_size_cm / rank, 4)  # mm
mask_size_mm = round(pixel_size_mm * (2 * rank - 1), 2)
mask_diagonal_mm = mask_size_mm * np.sqrt(2)

detector_half_diagonal_mm = detector_diagonal_mm / 2
mask_half_diagonal_mm = mask_diagonal_mm / 2

# FCFOV half angle
theta_fcfov_deg = np.rad2deg(
    np.arctan((mask_half_diagonal_mm - detector_half_diagonal_mm) / (distance * 10))
)

fov = np.rad2deg(
    np.arctan(
        (detector_diagonal_mm + (mask_half_diagonal_mm - detector_half_diagonal_mm))
        / (distance * 10)
    )
)
pcfov = fov - theta_fcfov_deg

# get size of plane
mask_plane_distance_mm = 5
plane_size_mm = get_plane_size(
    distance, det_size_cm, mask_plane_distance_mm, theta_fcfov_deg
)
plane_size_cm = plane_size_mm / 10
plane_location_cm = (
    (1111 * 0.45) - distance - (thickness_um * 1e-4 / 2) - (mask_plane_distance_mm / 10)
)

theta_lower = 0
theta = 0

det_thicknesses = np.arange(50, 1000, 75)  # change
energies = np.logspace(2.4771, 4.17609, 15)

# energies = [1604]
# det_thicknesses = [125]

# energy_level = 0.235  # keV
# det_thick = 100  # um
"""
all_deps = []
for energy_level in energies:
    energy_level = round(energy_level)
    avg_dep = []
    for det_thick in det_thicknesses:
        det_thick = round(det_thick)

        simulation_engine.set_config(
            det1_thickness_um=det_thick,
            det_gap_mm=30,  # gap between first and second (unused detector)
            win_thickness_um=100,  # window is not actually in there
            det_size_cm=det_size_cm,
            n_elements=n_elements,
            mask_thickness_um=thickness_um,
            mask_gap_cm=distance,
            element_size_mm=pixel_size_mm,
            mosaic=mosaic,
            mask_size=mask_size_mm,
            radius_cm=1,
        )

        formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
        fname_tag = f"{energy_level}-{round(det_thick)}"

        fname = f"../simulation-data/e-dep/{fname_tag}.csv"
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
            plane_size_cm=plane_size_cm,
            ring=True,
            sphere_center=plane_location_cm,
            # radius_cm=3,
        )

        # simulation_engine.run_simulation(fname, build=False, rename=True)
        results_dir = "../simulation-results/e-dep/"
        results_tag = f"{fname_tag}"
        results_save = results_dir + results_tag

        edeps = []
        myhits = get_hits(fname, det_thick * 1e-4, energy_level)
        for ei, e_init in enumerate(myhits.hits_dict["E0"]):
            if e_init == energy_level:
                # get Edep
                edeps.append(myhits.hits_dict["Edep"][ei])

        print(
            "average energy deposition",
            np.average(np.array(edeps)),
            energy_level,
            det_thick,
        )
        avg_dep.append(np.average(np.array(edeps)))
    all_deps.append(avg_dep)
"""
results_dir = "../simulation-results/e-dep/"
all_deps = read_float_lists_from_file("edep_results.txt")

print(all_deps)
fig, ax = plt.subplots()
cmap = cmocean.cm.thermal
for ii, edep_energy in enumerate(all_deps):
    ax.plot(det_thicknesses, edep_energy, color=cmap(ii / len(all_deps)))

cbar = plt.colorbar(
    plt.cm.ScalarMappable(
        cmap=cmap, norm=LogNorm(vmin=min(energies), vmax=max(energies))
    ),
    ax=plt.gca(),
    label="Initial energy [keV]",
    pad=0.01,
)
ax.set_xlabel("Silicon detector thickness [um]")
ax.set_ylabel("Average energy deposited [keV]")
ax.set_ylim([0, np.amax(np.array(all_deps))])
ax.set_xlim([50, 925])

left, bottom, width, height = [
    0.22,
    0.65,
    0.26,
    0.2,
]  # Adjust these values for your inset position
ax_inset = plt.gcf().add_axes([left, bottom, width, height])
for ii, edep_energy in enumerate(all_deps):
    ax_inset.plot(det_thicknesses[:2], edep_energy[:2], color=cmap(ii / len(all_deps)))

ax_inset.set_xlabel("Si thickness [um]", fontsize=8)
ax_inset.set_ylabel("Avg E dep [keV]", fontsize=8)
ax_inset.set_ylim([0, 215])
ax_inset.set_xlim([50, 100])

plt.savefig(
    results_dir + "edep.png",
    dpi=500,
    transparent=False,
    bbox_inches="tight",
    pad_inches=0.02,
)

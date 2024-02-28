# evaluate the impact of shielding

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
from run_part_csv import read_csv_in_parts

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

txt = False
simulate = True

# general detector design
det_size_cm = 4.956  # cm
pixel = 0.28  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 59
multiplier = 3
pixel_size = multiplier*pixel_size

element_size = pixel * multiplier
n_elements = (2 * n_elements_original) - 1

mask_size = element_size * n_elements
# no trim needed for custom design
trim = None
mosaic = True

# thickness of mask

thicknesses = np.linspace(100,4000, 25)
energies = np.linspace(10**2.4, 10**4.05, 25) # keV

# focal length
distance = 3.47  # cm

# ------------------- simulation parameters ------------------
for energy in energies:
    primaries_len = []
    secondary_e_len = []
    secondary_gamma_len = []

    for thickness in thicknesses:
        print(f"RUNNING {energy} AND {thickness}")
        n_particles = 1e6
        world_offset = 1111 * 0.45
        det_thickness = 300
        
        detector_loc = world_offset - ((det_thickness / 2) * 1e-4)
        ca_pos = world_offset - distance - (((thickness / 2) + (det_thickness / 2)) * 1e-4)
        src_plane_distance = world_offset + 500

        px = 0
        py = 0
        pz = detector_loc

        # source plane
        src_plane_normal = np.array([0, 0, 1])  # Normal vector of the plane
        src_plane_pt = np.array([0, 0, -500])  # A point on the plane

        px_point = np.array([px, py, pz])  # A point on the line
        px_ray = np.array([0 - px, 0 - py, ca_pos - pz])

        ndotu = src_plane_normal.dot(px_ray)

        epsilon = 1e-8
        if abs(ndotu) < epsilon:
            print("no intersection or line is within plane")

        w = px_point - src_plane_pt
        si = -src_plane_normal.dot(w) / ndotu
        src_point = w + si * px_ray + src_plane_pt


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
        energy_level = energy  # keV

        # --------------set up data naming---------------
        fname_tag = f"{n_elements_original}-{distance}-{int(thickness)}-deg"

        fname = f"../simulation-data/brehm/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

        if txt:
            fname = f"../simulation-results/brehm/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
        
        simulation_engine.set_macro(
            n_particles=int(n_particles),
            energy_keV=[energy_level],
            surface=False,
            positions=[[src_point[0], src_point[1], -500]],
            directions=[1],
            progress_mod=int(n_particles / 10),  # set with 10 steps
            fname_tag=fname_tag,
            detector_dim=det_size_cm,
        )

        # --------------RUN---------------
        if simulate:
            simulation_engine.run_simulation(fname, build=False, rename=True)

        # ---------- process results -----------
        # directory to save results in
        results_dir = "../simulation-results/brehm/"
        results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
        results_save = results_dir + results_tag

        myhits = Hits(fname=fname, experiment=False, txt_file=txt)
        if not txt:
            _, secondary_e, secondary_gamma = myhits.get_det_hits(
                remove_secondaries=False, second_axis="y", energy_level=energy_level
            )
            # need to make a function that returns secondary electrons and secondary brehm
            print(f"SECONDARY ELECTRONS COUNT = {secondary_e} and SECONDARY GAMMA COUNT = {secondary_gamma}")
            secondary_e_len.append(secondary_e)
            secondary_gamma_len.append(secondary_gamma)
            primaries_len.append(len(myhits.hits_dict["Energy"]))

        # deconvolution steps
        deconvolver = Deconvolution(myhits, simulation_engine)

        deconvolver.deconvolve(
            downsample=int(n_elements_original),
            trim=trim,
            vmax=None,
            plot_deconvolved_heatmap=False,
            plot_raw_heatmap=False,
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

    # okay now I should save it
    results_txt = results_dir + f"{int(energy_level)}"
    np.savetxt(results_txt+"_primaries.txt", np.array(primaries_len))
    np.savetxt(results_txt+"_sec_e.txt", np.array(secondary_e_len))
    np.savetxt(results_txt+"_sec_gamma.txt", np.array(secondary_gamma_len))

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

txt = False
simulate = False

# general detector design
det_size_cm = 4.956  # cm
pixel = 0.28  # mm
pixel_size = pixel * 0.1

# ---------- coded aperture set up ---------
# set number of elements
n_elements_original = 59
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
distance = 3.47  # cm

thetas = [2, 13, 24, 35, 46]

n_p = [5e6, 1.5e7, 5e7]

n_particles_frac = np.logspace(0, -2, 8)
n_particles_frac = n_particles_frac[1:]

energies = np.logspace(1, 4.5, 20)
# for pinhole
# n_elements_original = 1
# n_elements = 1
# mosaic = False
# mask_size = 21.9

transparencies = []
pixel_size = multiplier*pixel_size

# ------------------- simulation parameters ------------------
for ii, theta in enumerate(thetas):
    print(theta)
    # simulate 3e8 per cm^2 per sr per s
    # 3e8 for full circle * sr * cm
    # 1e7 for ring * cm
    # 5e8 for sphere
    #n_particles = int((8e7 * (5.030 * 2) ** 2) * (1 - np.cos(np.deg2rad(theta)))) * npf
    for n in [n_p[1]]:
        n_particles = int((n * (5.265 * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))
        large_file = False
        #txt=False
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
        fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg"

        fname = f"../simulation-data/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_{formatted_theta}.csv"

        if txt:
            fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
            #raw = np.loadtxt(fname)
            #fname_tag_r = f"{n_elements_original}-{distance}-{formatted_theta}-deg-rot"
            #fname_r = f"../simulation-results/rings/{fname_tag_r}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"
            #rawr = np.loadtxt(fname_r)
            #rawr = resample(rawr)
            #fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-combo"
            #fname = f"../simulation-results/rings/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_raw.txt"

            #np.savetxt(fname,raw + rawr)

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

        if large_file and not txt:
            raw_hits = read_csv_in_parts(
                fname,
                fname_tag,
                simulation_engine,
                n_elements_original,
                energy_level,
                multiplier,
            )
            np.savetxt(results_save + "_raw.txt", raw_hits)
            txt = True
            fname = results_save + "_raw.txt"

        myhits = Hits(fname=fname, experiment=False, txt_file=txt)
        if not txt:
            _, sec_brehm, sec_e = myhits.get_det_hits(
                remove_secondaries=False, second_axis="y", energy_level=energy_level
            )
            print(sec_brehm, sec_e, len(myhits.hits_dict["Position"]))

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
            resample_array=True,
        )
        np.savetxt(results_save + "_dc.txt", deconvolver.deconvolved_image)
        print(np.sum(deconvolver.raw_heatmap))
        print(np.sum(deconvolver.deconvolved_image))
        #print('secondary', sec_brehm, sec_e)
        """
        #transparencies.append((np.sum(deconvolver.deconvolved_image) / 5e5) - 1)


        # dc = np.loadtxt(results_save + "_combined.txt")
        # plt.imshow(dc, cmap=cmap)
        # plt.savefig(results_save + "_combined.png", dpi=300)


        # loop through pixels
        #if theta > 40:
        #    signal = np.loadtxt(results_save + "_combined.txt")
        #else:
        signal = deconvolver.deconvolved_image
        pixel_count = int(n_elements_original)
        max_value = np.max(signal)
        signal_count = 0
        total_count = 0
        center_pixel = int(59 / 2)
        geometric_factor = 18

        for x in range(pixel_count):
            for y in range(pixel_count):
                relative_x = (x - center_pixel) * pixel_size
                relative_y = (y - center_pixel) * pixel_size

                aa = np.sqrt(relative_x**2 + relative_y**2)

                # find the geometrical theta angle of the pixel
                angle = np.arctan(aa / distance)

                # signal[y,x] > max_value / 4 and

                if np.rad2deg(angle) < (theta + 0.5):
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
        print("recorded flux", total_count / (geometric_factor * px_factor))
    #print(transparencies)
    """

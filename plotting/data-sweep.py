import numpy as np
import shutil
import os

# find all the data files used in the paper and combine into a folder
files = []
# FWHM simulations
fwhm_results = "/home/rileyannereid/workspace/geant4/simulation-results/fwhm-figure/"
thin_raw_center = fwhm_results + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_600_raw.txt"
thick_raw_edge = fwhm_results + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_6000_raw.txt"
thin_raw_edge = fwhm_results + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_600_raw.txt"
thin_dc_center = fwhm_results + "47-2-300/47-2.2-c-xy-0_1.00E+06_Mono_600_dc.txt"
thick_dc_edge = fwhm_results + "47-2-15/47-2.2-e-xy-0_1.00E+06_Mono_6000_dc.txt"
thin_dc_edge = fwhm_results + "47-2-300/47-2.2-e-xy-0_1.00E+06_Mono_600_dc.txt"

files.append(thin_raw_center)
files.append(thick_raw_edge)
files.append(thin_raw_edge)
files.append(thin_dc_center)
files.append(thick_dc_edge)
files.append(thin_dc_edge)

# extended sources - rings
files.append("../simulation-results/rings/89-4.49-ring-pt_1.00E+08_Mono_100_dc.txt")

# extended sources - circles
n_p = [5e6, 5e7, 5e8]
thetas = [2, 13, 24, 35, 46]
for theta in thetas:
    for n in n_p:
        formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
        n_particles = int((n * (5.265 * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))
        files.append(
            f"/home/rileyannereid/workspace/geant4/simulation-results/rings/59-3.47-{formatted_theta}-deg_{n_particles:.2E}_Mono_500_dc.txt"
        )

# second order
n_elements_original = 73
distance = 2
theta = 33.025
area = 2.445
n_particles = int((3e8 * (area * 2) ** 2) * (1 - np.cos(np.deg2rad(theta))))
energy_type = "Mono"
energy_level = 100  # keV
formatted_theta = "{:.0f}p{:02d}".format(int(theta), int((theta % 1) * 100))
fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle"
files.append(
    f"../simulation-results/rings/final_image/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.txt"
)
fname_tag = f"{n_elements_original}-{distance}-{formatted_theta}-deg-circle-rotate-0"
files.append(
    f"../simulation-results/rings/final_image/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}_dc.txt"
)
# PCFOV
n_particles_frac = np.logspace(0, -2, 8)
npf = n_particles_frac[5]
print(npf / n_particles_frac[1])
n_particles = (
    int((8e7 * (5.030 * 2) ** 2) * (np.cos(np.deg2rad(45)) - np.cos(np.deg2rad(70))))
    * npf
)
files.append(
    f"../simulation-results/rings/59-3.47-45p00-deg-rotate_{n_particles:.2E}_Mono_100_dc.txt"
)
files.append(
    f"../simulation-results/rings/59-3.47-45p00-deg_{n_particles:.2E}_Mono_100_dc.txt"
)
files.append("../simulation-results/rings/59-3.47-22p00-deg_5.90E+08_Mono_100_dc.txt")
files.append("../simulation-results/rings/59-3.47-35p00-deg_1.00E+09_Mono_100_dc.txt")
files.append(
    "../simulation-results/rings/59-3.47-22-45-deg-fov_1.53E+09_Mono_100_combined.txt"
)

# secondary particles
rank_array = [11, 31]
thicknesses = [500, 1250, 2500]
det_size_cm = 1.408
distances = np.linspace(0.1 * 2 * det_size_cm, 1 * 2 * det_size_cm, 5)
energy_levels = np.logspace(2, 4, 10)
for ri, rank in enumerate(rank_array):
    for thickness in thicknesses:
        for di, start_distance in enumerate(distances[2:]):
            for energy_level in energy_levels:
                filename = f"{rank}-{thickness}-{di}_1.00E+07_Mono_{round(energy_level)}-hits.txt"
                files.append(
                    f"/home/rileyannereid/workspace/geant4/simulation-results/secondaries/{filename}"
                )


for file_name in files:
    source_path = file_name

    # Copy the file
    shutil.copy2(source_path, "../simulation-results/paper-simulation-data/")
    print(file_name)

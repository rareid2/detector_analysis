from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np

# construct = CA and TD
# source = DS and PS

# RUN THIS WITH OLD GEANT4 INSTALL (NOT ONE WITH EXPERIMENT)

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# timepix design
det_size_cm = 1.408

# loop through rank options (primes)
n_elements_original = [11, 31]  # n elements no mosaic
multipliers = [22, 8]
pixel = 0.055  # mm

element_size_mm_list = [
    pixel * multiplier for multiplier in multipliers
]  # element size in mm

n_elements_list = [
    (ne * 2) - 1 for ne in n_elements_original
]  # total number of elements
mask_size_list = [
    round(es * ne, 2) for (es, ne) in zip(element_size_mm_list, n_elements_list)
]  # mask size in mm
trims = [7, 4]

mask_config_list = [
    [n_elements, rank, (256 - 2 * trim) / multiplier, trim, element_size_mm, mask_size]
    for n_elements, rank, multiplier, trim, element_size_mm, mask_size in zip(
        n_elements_list,
        n_elements_original,
        multipliers,
        trims,
        element_size_mm_list,
        mask_size_list,
    )
]

# thickness of mask
thicknesses = [500, 1250, 2500]  # im um, mask thickness  # im um, mask thickness

# distance between mask and detector
distances = np.linspace(0.1 * 2 * det_size_cm, 1 * 2 * det_size_cm, 5)

base_radius = 3.25

for mask_config in mask_config_list[:1]:
    for thickness in thicknesses:
        for di, start_distance in enumerate(distances):
            distance = round(start_distance - ((150 + (thickness / 2)) * 1e-4), 2)
            n_particles = 1e8
            if distance > 1:
                sphere_radius = round(base_radius + (distance - 1), 2)
            else:
                sphere_radius = base_radius

            simulation_engine.set_config(
                det1_thickness_um=300,
                det_gap_mm=30,
                win_thickness_um=100,
                det_size_cm=det_size_cm,
                n_elements=mask_config[0],
                mask_thickness_um=thickness,
                mask_gap_cm=distance,
                element_size_mm=mask_config[4],
                mosaic=True,
                mask_size=mask_config[5],
                radius_cm=sphere_radius,
            )
            # --------------set up source---------------
            energy_type = "Mono"
            energy_level = 500  # keV

            # --------------set up data naming---------------
            fname_tag = f"flat-field-sweep-{mask_config[1]}-{round(thickness)}-{di}"
            fname = f"../simulation-data/flat-field-sweep/{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}.csv"

            # run a test particle
            simulation_engine.set_macro(
                n_particles=1,
                energy_keV=[energy_type, energy_level, None],
                sphere=True,
                radius_cm=sphere_radius,
                progress_mod=int(n_particles / 10),  # set with 10 steps
                fname_tag=fname_tag,
                dist=None,
                confine=False,
            )
            # --------------RUN---------------
            simulation_engine.run_simulation(fname, build=True, rename=False)

            # run the true one
            simulation_engine.set_macro(
                n_particles=n_particles,
                energy_keV=[energy_type, energy_level, None],
                sphere=True,
                radius_cm=sphere_radius,
                progress_mod=int(n_particles / 10),  # set with 10 steps
                fname_tag=fname_tag,
                dist=None,
                confine=True,
            )
            simulation_engine.run_simulation(fname, build=False, rename=True)

            # ---------- process results -----------
            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits()

            # deconvolution steps
            deconvolver = Deconvolution(myhits, simulation_engine)

            # directory to save results in
            results_dir = "../simulation-results/flat-field/"
            results_tag = f"{fname_tag}_{n_particles:.2E}_{energy_type}_{energy_level}"
            results_save = results_dir + results_tag

            deconvolver.deconvolve(
                downsample=8,
                trim=mask_config[3],
                vmax=None,
                plot_deconvolved_heatmap=True,
                plot_raw_heatmap=True,
                save_raw_heatmap=results_save + "_raw.png",
                save_deconvolve_heatmap=results_save + "_dc.png",
                plot_signal_peak=True,
                plot_conditions=False,
                save_peak=results_save + "_peak.png",
                normalize_signal=True,
                axis=1,
            )

            # get signal over deconvolved image along one axis
            signal = deconvolver.signal
            pixels = [i for i in range(len(signal))]

            # convert pixel space to fov
            fov = np.rad2deg(
                np.arctan((((mask_config[5] / 10) - det_size_cm) / 2) / distance)
            )

            fovs = np.linspace(-1 * fov, fov, len(signal))

            # save the data to a text file
            array = np.column_stack((signal, fovs))
            np.savetxt(results_save + "fov.txt", array, delimiter=" ", fmt="%s")

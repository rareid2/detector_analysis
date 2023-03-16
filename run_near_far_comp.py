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
n_elements_original = [7, 11, 17, 31]  # n elements no mosaic
multipliers = [36, 22, 14, 8]
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
trims = [2, 7, 9, 4]

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
multipliers_fwhm = [22, 8]
# thickness of mask
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness  # im um, mask thickness
thickness = thicknesses[2]
# distance between mask and detector
distances = np.flip(np.linspace(0.1 * 2 * det_size_cm, 10 * 2 * det_size_cm, 50))
# just grab the last 2
distances = [5, distances[-1]]
distances =[5.05]
source_distances = np.linspace(-500,10,50)
source_distances = [-26]
for mi, mask_config in enumerate([mask_config_list[1], mask_config_list[3]]):
    for start_distance in distances:
        sim_count = 0

        # set up correct mask distance
        distance = start_distance - (150 + (thickness / 2)) * 1e-4

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
        )

        for source_distance in source_distances:
            set_source = source_distance + start_distance

            simulation_engine.set_macro(
                n_particles=1000000,
                energy_keV=[500],
                positions=[[0, 0, set_source]],
                directions=[0],
            )

            fname = (
                "../simulation_data/timepix_sim/prospectus_plots/fwhm/%d_%d_%.2f_%.2f.csv"
                % (thickness, mask_config[1], distance, set_source)
            )

            if sim_count == 0:
                simulation_engine.run_simulation(fname, build=True)
            else:
                simulation_engine.run_simulation(fname, build=False)

            myhits = Hits(fname=fname, experiment=False)
            myhits.get_det_hits()
            deconvolver = Deconvolution(myhits, simulation_engine)
            print(set_source, source_distance)
            _, _, _, fwhm = deconvolver.deconvolve(
                plot_deconvolved_heatmap=True,
                save_raw_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fwhm/%d_%d_%.2f_%.2f_raw.png"
                % (thickness, mask_config[1], distance, set_source),
                plot_raw_heatmap=True,
                downsample=22,
                trim=mask_config[3],
                save_deconvolve_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fwhm/%d_%d_%.2f_%.2f_dc.png"
                % (thickness, mask_config[1], distance, set_source),
                save_peak="../simulation_results/parameter_sweeps/prospectus_plots/fwhm/%d_%d_%.2f_%.2f_peak.png"
                % (thickness, mask_config[1], distance, set_source),
                plot_signal_peak=True,
                plot_conditions=False,
                check_resolved=False,
                condition="half_val",
            )

            file1 = open(
                "../simulation_results/parameter_sweeps/prospectus_plots/fwhm/%d_%d_%.2f.txt"
                % (thickness, mask_config[1], distance),
                "a",
            )  # append mode

            file1.write("%.2f %.4f \n" % (set_source - start_distance, fwhm))
            file1.close()
            sim_count += 1

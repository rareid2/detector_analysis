from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

from macros import find_disp_pos
import numpy as np

# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="CA", source="PS", write_files=True)

# timepix design
det_size_cm = 1.408

# loop through rank options (primes)
n_elements_original = [11, 17, 31]  # n elements no mosaic
multipliers = [22, 14, 8]
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
trims = [7, 9, 4]

mask_config_list = [
    [n_elements, rank, multiplier, trim, element_size_mm, mask_size]
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
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness

# distance between mask and detector
distances = np.linspace(0.25, 8, 15)

sim_count = 0
# for each mask thickness
for thickness in thicknesses:
    # loop through mask configurations
    for mask_config in mask_config_list:
        for distance in distances:
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

            # set macro to run one point src in the center and one next to it
            # run through possible resolutions(!)
            res = False
            # get theoretical resolution and center around it
            th_res = 2 * np.arctan((mask_config[4] / 2) / (10 * distance))
            th_res = 2 * np.rad2deg(th_res)
            for res_deg in np.linspace(
                th_res - (th_res / 2), th_res + (th_res / 2), 20
            ):
                xx = find_disp_pos(
                    res_deg, z_disp=simulation_engine.detector_placement + (500)
                )
                simulation_engine.set_macro(
                    n_particles=1000000,
                    energy_keV=[500, 500],
                    positions=[[0, 0, -500], [xx, 0, -500]],
                    directions=[0, 1],
                )

                fname = "../data/timepix_sim/%d_%d_%.2f_%.2f.csv" % (
                    thickness,
                    mask_config[1],
                    distance,
                    res_deg,
                )
                if sim_count == 0:
                    simulation_engine.run_simulation(fname, build=True)
                else:
                    simulation_engine.run_simulation(fname, build=False)

                myhits = Hits(simulation_engine, fname)
                myhits.get_det_hits()
                deconvolver = Deconvolution(myhits, simulation_engine)
                res = deconvolver.deconvolve(
                    plot_deconvolved_heatmap=True,
                    plot_raw_heatmap=False,
                    multiplier=mask_config[2],
                    trim=mask_config[3],
                    save_deconvolve_heatmap="../results/parameter_sweeps/timepix_sim/%d_%d_%.2f_%.2f.png"
                    % (thickness, mask_config[1], distance, res_deg),
                    save_peak="../results/parameter_sweeps/timepix_sim/%d_%d_%.2f_%.2f_peak.png"
                    % (thickness, mask_config[1], distance, res_deg),
                    plot_signal_peak=True,
                    plot_conditions=True,
                    check_resolved=True,
                    condition="half_val",
                )
                sim_count += 1

                if res == True:
                    break
                else:
                    continue

            file1 = open(
                "../results/parameter_sweeps/timepix_sim/%d_%d.txt"
                % (thickness, mask_config[1]),
                "a",
            )  # append mode
            file1.write("%.2f %.2f \n" % (distance, res_deg))
            file1.close()

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

# lets just use two of them rn
mask_config = mask_config_list[3]

# thickness of mask
thicknesses = [100,562,1333]  # im um, mask thickness  # im um, mask thickness

# distance between mask and detector
distance = 1.5
sim_count = 0

# for each mask thickness
# loop through mask configurations
fovs = [0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
for thickness in thicknesses:
        
    for fov_edge in fovs:
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

        # set macro to run one point src across fov

        # get side location
        xx = find_disp_pos(
            fov_edge, z_disp=simulation_engine.detector_placement + (500)
        )

        simulation_engine.set_macro(
            n_particles=1000000,
            energy_keV=[500],
            positions=[[xx, 0, -500]],
            directions=[1],
        )

        fname = "../data/timepix_sim/signal_fov/%d_%d_%.2f.csv" % (
            thickness,
            mask_config[1],
            fov_edge,
        )
        if sim_count == 0:
            simulation_engine.run_simulation(fname, build=True)
        else:
            simulation_engine.run_simulation(fname, build=False)

        myhits = Hits(simulation_engine, fname)
        myhits.get_det_hits()
        deconvolver = Deconvolution(myhits, simulation_engine)
        res,strength = deconvolver.deconvolve(
            plot_deconvolved_heatmap=True,
            plot_raw_heatmap=False,
            trim=mask_config[3],
            downsample=mask_config[2],
            save_deconvolve_heatmap="../results/parameter_sweeps/timepix_sim/signal_fov/%d_%d_%.2f.png"
            % (thickness, mask_config[1], fov_edge),
            save_peak="../results/parameter_sweeps/timepix_sim/signal_fov/%d_%d_%.2f_peak.png"
            % (thickness, mask_config[1], fov_edge),
            plot_signal_peak=True,
            plot_conditions=True,
            check_resolved=True,
            condition="half_val",
        )

        file1 = open(
            "../results/parameter_sweeps/timepix_sim/signal_fov/%d_%d.txt"
            % (thickness, mask_config[1]),
            "a",
        )  # append mode
        file1.write("%.2f %.2f \n" % (fov_edge, strength))
        file1.close()

        sim_count+=1
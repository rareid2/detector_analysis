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
    [n_elements, rank, (256-2*trim)/multiplier, trim, element_size_mm, mask_size]
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
thicknesses = np.logspace(2, 3.5, 5)  # im um, mask thickness  # im um, mask thickness

# distance between mask and detector
distances = np.flip(np.linspace(0.1*2*det_size_cm,10*2*det_size_cm,50))
sim_count = 0

# for each mask thickness
# loop through mask configurations
fovs = [1.34 ,1.37 ,1.39 ,1.42 ,1.45 ,1.48 ,1.51 ,1.54 ,1.58 ,1.62 ,1.65 ,1.70 , 1.74 , 
        1.78 ,1.93 ,1.98 ,2.03 , 2.19 , 2.15 , 2.31 , 2.38 , 2.45 , 2.53 , 2.52 ,2.61 , 
        2.81 ,2.81 , 3.03 ,3.15 ,3.39 , 3.55 , 3.72 , 3.81 , 4.02 , 4.36 , 4.53 , 4.94 , 
        5.30 , 5.92 , 6.32 , 7.02 , 7.77 ,8.80 ,10.20 ,12.02 ,14.68,18.56,25.19,20,25]

for mask_config in [mask_config_list[3]]:
    for thickness in thicknesses:
        for start_distance, fov_test in zip([distances[22]],[fovs[22]]):
            distance = (start_distance - (150 + (thickness/2))*1e-4)
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
            # get theoretical resolution and center around it
            #th_res = 2 * np.arctan((mask_config[4] / 2) / (10 * distance))
            #th_res = 2 * np.rad2deg(th_res)
            #th_fov = np.arctan(
            #    ((mask_config[5] - (pixel * mask_config[2] * mask_config[1])) / 2)
            #    / (10 * distance)
            #)
            #th_fov = np.rad2deg(th_fov)
            #th_res = th_fov
            res_deg = 0.25
            fov_edge = np.linspace(0,2.5,10)
            #res_fov = [2.74, 2.74, 2.76, 4, 4.86, 5.14, 7, 6.5, 8, 9.8]

            for fov in fov_edge:
                res_deg -= 0.001

                #if res_deg > 0.3:
                #    res_deg -= 0.02
                #else:
                #    res_deg = 2.74
                res=False
                while res==False:
                    xx = find_disp_pos(
                        fov, z_disp=simulation_engine.detector_placement + (500)
                    )
                    yy = find_disp_pos(
                        res_deg, z_disp=simulation_engine.detector_placement + (500)
                    )
                    simulation_engine.set_macro(
                        n_particles=1000000,
                        energy_keV=[500,500],
                        positions=[[xx, 0, -500],[xx-yy, 0, -500]],
                        #positions=[[xx, 0, -500]],
                        directions=[1,1],
                    )

                    fname = "../simulation_data/timepix_sim/prospectus_plots/fov/%d_%d_%.2f_%.2f_%.2f.csv" % (
                        thickness,
                        mask_config[1],
                        distance,
                        res_deg,
                        fov
                    )
                    if sim_count == 0:
                        simulation_engine.run_simulation(fname, build=True)
                    else:
                        simulation_engine.run_simulation(fname, build=False)

                    myhits = Hits(fname=fname, experiment=False)
                    myhits.get_det_hits()
                    deconvolver = Deconvolution(myhits, simulation_engine)
                    # res is changed from true or false to a peak location
                    res, _, _, _ = deconvolver.deconvolve(
                        plot_deconvolved_heatmap=True,
                        save_raw_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fov/%d_%d_%.2f_%.2f_raw.png" % (thickness, mask_config[1], distance, res_deg),
                        plot_raw_heatmap=False,
                        downsample=1,
                        trim=mask_config[3],
                        save_deconvolve_heatmap="../simulation_results/parameter_sweeps/prospectus_plots/fov/%d_%d_%.2f_%.2f_%.2f.png"
                        % (thickness, mask_config[1], distance, res_deg, fov),
                        save_peak="../simulation_results/parameter_sweeps/prospectus_plots/fov/%d_%d_%.2f_%.4f_%.2f_peak.png"
                        % (thickness, mask_config[1], distance, res_deg, fov),
                        plot_signal_peak=True,
                        plot_conditions=True,
                        check_resolved=True,
                        condition="half_val",
                    )
                    #print(peak_loc)
                    #if peak_loc < 100:
                    #    # moved over to the left
                    #    res = True
                    #    print('PEAK EXCEED FCFOV')
                    #else:
                    #    res = False

                    if res_deg > 0.28:
                        break

                    sim_count += 1
                    res_deg+= 0.001

                    #if res_deg > 50:
                    #    break

                file1 = open(
                    "../simulation_results/parameter_sweeps/prospectus_plots/fov/%d_%d_res_2.txt"
                    % (thickness, mask_config[1]),
                    "a",
                )  # append mode
                # write the last res deg before 
                file1.write("%.2f %.4f \n" % (fov, res_deg-0.001))
                file1.close()
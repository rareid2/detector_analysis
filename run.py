from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

simulation_engine = SimulationEngine(sim_type="ca-pt-source", write_files=False)
simulation_engine.set_config(
    det1_thickness_um=140, det_gap_mm=30, win_thickness_um=100, det_size_cm=6.3
)

simulation_engine.set_macro(n_particles=100000, energy_keV=1000)
# simulation_engine.run_simulation()

myhits = Hits(simulation_engine, "hits_67_3300_6_1.0_28.071.csv")
myhits.getDetHits()
# myhits.update_pos_uncertainty("Gaussian", 1)

# scattering = Scattering(myhits, simulation_engine, save_fig_directory="")
# scattering.get_thetas()
# scattering.get_theoretical_dist()
# scattering.plot_theoretical(save_fig=False)
# scattering.plot_compare_th_sim(save_fig=False)
# could put in a loop of energies and thicknesses and just save result - easy

deconvolution = Deconvolution(myhits, simulation_engine)
resolved = deconvolution.deconvolve(multiplier=12,plot_raw_heatmap=True,plot_deconvolved_heatmap=True,plot_signal_peak=True,plot_conditions=True,condition='quarter_val')
print(resolved)
from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering

simulation_engine = SimulationEngine(sim_type="two-detectors")
simulation_engine.write_config(
    det1_thickness_um=140, det_gap_mm=30, win_thickness_um=100
)
simulation_engine.write_macro(n_particles=100000, energy_keV=1000)
#simulation_engine.run_simulation()

myhits = Hits("/home/rileyannereid/workspace/geant4/data/hits.csv")
myhits.getBothDetHits()
#myhits.update_pos_uncertainty("Gaussian", 1)

scattering = Scattering(myhits, simulation_engine)
scattering.get_thetas()
scattering.get_theoretical_dist()
scattering.plot_theoretical(bin_size=5)
#scattering.plot_compare_th_sim(bin_size=2)


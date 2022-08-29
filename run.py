from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

# example run for coded aperture with point sources

# branch names:
# construct = CA and TD
# source = DS and PS

simulation_engine = SimulationEngine(construct="TD", source="PS", write_files=True)
simulation_engine.set_config(
    det1_thickness_um=140,
    det_gap_mm=30,
    win_thickness_um=100,
    det_size_cm=4.389,
)

simulation_engine.set_macro(n_particles=10000, energy_keV=5000)
# fname = "../data/hits_1e9_3_400_sin.csv"
fname = "../data/test.csv"
simulation_engine.run_simulation(fname, build=True)
myhits = Hits(simulation_engine, fname)
myhits.get_both_det_hits()

scattering = Scattering(myhits, simulation_engine)
scattering.get_thetas()
scattering.get_theoretical_dist()
scattering.plot_theoretical()
scattering.plot_compare_th_sim()

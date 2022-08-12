from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution

# example run for coded aperture with point sources

#branch names:
# ca-pt-source
# main
# two detectors


simulation_engine = SimulationEngine(sim_type="main", write_files=True)
simulation_engine.set_config(det1_thickness_um=140,
        det_gap_mm=30,
        win_thickness_um=100,
        det_size_cm=4.389,
        n_elements=133,
        mask_thickness_um=400,
        mask_gap_cm=3,
        element_size_mm=0.66)

simulation_engine.set_macro(n_particles=10e8,
        energy_keV=[500], PAD_run=1)
fname='../data/sin_attempt.csv'

simulation_engine.run_simulation(fname)

myhits = Hits(simulation_engine, fname)
myhits.get_det_hits()
Deconvolution(myhits, simulation_engine)